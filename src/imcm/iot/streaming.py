"""Simulated MQTT streaming module for IMCM V2.4.

Provides message schema validation, chunking simulation traces, buffer reassembly,
and gap/duplicate/out-of-order detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence
import numpy as np

from imcm.data import SignalChannel, SignalTrace


@dataclass
class MQTTMessageSchema:
    """Versioned JSON message schema for simulated telemetry chunks."""
    schema_version: str = "1.0.0"
    node_id: str = "motor-sensor-node-01"
    seq: int = 0
    ts_start: float = 0.0
    fs_hz: float = 10000.0
    channel: str = "current_a"
    unit: str = "A"
    samples: list[float] = field(default_factory=list)
    dataset_label: str = "healthy"

    def to_json(self) -> str:
        return json.dumps({
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "seq": self.seq,
            "ts_start": self.ts_start,
            "fs_hz": self.fs_hz,
            "channel": self.channel,
            "unit": self.unit,
            "samples": self.samples,
            "dataset_label": self.dataset_label,
        })

    @classmethod
    def from_json(cls, payload: str | bytes) -> MQTTMessageSchema:
        data = json.loads(payload)
        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            node_id=data.get("node_id", "motor-sensor-node-01"),
            seq=int(data.get("seq", 0)),
            ts_start=float(data.get("ts_start", 0.0)),
            fs_hz=float(data.get("fs_hz", 10000.0)),
            channel=data.get("channel", "current_a"),
            unit=data.get("unit", "A"),
            samples=[float(s) for s in data.get("samples", [])],
            dataset_label=data.get("dataset_label", "healthy"),
        )


class MQTTReplayPublisher:
    """Publishes simulation trace channels as chunked JSON messages."""

    def __init__(
        self,
        trace: SignalTrace,
        chunk_size: int = 100,
        node_id: str = "motor-sensor-node-01",
    ) -> None:
        self.trace = trace
        self.chunk_size = chunk_size
        self.node_id = node_id

    def stream_chunks(self, channel_name: str) -> Iterator[MQTTMessageSchema]:
        channel = self.trace.channel(channel_name)
        values = channel.values
        time_s = self.trace.time_s
        fs = self.trace.sample_rate_hz

        n = len(values)
        seq = 0
        for start_idx in range(0, n, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n)
            chunk_vals = values[start_idx:end_idx].tolist()
            t_start = float(time_s[start_idx])

            msg = MQTTMessageSchema(
                schema_version="1.0.0",
                node_id=self.node_id,
                seq=seq,
                ts_start=t_start,
                fs_hz=float(fs),
                channel=channel_name,
                unit=channel.unit,
                samples=chunk_vals,
                dataset_label=self.trace.metadata.get("label", "healthy"),
            )
            yield msg
            seq += 1


@dataclass
class StreamHealthReport:
    """Report on stream quality, gaps, duplicates, and out-of-order packets."""
    total_packets_received: int = 0
    duplicate_count: int = 0
    gap_count: int = 0
    out_of_order_count: int = 0
    malformed_count: int = 0
    min_seq: int = 0
    max_seq: int = -1
    missing_sequences: list[int] = field(default_factory=list)


class SignalWindowBuffer:
    """Reassembles incoming telemetry chunks into a continuous SignalTrace."""

    def __init__(self, expected_fs_hz: float = 10000.0) -> None:
        self.expected_fs_hz = expected_fs_hz
        self.chunks_by_channel: dict[str, dict[int, list[tuple[float, list[float]]]]] = {}
        self.health_by_channel: dict[str, StreamHealthReport] = {}

    def ingest(self, msg: MQTTMessageSchema) -> None:
        ch = msg.channel
        if ch not in self.chunks_by_channel:
            self.chunks_by_channel[ch] = {}
            self.health_by_channel[ch] = StreamHealthReport()

        report = self.health_by_channel[ch]
        report.total_packets_received += 1

        seq = msg.seq
        if report.total_packets_received == 1:
            report.min_seq = seq
            report.max_seq = seq
        else:
            if seq < report.max_seq:
                report.out_of_order_count += 1
            if seq == report.max_seq:
                report.duplicate_count += 1
                return
            if seq > report.max_seq + 1:
                report.gap_count += 1
                for missing in range(report.max_seq + 1, seq):
                    report.missing_sequences.append(missing)
            if seq > report.max_seq:
                report.max_seq = seq

        if seq in self.chunks_by_channel[ch]:
            report.duplicate_count += 1
            return

        self.chunks_by_channel[ch][seq] = (msg.ts_start, msg.samples)

    def reassemble(self, channel_name: str) -> SignalTrace | None:
        if channel_name not in self.chunks_by_channel:
            return None

        chunks_dict = self.chunks_by_channel[channel_name]
        if not chunks_dict:
            return None

        sorted_seqs = sorted(chunks_dict.keys())
        all_time: list[float] = []
        all_vals: list[float] = []

        dt = 1.0 / self.expected_fs_hz
        for seq in sorted_seqs:
            t_start, samples = chunks_dict[seq]
            if len(samples) == 0:
                continue
            for i, val in enumerate(samples):
                all_time.append(t_start + i * dt)
                all_vals.append(val)

        time_arr = np.asarray(all_time, dtype=float)
        val_arr = np.asarray(all_vals, dtype=float)

        channel = SignalChannel(
            name=channel_name,
            values=val_arr,
            unit="A" if "current" in channel_name else "V",
            source_column=channel_name,
            signal_type="current" if "current" in channel_name else "voltage",
        )

        return SignalTrace.from_arrays(
            time_s=time_arr,
            channels={channel_name: val_arr},
            units={channel_name: channel.unit},
            signal_types={channel_name: channel.signal_type},
            sample_rate_hz=self.expected_fs_hz,
            sample_rate_source="declared",
            declared_sample_rate_hz=self.expected_fs_hz,
        )
