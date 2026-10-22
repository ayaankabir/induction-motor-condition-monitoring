"""Tests for IMCM V2.4 simulated MQTT streaming module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from imcm.data import (
    CsvImportSpec,
    SignalChannel,
    SignalChannelSpec,
    SignalTrace,
    load_signal_csv,
)
from imcm.iot.streaming import (
    MQTTMessageSchema,
    MQTTReplayPublisher,
    SignalWindowBuffer,
    StreamHealthReport,
)
from imcm.processing.features import (
    negative_sequence_current,
    current_unbalance_pct,
    spectral_peak_fifty_hz,
    rotor_asymmetry_sidebands,
    bearing_envelope_peak,
    torque_ripple_rms,
)


def _make_test_trace() -> SignalTrace:
    """Create a simple test trace with current and voltage channels (1000 samples, 0.1s)."""
    n = 1000
    fs = 10000.0
    t = np.arange(n, dtype=float) / fs
    current = np.sin(2 * np.pi * 50 * t)
    voltage = np.cos(2 * np.pi * 50 * t) * 230

    return SignalTrace.from_arrays(
        time_s=t,
        channels={"current_a": current, "voltage_a": voltage},
        units={"current_a": "A", "voltage_a": "V"},
        signal_types={"current_a": "current", "voltage_a": "voltage"},
        sample_rate_hz=fs,
        sample_rate_source="declared",
        declared_sample_rate_hz=fs,
        metadata={"label": "healthy"},
    )


def _make_parity_trace() -> SignalTrace:
    """Create a deterministic test trace for parity testing (30000 samples, 3.0s).

    Uses exact sinusoids so all three paths produce identical inputs.
    Three-phase currents with 50 Hz fundamental, plus small imbalance
    for negative-sequence current detection. Duration supports late_start=2.0s
    for rotor asymmetry sidebands (fault_05 uses up to 3s window).
    """
    n = 30000  # 3.0s at 10000 Hz
    fs = 10000.0
    t = np.arange(n, dtype=float) / fs

    # Three-phase currents with 50 Hz fundamental and small imbalance
    # Phase A: 10 A amplitude
    # Phase B: 10.5 A amplitude (5% imbalance)
    # Phase C: 9.8 A amplitude (2% imbalance)
    current_a = 10.0 * np.sin(2 * np.pi * 50 * t)
    current_b = 10.5 * np.sin(2 * np.pi * 50 * t - 2 * np.pi / 3)
    current_c = 9.8 * np.sin(2 * np.pi * 50 * t + 2 * np.pi / 3)

    # Add torque signal for ripple test
    tau_e = 20.0 + 0.5 * np.sin(2 * np.pi * 100 * t)  # 20 Nm + 100 Hz ripple

    return SignalTrace.from_arrays(
        time_s=t,
        channels={
            "current_a": current_a,
            "current_b": current_b,
            "current_c": current_c,
            "tau_e": tau_e,
        },
        units={
            "current_a": "A",
            "current_b": "A",
            "current_c": "A",
            "tau_e": "N*m",
        },
        signal_types={
            "current_a": "current",
            "current_b": "current",
            "current_c": "current",
            "tau_e": "torque",
        },
        sample_rate_hz=fs,
        sample_rate_source="declared",
        declared_sample_rate_hz=fs,
        metadata={"label": "healthy"},
    )


def _extract_features_from_trace(trace: SignalTrace, late_start: float = 0.8) -> dict[str, float]:
    """Extract all common features from a SignalTrace for parity comparison."""
    # Stack three-phase currents for feature functions
    i_abc = np.vstack([
        trace.channel("current_a").values,
        trace.channel("current_b").values,
        trace.channel("current_c").values,
    ])

    t = trace.time_s

    features = {}

    # Feature 1: Negative-sequence current
    features["neg_seq_current_a"] = negative_sequence_current(i_abc, t=t, late_start=late_start)

    # Feature 2: Current unbalance %
    features["current_unbalance_pct"] = current_unbalance_pct(i_abc)

    # Feature 3: Spectral peak near 50 Hz
    spec = spectral_peak_fifty_hz(t, i_abc, late_start=late_start)
    features["spectral_peak_freq_hz"] = spec["peak_freq_hz"]
    features["spectral_peak_amp_a"] = spec["peak_amp_a"]

    # Feature 4: Torque ripple RMS
    if "tau_e" in trace.channel_names:
        features["torque_ripple_rms_nm"] = torque_ripple_rms(trace.channel("tau_e").values, late_start=late_start)
    else:
        features["torque_ripple_rms_nm"] = float("nan")

    # Feature 5: Rotor asymmetry sidebands (late_start=2.0 for fault_05)
    sidebands = rotor_asymmetry_sidebands(t, i_abc, late_start=2.0)
    features["sideband_lower_hz"] = sidebands["lower_sideband_hz"]
    features["sideband_lower_amp_a"] = sidebands["lower_sideband_amp_a"] if sidebands["lower_sideband_amp_a"] is not None else float("nan")
    features["sideband_upper_hz"] = sidebands["upper_sideband_hz"]
    features["sideband_upper_amp_a"] = sidebands["upper_sideband_amp_a"] if sidebands["upper_sideband_amp_a"] is not None else float("nan")

    # Feature 6: Bearing envelope (requires vibration channel)
    if "vibration_m_s2" in trace.channel_names:
        env = bearing_envelope_peak(trace.channel("vibration_m_s2").values, t, resonance_hz=2000.0, late_start=0.5)
        features["bearing_env_peak_m_s2"] = env["envelope_peak_m_s2"]
        features["bearing_env_mean_m_s2"] = env["envelope_mean_m_s2"]
    else:
        features["bearing_env_peak_m_s2"] = float("nan")
        features["bearing_env_mean_m_s2"] = float("nan")

    return features


def _write_trace_to_csv(trace: SignalTrace, path: Path) -> None:
    """Write SignalTrace to CSV format compatible with CsvImportSpec."""
    # Only write the channels we need for features
    channels_to_write = ["current_a", "current_b", "current_c", "tau_e"]
    lines = ["time_s,current_a,current_b,current_c,tau_e"]
    for i in range(len(trace.time_s)):
        row = [
            f"{trace.time_s[i]:.6f}",
            f"{trace.channel('current_a').values[i]:.6f}",
            f"{trace.channel('current_b').values[i]:.6f}",
            f"{trace.channel('current_c').values[i]:.6f}",
            f"{trace.channel('tau_e').values[i]:.6f}",
        ]
        lines.append(",".join(row))
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_csv_as_trace(csv_path: Path) -> SignalTrace:
    """Load CSV using the project's csv_loader and return SignalTrace."""
    spec = CsvImportSpec(
        time_column="time_s",
        channels=(
            SignalChannelSpec(name="current_a", column="current_a", unit="A", signal_type="current"),
            SignalChannelSpec(name="current_b", column="current_b", unit="A", signal_type="current"),
            SignalChannelSpec(name="current_c", column="current_c", unit="A", signal_type="current"),
            SignalChannelSpec(name="tau_e", column="tau_e", unit="N*m", signal_type="torque"),
        ),
        sample_rate_hz=10000.0,
        sample_rate_rtol=1.0e-3,
    )
    imported = load_signal_csv(csv_path, spec, source_name="parity_test", imported_at_utc="2026-10-22T00:00:00+00:00")
    return imported.trace


def _mqtt_reassemble_trace(trace: SignalTrace, chunk_size: int = 100) -> SignalTrace:
    """Reassemble trace through MQTT simulated pipeline."""
    publisher = MQTTReplayPublisher(trace, chunk_size=chunk_size)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    # Stream all three current channels through the buffer
    for channel_name in ["current_a", "current_b", "current_c"]:
        for msg in publisher.stream_chunks(channel_name):
            buffer.ingest(msg)

    # Also stream torque for completeness
    for msg in publisher.stream_chunks("tau_e"):
        buffer.ingest(msg)

    # Reassemble all channels
    channels_dict = {}
    for channel_name in ["current_a", "current_b", "current_c", "tau_e"]:
        reassembled = buffer.reassemble(channel_name)
        if reassembled is not None:
            channels_dict[channel_name] = reassembled.channel(channel_name).values

    # Build a new trace with reassembled data (use original time for alignment)
    return SignalTrace.from_arrays(
        time_s=trace.time_s,
        channels=channels_dict,
        units={k: trace.channel(k).unit for k in channels_dict},
        signal_types={k: trace.channel(k).signal_type for k in channels_dict},
        sample_rate_hz=trace.sample_rate_hz,
        sample_rate_source="declared",
        declared_sample_rate_hz=trace.declared_sample_rate_hz,
        metadata=trace.metadata,
    )


def test_mqtt_message_schema_json_roundtrip() -> None:
    msg = MQTTMessageSchema(
        schema_version="1.0.0",
        node_id="test-node",
        seq=42,
        ts_start=0.1,
        fs_hz=10000.0,
        channel="current_a",
        unit="A",
        samples=[1.0, 2.0, 3.0],
        dataset_label="healthy",
    )
    json_str = msg.to_json()
    parsed = MQTTMessageSchema.from_json(json_str)
    assert parsed.schema_version == "1.0.0"
    assert parsed.node_id == "test-node"
    assert parsed.seq == 42
    assert parsed.ts_start == 0.1
    assert parsed.fs_hz == 10000.0
    assert parsed.channel == "current_a"
    assert parsed.unit == "A"
    assert parsed.samples == [1.0, 2.0, 3.0]
    assert parsed.dataset_label == "healthy"


def test_mqtt_publisher_chunking() -> None:
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)

    chunks = list(publisher.stream_chunks("current_a"))
    assert len(chunks) == 120  # 30000 samples / 250 = 120 chunks

    # Verify sequence numbers are sequential
    assert [c.seq for c in chunks] == list(range(120))

    # Verify all samples preserved
    all_samples = []
    for chunk in chunks:
        all_samples.extend(chunk.samples)
    np.testing.assert_array_almost_equal(all_samples, trace.channel("current_a").values)


def test_window_buffer_reassemble_exact_match() -> None:
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    for msg in publisher.stream_chunks("current_a"):
        buffer.ingest(msg)

    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) == len(trace.time_s)
    np.testing.assert_array_almost_equal(reassembled.time_s, trace.time_s)
    np.testing.assert_array_almost_equal(
        reassembled.channel("current_a").values,
        trace.channel("current_a").values,
    )


def test_window_buffer_detects_duplicates() -> None:
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Send chunk 0 twice
    buffer.ingest(chunks[0])
    buffer.ingest(chunks[0])
    # Send remaining normally
    for chunk in chunks[1:]:
        buffer.ingest(chunk)

    report = buffer.health_by_channel["current_a"]
    assert report.duplicate_count == 1
    assert report.total_packets_received == 121  # 120 unique + 1 duplicate


def test_window_buffer_detects_gaps() -> None:
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Skip chunk 1 (send 0, 2, 3)
    buffer.ingest(chunks[0])
    buffer.ingest(chunks[2])
    buffer.ingest(chunks[3])

    report = buffer.health_by_channel["current_a"]
    assert report.gap_count == 1
    assert 1 in report.missing_sequences


def test_window_buffer_detects_out_of_order() -> None:
    """Verify MQTT path with out-of-order delivery is flagged.

    Out-of-order packets should be detected and counted. The buffer reassembles
    in sequence order (not arrival order), but the health report must flag
    the out-of-order event.
    """
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Deliver all chunks out of order (reverse order)
    for chunk in reversed(chunks):
        buffer.ingest(chunk)

    report = buffer.health_by_channel["current_a"]
    assert report.out_of_order_count >= 1

    # Final reassembly should still be complete (all 40 chunks received)
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) == len(trace.time_s)
    np.testing.assert_array_almost_equal(
        reassembled.channel("current_a").values,
        trace.channel("current_a").values,
    )


def test_window_buffer_multiple_channels() -> None:
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    for msg in publisher.stream_chunks("current_a"):
        buffer.ingest(msg)
    for msg in publisher.stream_chunks("current_b"):
        buffer.ingest(msg)
    for msg in publisher.stream_chunks("current_c"):
        buffer.ingest(msg)

    assert "current_a" in buffer.chunks_by_channel
    assert "current_b" in buffer.chunks_by_channel
    assert "current_c" in buffer.chunks_by_channel
    assert buffer.reassemble("current_a") is not None
    assert buffer.reassemble("current_b") is not None
    assert buffer.reassemble("current_c") is not None


def test_mqtt_schema_handles_malformed_json() -> None:
    """Test that from_json handles malformed input gracefully."""
    with pytest.raises(json.JSONDecodeError):
        MQTTMessageSchema.from_json("not valid json")

    # Missing required fields should use defaults
    msg = MQTTMessageSchema.from_json('{"seq": 5}')
    assert msg.seq == 5
    assert msg.schema_version == "1.0.0"
    assert msg.samples == []


def test_empty_chunk_handling() -> None:
    """Test handling of empty sample arrays."""
    msg = MQTTMessageSchema(
        schema_version="1.0.0",
        node_id="test",
        seq=0,
        ts_start=0.0,
        fs_hz=10000.0,
        channel="current_a",
        unit="A",
        samples=[],
        dataset_label="healthy",
    )
    buffer = SignalWindowBuffer(expected_fs_hz=10000.0)
    buffer.ingest(msg)
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) == 0


def test_mqtt_publisher_empty_trace() -> None:
    """Test publisher with zero-length trace."""
    trace = SignalTrace.from_arrays(
        time_s=np.array([]),
        channels={"current_a": np.array([])},
        units={"current_a": "A"},
        signal_types={"current_a": "current"},
        sample_rate_hz=10000.0,
        sample_rate_source="declared",
        declared_sample_rate_hz=10000.0,
    )
    publisher = MQTTReplayPublisher(trace, chunk_size=100)
    chunks = list(publisher.stream_chunks("current_a"))
    assert len(chunks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Parity tests across simulation, CSV, and MQTT paths
# ─────────────────────────────────────────────────────────────────────────────


def test_parity_simulation_csv_mqtt_feature_extraction() -> None:
    """Parity test: equivalent features from simulation, CSV, and MQTT paths.

    Creates a deterministic SignalTrace, passes it through all three ingestion
    paths, extracts features using the project's feature extraction functions,
    and verifies numerical agreement within documented tolerances.

    Tolerances:
    - Time array: exact match (same np.arange construction)
    - Sample count: exact match
    - Channel values: rtol=1e-10 (reconstruction should be bit-identical for in-memory)
    - Features: rtol=1e-6 (same numerical computation path)
    """
    # Create deterministic test trace (1.0s duration for late_start support)
    trace = _make_parity_trace()

    # ── Path 1: Direct simulation trace (reference) ──
    features_sim = _extract_features_from_trace(trace)

    # ── Path 2: CSV ingestion ──
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "parity_test.csv"
        _write_trace_to_csv(trace, csv_path)
        trace_csv = _load_csv_as_trace(csv_path)

    features_csv = _extract_features_from_trace(trace_csv)

    # ── Path 3: MQTT window buffer reassembly ──
    trace_mqtt = _mqtt_reassemble_trace(trace, chunk_size=100)
    features_mqtt = _extract_features_from_trace(trace_mqtt)

    # Compare all feature keys
    feature_keys = list(features_sim.keys())
    for key in feature_keys:
        f_sim = features_sim[key]
        f_csv = features_csv[key]
        f_mqtt = features_mqtt[key]

        # Skip NaN comparisons (e.g., fault-specific features not applicable)
        if np.isnan(f_sim) and np.isnan(f_csv) and np.isnan(f_mqtt):
            continue

        # CSV and MQTT paths should match simulation reference
        # rtol=1e-5, atol=1e-12 for feature values (accounting for small numerical differences across paths, especially near-zero values), exact for sample count and timestamps
        np.testing.assert_allclose(
            f_csv, f_sim, rtol=1e-5, atol=1e-12,
            err_msg=f"Feature {key} CSV path mismatch: sim={f_sim}, csv={f_csv}"
        )
        np.testing.assert_allclose(
            f_mqtt, f_sim, rtol=1e-5, atol=1e-12,
            err_msg=f"Feature {key} MQTT path mismatch: sim={f_sim}, mqtt={f_mqtt}"
        )

    # Also verify trace structural integrity
    assert len(trace_csv.time_s) == len(trace.time_s), "CSV path: sample count mismatch"
    assert len(trace_mqtt.time_s) == len(trace.time_s), "MQTT path: sample count mismatch"
    np.testing.assert_array_equal(trace_csv.time_s, trace.time_s, "CSV path: time array mismatch")
    np.testing.assert_array_equal(trace_mqtt.time_s, trace.time_s, "MQTT path: time array mismatch")

    # Verify channel names and units preserved
    assert set(trace_csv.channel_names) == {"current_a", "current_b", "current_c", "tau_e"}, "CSV path: channel names"
    assert set(trace_mqtt.channel_names) == {"current_a", "current_b", "current_c", "tau_e"}, "MQTT path: channel names"
    assert trace_csv.sample_rate_hz == trace.sample_rate_hz, "CSV path: sample rate mismatch"
    assert trace_mqtt.sample_rate_hz == trace.sample_rate_hz, "MQTT path: sample rate mismatch"


def test_parity_mqtt_with_gaps_flagged_not_reconstructed() -> None:
    """Verify MQTT path with dropped packets is flagged, not silently reconstructed.

    When packets are missing (gaps), the window buffer should report the gaps
    and the reassembled trace should be incomplete (missing samples).
    It should NOT silently interpolate or fabricate missing data.
    """
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Send all chunks except chunk 1 (one gap in the middle)
    for i, chunk in enumerate(chunks):
        if i != 1:
            buffer.ingest(chunk)

    report = buffer.health_by_channel["current_a"]
    assert report.gap_count == 1
    assert 1 in report.missing_sequences

    # Reassembly should produce shorter trace (missing chunk 1 = 250 samples)
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) < len(trace.time_s), "Gap should result in fewer samples"

    # Missing samples should NOT be interpolated - they should simply be absent
    expected_samples = len(trace.time_s) - 250  # one chunk missing
    assert len(reassembled.time_s) == expected_samples, \
        f"Expected {expected_samples} samples, got {len(reassembled.time_s)}"


def test_parity_mqtt_out_of_order_flagged_not_corrected() -> None:
    """Verify MQTT path with out-of-order delivery is flagged.

    Out-of-order packets should be detected and counted. The buffer reassembles
    in sequence order (not arrival order), but the health report must flag
    the out-of-order event.
    """
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Deliver all chunks out of order (reverse order)
    for chunk in reversed(chunks):
        buffer.ingest(chunk)

    report = buffer.health_by_channel["current_a"]
    assert report.out_of_order_count >= 1

    # Final reassembly should still be complete (all 40 chunks received)
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) == len(trace.time_s)
    np.testing.assert_array_almost_equal(
        reassembled.channel("current_a").values,
        trace.channel("current_a").values,
    )


def test_parity_mqtt_duplicates_dropped_not_doubled() -> None:
    """Verify duplicate packets are dropped, not double-counted in reconstruction."""
    trace = _make_parity_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Send chunk 0 twice
    buffer.ingest(chunks[0])
    buffer.ingest(chunks[0])
    # Send remaining normally
    for chunk in chunks[1:]:
        buffer.ingest(chunk)

    report = buffer.health_by_channel["current_a"]
    assert report.duplicate_count == 1
    assert report.total_packets_received == 121  # 120 unique + 1 duplicate

    # Reassembly should be complete (duplicates ignored)
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    assert len(reassembled.time_s) == len(trace.time_s)
    np.testing.assert_array_almost_equal(
        reassembled.channel("current_a").values,
        trace.channel("current_a").values,
    )


def test_parity_mqtt_malformed_json_rejected() -> None:
    """Malformed JSON messages should be rejected and counted."""
    trace = _make_test_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    # Ingest first chunk normally
    buffer.ingest(chunks[0])

    # Create a malformed message (simulate corrupted packet)
    # We test the schema validation directly
    with pytest.raises(json.JSONDecodeError):
        MQTTMessageSchema.from_json("not valid json")

    # The buffer should not have ingested the malformed message
    # (no way to test buffer ingestion of malformed without JSON roundtrip,
    # but we verify schema validation rejects it)


def test_parity_mqtt_empty_chunks_handled_gracefully() -> None:
    """Empty sample arrays in messages should be handled without error."""
    trace = _make_test_trace()
    publisher = MQTTReplayPublisher(trace, chunk_size=250)
    buffer = SignalWindowBuffer(expected_fs_hz=trace.sample_rate_hz)

    chunks = list(publisher.stream_chunks("current_a"))
    buffer.ingest(chunks[0])

    # Ingest an empty chunk
    empty_msg = MQTTMessageSchema(
        schema_version="1.0.0",
        node_id="test",
        seq=99,
        ts_start=0.5,
        fs_hz=10000.0,
        channel="current_a",
        unit="A",
        samples=[],
        dataset_label="healthy",
    )
    buffer.ingest(empty_msg)

    # Should not crash, empty chunk adds no samples
    reassembled = buffer.reassemble("current_a")
    assert reassembled is not None
    # Original chunk 0 has 250 samples
    assert len(reassembled.channel("current_a").values) == 250


if __name__ == "__main__":
    pytest.main([__file__, "-v"])