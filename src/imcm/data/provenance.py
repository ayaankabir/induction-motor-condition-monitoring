"""Provenance and summary records for unvalidated external signal input.

External CSV data is intentionally kept outside the simulation provenance
schema.  It is not automatically labelled ``simulated`` or ``experimental``,
because this project has not established the source, calibration, or hardware
validity of an imported file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Literal, Mapping

from imcm.data.trace import CANONICAL_UNITS, SignalTrace

EXTERNAL_PROVENANCE_SCHEMA_VERSION = "external_signal_provenance_v1"
EXTERNAL_SOURCE_TYPE = "external_csv"
EXTERNAL_PROVENANCE_CLASS = "external_unvalidated"
EXTERNAL_VALIDATION_STATUS = "unvalidated_external_input"
EXTERNAL_PARSER_VERSION = "imcm_csv_loader_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for an import record."""
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Any) -> str:
    """Return the SHA-256 digest of a local file without loading it all at once."""
    from pathlib import Path

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _require_mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _validate_timestamp(value: object) -> str:
    text = _require_text("imported_at_utc", value)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError("imported_at_utc must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("imported_at_utc must include a timezone")
    return text


def _validate_hash(value: object) -> str:
    text = _require_text("source_sha256", value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError("source_sha256 must be a 64-character hexadecimal SHA-256 digest")
    return text


@dataclass(frozen=True)
class ExternalSignalProvenance:
    """Self-describing, deliberately conservative provenance for a CSV file."""

    source_name: str
    source_sha256: str
    imported_at_utc: str
    time_column: str
    channel_mapping: Mapping[str, str]
    units: Mapping[str, str]
    sample_rate_hz: float
    sample_rate_source: Literal["declared", "derived_from_time"]
    declared_sample_rate_hz: float | None
    row_count: int
    parser_version: str = EXTERNAL_PARSER_VERSION
    schema_version: str = EXTERNAL_PROVENANCE_SCHEMA_VERSION
    source_type: str = EXTERNAL_SOURCE_TYPE
    provenance_class: str = EXTERNAL_PROVENANCE_CLASS
    validation_status: str = EXTERNAL_VALIDATION_STATUS
    experimental_validation: bool = False
    reliability_note: str = (
        "Imported external input is unvalidated. It is not simulated, not "
        "experimental validation, not calibrated, and not evidence of hardware "
        "validity or a diagnosis."
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_name", _require_text("source_name", self.source_name))
        object.__setattr__(self, "source_sha256", _validate_hash(self.source_sha256))
        object.__setattr__(self, "imported_at_utc", _validate_timestamp(self.imported_at_utc))
        object.__setattr__(self, "time_column", _require_text("time_column", self.time_column))
        channel_mapping = {
            _require_text("channel_mapping key", name): _require_text(
                f"channel_mapping[{name!r}]", column
            )
            for name, column in _require_mapping(
                "channel_mapping", self.channel_mapping
            ).items()
        }
        units = {
            _require_text("units key", name): _require_text(
                f"units[{name!r}]", unit
            )
            for name, unit in _require_mapping("units", self.units).items()
        }
        object.__setattr__(self, "channel_mapping", channel_mapping)
        object.__setattr__(self, "units", units)
        if not channel_mapping:
            raise ValueError("channel_mapping must not be empty")
        if set(channel_mapping) != set(units):
            raise ValueError("units must describe exactly the mapped channels")
        if len(set(channel_mapping.values())) != len(channel_mapping):
            raise ValueError("channel_mapping source columns must be unique")
        unsupported_units = sorted(set(units.values()) - CANONICAL_UNITS)
        if unsupported_units:
            allowed = ", ".join(sorted(CANONICAL_UNITS))
            raise ValueError(
                f"unsupported unit(s) {', '.join(repr(unit) for unit in unsupported_units)}; "
                f"expected canonical units: {allowed}"
            )
        if not isinstance(self.sample_rate_hz, (int, float)) or isinstance(self.sample_rate_hz, bool):
            raise ValueError("sample_rate_hz must be a positive finite number")
        if not float(self.sample_rate_hz) > 0.0 or not np.isfinite(self.sample_rate_hz):
            raise ValueError("sample_rate_hz must be a positive finite number")
        if self.sample_rate_source not in {"declared", "derived_from_time"}:
            raise ValueError("invalid sample_rate_source")
        if self.declared_sample_rate_hz is not None:
            if (
                not isinstance(self.declared_sample_rate_hz, (int, float))
                or isinstance(self.declared_sample_rate_hz, bool)
                or not float(self.declared_sample_rate_hz) > 0.0
                or not np.isfinite(self.declared_sample_rate_hz)
            ):
                raise ValueError("declared_sample_rate_hz must be positive and finite")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int) or self.row_count < 1:
            raise ValueError("row_count must be a positive integer")
        object.__setattr__(self, "parser_version", _require_text("parser_version", self.parser_version))
        object.__setattr__(self, "schema_version", _require_text("schema_version", self.schema_version))
        if self.source_type != EXTERNAL_SOURCE_TYPE:
            raise ValueError(f"source_type must be {EXTERNAL_SOURCE_TYPE!r}")
        if self.provenance_class != EXTERNAL_PROVENANCE_CLASS:
            raise ValueError(f"provenance_class must be {EXTERNAL_PROVENANCE_CLASS!r}")
        if self.validation_status != EXTERNAL_VALIDATION_STATUS:
            raise ValueError(f"validation_status must be {EXTERNAL_VALIDATION_STATUS!r}")
        if self.experimental_validation is not False:
            raise ValueError("external input cannot be marked experimental_validation=True")
        object.__setattr__(self, "reliability_note", _require_text("reliability_note", self.reliability_note))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe provenance mapping without waveform samples."""
        return {
            "schema_version": self.schema_version,
            "source_type": self.source_type,
            "provenance_class": self.provenance_class,
            "validation_status": self.validation_status,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "imported_at_utc": self.imported_at_utc,
            "parser_version": self.parser_version,
            "time_column": self.time_column,
            "channel_mapping": dict(self.channel_mapping),
            "units": dict(self.units),
            "time_unit": "s",
            "sample_rate_hz": float(self.sample_rate_hz),
            "sample_rate_source": self.sample_rate_source,
            "declared_sample_rate_hz": (
                None
                if self.declared_sample_rate_hz is None
                else float(self.declared_sample_rate_hz)
            ),
            "row_count": self.row_count,
            "experimental_validation": False,
            "reliability_note": self.reliability_note,
        }


@dataclass(frozen=True)
class ImportedSignal:
    """A source trace plus its external-input provenance."""

    trace: SignalTrace
    provenance: ExternalSignalProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.trace, SignalTrace):
            raise TypeError("trace must be a SignalTrace")
        if not isinstance(self.provenance, ExternalSignalProvenance):
            raise TypeError("provenance must be ExternalSignalProvenance")
        if self.trace.sample_count != self.provenance.row_count:
            raise ValueError("provenance row_count must match trace sample count")
        if self.trace.channel_names != tuple(self.provenance.channel_mapping):
            raise ValueError("provenance channel mapping must match trace channel order")
        expected_mapping = {
            channel.name: channel.source_column for channel in self.trace.channels
        }
        if dict(self.provenance.channel_mapping) != expected_mapping:
            raise ValueError("provenance channel mapping must match trace source columns")
        expected_units = {channel.name: channel.unit for channel in self.trace.channels}
        if dict(self.provenance.units) != expected_units:
            raise ValueError("provenance units must match trace channel units")
        if self.trace.sample_rate_hz != self.provenance.sample_rate_hz:
            raise ValueError("provenance sample rate must match trace sample rate")
        if self.trace.sample_rate_source != self.provenance.sample_rate_source:
            raise ValueError("provenance sample-rate source must match trace")
        if self.trace.declared_sample_rate_hz != self.provenance.declared_sample_rate_hz:
            raise ValueError("provenance declared sample rate must match trace")

    def to_summary_dict(self) -> dict[str, Any]:
        """Return metadata and quality-neutral trace information, not samples."""
        return {
            "provenance": self.provenance.to_dict(),
            "trace": {
                "sample_count": self.trace.sample_count,
                "duration_s": (
                    float(self.trace.time_s[-1] - self.trace.time_s[0])
                    if self.trace.sample_count
                    else 0.0
                ),
                "sample_rate_hz": float(self.trace.sample_rate_hz),
                "sample_rate_source": self.trace.sample_rate_source,
                "declared_sample_rate_hz": self.trace.declared_sample_rate_hz,
                "time_unit": self.trace.time_unit,
                "channels": [
                    {
                        "name": channel.name,
                        "unit": channel.unit,
                        "signal_type": channel.signal_type,
                        "source_column": channel.source_column,
                        "clipping_bounds": channel.clipping_bounds,
                    }
                    for channel in self.trace.channels
                ],
            },
        }


# Imported lazily by the dataclass module without making the public import
# surface depend on NumPy at provenance-only call sites.
import numpy as np  # noqa: E402  (kept after helpers for a compact standard API)

__all__ = [
    "EXTERNAL_PARSER_VERSION",
    "EXTERNAL_PROVENANCE_CLASS",
    "EXTERNAL_PROVENANCE_SCHEMA_VERSION",
    "EXTERNAL_SOURCE_TYPE",
    "EXTERNAL_VALIDATION_STATUS",
    "ExternalSignalProvenance",
    "ImportedSignal",
    "file_sha256",
    "utc_now_iso",
]