"""Source-agnostic time-series structures for imported and generated signals.

The structures in this module deliberately do not describe an induction-motor
model.  They provide a small, explicit representation that can hold a CSV
trace or arrays extracted from a generated simulation without changing the
meaning of the existing simulation result classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Mapping

import numpy as np

# Canonical units accepted by the first external-signal ingestion contract.
# The importer never converts units; callers must provide one of these exact
# spellings explicitly.
CANONICAL_UNITS: frozenset[str] = frozenset(
    {
        "A",
        "V",
        "m/s^2",
        "rad/s",
        "rpm",
        "N*m",
        "s",
        "Hz",
        "dimensionless",
    }
)

CANONICAL_SIGNAL_TYPES: frozenset[str] = frozenset(
    {"current", "voltage", "vibration", "speed", "torque", "time", "other"}
)

SampleRateSource = Literal["declared", "derived_from_time"]


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _require_unit(value: object) -> str:
    unit = _require_text("unit", value)
    if unit not in CANONICAL_UNITS:
        allowed = ", ".join(sorted(CANONICAL_UNITS))
        raise ValueError(
            f"unsupported unit {unit!r}; expected one of the canonical units: {allowed}"
        )
    return unit


def _require_signal_type(value: object) -> str:
    signal_type = _require_text("signal_type", value).lower()
    if signal_type not in CANONICAL_SIGNAL_TYPES:
        allowed = ", ".join(sorted(CANONICAL_SIGNAL_TYPES))
        raise ValueError(
            f"unsupported signal_type {value!r}; expected one of: {allowed}"
        )
    return signal_type


def _require_positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_bounds(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError("clipping_bounds must be a (low, high) pair or None")
    if len(value) != 2:
        raise ValueError("clipping_bounds must be a (low, high) pair")
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("clipping_bounds must contain finite numbers") from exc
    if not isfinite(low) or not isfinite(high) or not low < high:
        raise ValueError("clipping_bounds must be finite and satisfy low < high")
    return low, high


@dataclass(frozen=True)
class SignalChannel:
    """One named scalar channel in a source-agnostic trace."""

    name: str
    values: np.ndarray
    unit: str
    source_column: str
    signal_type: str = "other"
    clipping_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "source_column", _require_text("source_column", self.source_column))
        object.__setattr__(self, "unit", _require_unit(self.unit))
        object.__setattr__(self, "signal_type", _require_signal_type(self.signal_type))
        object.__setattr__(self, "clipping_bounds", _validate_bounds(self.clipping_bounds))
        try:
            values = np.asarray(self.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel values must be numeric") from exc
        if values.ndim != 1:
            raise ValueError(f"channel {self.name!r} values must be one-dimensional")
        # Copy so a caller cannot mutate the object after construction.  NaN and
        # Inf are intentionally retained; the quality layer reports them.
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class SignalChannelSpec:
    """Declarative mapping and unit contract for one CSV channel."""

    name: str
    column: str
    unit: str
    signal_type: str = "other"
    clipping_bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text("name", self.name))
        object.__setattr__(self, "column", _require_text("column", self.column))
        object.__setattr__(self, "unit", _require_unit(self.unit))
        object.__setattr__(self, "signal_type", _require_signal_type(self.signal_type))
        object.__setattr__(self, "clipping_bounds", _validate_bounds(self.clipping_bounds))


@dataclass(frozen=True)
class SignalTrace:
    """A time vector and equally sampled scalar channels.

    This class is intentionally independent of ``HealthySimulationResult``.  It
    can wrap imported CSV data or arrays extracted from a generated run, but it
    never implies a solver, motor parameters, or a condition label.
    """

    time_s: np.ndarray
    channels: tuple[SignalChannel, ...]
    sample_rate_hz: float
    sample_rate_source: SampleRateSource
    declared_sample_rate_hz: float | None = None
    time_unit: Literal["s"] = "s"
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        try:
            time_s = np.asarray(self.time_s, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_s must be numeric") from exc
        if time_s.ndim != 1:
            raise ValueError("time_s must be one-dimensional")
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("a signal trace must contain at least one channel")
        if not all(isinstance(channel, SignalChannel) for channel in channels):
            raise ValueError("channels must contain only SignalChannel instances")
        names = [channel.name for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        expected_length = time_s.size
        for channel in channels:
            if channel.values.size != expected_length:
                raise ValueError(
                    f"channel {channel.name!r} has {channel.values.size} samples; "
                    f"expected {expected_length}"
                )
        sample_rate = _require_positive_finite("sample_rate_hz", self.sample_rate_hz)
        if self.sample_rate_source not in {"declared", "derived_from_time"}:
            raise ValueError(
                "sample_rate_source must be 'declared' or 'derived_from_time'"
            )
        if self.declared_sample_rate_hz is not None:
            declared = _require_positive_finite(
                "declared_sample_rate_hz", self.declared_sample_rate_hz
            )
            object.__setattr__(self, "declared_sample_rate_hz", declared)
        if self.time_unit != "s":
            raise ValueError("time_unit must be 's'; automatic time conversion is not supported")
        if self.metadata is None:
            metadata: Mapping[str, Any] = {}
        elif not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping or None")
        else:
            metadata = dict(self.metadata)
        object.__setattr__(self, "time_s", time_s.copy())
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "metadata", metadata)

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    def channel(self, name: str) -> SignalChannel:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"unknown signal channel: {name!r}")

    @classmethod
    def from_arrays(
        cls,
        time_s: np.ndarray,
        channels: Mapping[str, np.ndarray],
        *,
        units: Mapping[str, str],
        signal_types: Mapping[str, str] | None = None,
        sample_rate_hz: float,
        sample_rate_source: SampleRateSource = "declared",
        declared_sample_rate_hz: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SignalTrace":
        """Construct a trace from named arrays using the same unit contract."""
        if not channels:
            raise ValueError("channels must not be empty")
        if set(units) != set(channels):
            raise ValueError("units must provide exactly one unit per channel")
        signal_types = signal_types or {}
        built = tuple(
            SignalChannel(
                name=name,
                values=values,
                unit=units[name],
                source_column=name,
                signal_type=signal_types.get(name, "other"),
            )
            for name, values in channels.items()
        )
        return cls(
            time_s=time_s,
            channels=built,
            sample_rate_hz=sample_rate_hz,
            sample_rate_source=sample_rate_source,
            declared_sample_rate_hz=declared_sample_rate_hz,
            metadata=metadata,
        )


@dataclass(frozen=True)
class CsvImportSpec:
    """Strict wide-format CSV import contract."""

    time_column: str
    channels: tuple[SignalChannelSpec, ...]
    time_unit: Literal["s"] = "s"
    sample_rate_hz: float | None = None
    sample_rate_rtol: float = 1.0e-3
    delimiter: str = ","

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_column", _require_text("time_column", self.time_column))
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("CsvImportSpec must contain at least one channel")
        if not all(isinstance(channel, SignalChannelSpec) for channel in channels):
            raise ValueError("channels must contain only SignalChannelSpec instances")
        names = [channel.name for channel in channels]
        columns = [channel.column for channel in channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")
        if len(set(columns)) != len(columns):
            raise ValueError("channel columns must be unique")
        object.__setattr__(self, "channels", channels)
        if self.time_unit != "s":
            raise ValueError("time_unit must be 's'; automatic time conversion is not supported")
        if self.sample_rate_hz is not None:
            object.__setattr__(
                self,
                "sample_rate_hz",
                _require_positive_finite("sample_rate_hz", self.sample_rate_hz),
            )
        if (
            isinstance(self.sample_rate_rtol, bool)
            or not isinstance(self.sample_rate_rtol, (int, float))
            or not isfinite(float(self.sample_rate_rtol))
            or float(self.sample_rate_rtol) <= 0.0
        ):
            raise ValueError("sample_rate_rtol must be a positive finite number")
        object.__setattr__(self, "sample_rate_rtol", float(self.sample_rate_rtol))
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character")


__all__ = [
    "CANONICAL_SIGNAL_TYPES",
    "CANONICAL_UNITS",
    "CsvImportSpec",
    "SampleRateSource",
    "SignalChannel",
    "SignalChannelSpec",
    "SignalTrace",
]