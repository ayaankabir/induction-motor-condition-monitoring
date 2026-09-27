"""Read-only data-quality checks for source-agnostic signal traces.

The checks in this module describe structural properties of a :class:`SignalTrace`.
They do not validate the source, infer a machine condition, or repair, resample,
interpolate, filter, convert, or otherwise alter any sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import numpy as np

from imcm.data.trace import SignalTrace

DEFAULT_SAMPLE_RATE_RTOL = 1.0e-3
QualityStatus = Literal["no_detected_issues", "issues_detected"]


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


@dataclass(frozen=True)
class SignalQualityIssue:
    """One objective finding produced by the quality checks."""

    code: str
    message: str
    channel_name: str | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "channel_name": self.channel_name,
            "count": self.count,
        }


@dataclass(frozen=True)
class TimeAxisQuality:
    """Quality properties of a trace's time axis."""

    sample_count: int
    finite_time_count: int
    nonfinite_time_count: int
    start_s: float | None
    end_s: float | None
    duration_s: float | None
    strictly_increasing: bool
    duplicate_time_count: int
    decreasing_time_count: int
    nonpositive_interval_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "finite_time_count": self.finite_time_count,
            "nonfinite_time_count": self.nonfinite_time_count,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "strictly_increasing": self.strictly_increasing,
            "duplicate_time_count": self.duplicate_time_count,
            "decreasing_time_count": self.decreasing_time_count,
            "nonpositive_interval_count": self.nonpositive_interval_count,
        }


@dataclass(frozen=True)
class SamplingQuality:
    """Observed time spacing and consistency with declared sample rates."""

    sample_rate_hz: float
    sample_rate_source: Literal["declared", "derived_from_time"]
    declared_sample_rate_hz: float | None
    sample_rate_rtol: float
    interval_count: int
    minimum_interval_s: float | None
    median_interval_s: float | None
    maximum_interval_s: float | None
    derived_sample_rate_hz: float | None
    relative_interval_spread: float | None
    uniform_within_tolerance: bool
    effective_rate_relative_error: float | None
    effective_rate_consistent: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "sample_rate_source": self.sample_rate_source,
            "declared_sample_rate_hz": self.declared_sample_rate_hz,
            "sample_rate_rtol": self.sample_rate_rtol,
            "interval_count": self.interval_count,
            "minimum_interval_s": self.minimum_interval_s,
            "median_interval_s": self.median_interval_s,
            "maximum_interval_s": self.maximum_interval_s,
            "derived_sample_rate_hz": self.derived_sample_rate_hz,
            "relative_interval_spread": self.relative_interval_spread,
            "uniform_within_tolerance": self.uniform_within_tolerance,
            "effective_rate_relative_error": self.effective_rate_relative_error,
            "effective_rate_consistent": self.effective_rate_consistent,
        }


@dataclass(frozen=True)
class ChannelQuality:
    """Read-only numerical and bound-related findings for one channel."""

    name: str
    unit: str
    signal_type: str
    source_column: str
    sample_count: int
    finite_count: int
    nonfinite_count: int
    nan_count: int
    positive_infinity_count: int
    negative_infinity_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    standard_deviation: float | None
    constant: bool
    clipping_bounds: tuple[float, float] | None
    at_or_below_lower_bound_count: int
    at_or_above_upper_bound_count: int
    declared_bound_hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "signal_type": self.signal_type,
            "source_column": self.source_column,
            "sample_count": self.sample_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "nan_count": self.nan_count,
            "positive_infinity_count": self.positive_infinity_count,
            "negative_infinity_count": self.negative_infinity_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "constant": self.constant,
            "clipping_bounds": self.clipping_bounds,
            "at_or_below_lower_bound_count": self.at_or_below_lower_bound_count,
            "at_or_above_upper_bound_count": self.at_or_above_upper_bound_count,
            "declared_bound_hit_count": self.declared_bound_hit_count,
        }


@dataclass(frozen=True)
class SignalQualityReport:
    """A JSON-serializable, non-diagnostic signal-quality report."""

    time: TimeAxisQuality
    sampling: SamplingQuality
    channels: tuple[ChannelQuality, ...]
    issues: tuple[SignalQualityIssue, ...]
    analysis_scope: str = "data_quality_only"
    data_modified: bool = False
    status: QualityStatus = "no_detected_issues"

    def __post_init__(self) -> None:
        if not isinstance(self.time, TimeAxisQuality):
            raise TypeError("time must be TimeAxisQuality")
        if not isinstance(self.sampling, SamplingQuality):
            raise TypeError("sampling must be SamplingQuality")
        if not isinstance(self.channels, tuple) or not all(
            isinstance(channel, ChannelQuality) for channel in self.channels
        ):
            raise TypeError("channels must be a tuple of ChannelQuality instances")
        if not isinstance(self.issues, tuple) or not all(
            isinstance(issue, SignalQualityIssue) for issue in self.issues
        ):
            raise TypeError("issues must be a tuple of SignalQualityIssue instances")
        expected_status: QualityStatus = "issues_detected" if self.issues else "no_detected_issues"
        if self.status != expected_status:
            raise ValueError(f"status must be {expected_status!r} for this issue set")
        if self.analysis_scope != "data_quality_only":
            raise ValueError("analysis_scope must be 'data_quality_only'")
        if self.data_modified is not False:
            raise ValueError("data_modified must be False")

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def channel(self, name: str) -> ChannelQuality:
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(f"unknown signal channel: {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe descriptive report without waveform samples."""
        return {
            "analysis_scope": self.analysis_scope,
            "status": self.status,
            "data_modified": self.data_modified,
            "time": self.time.to_dict(),
            "sampling": self.sampling.to_dict(),
            "channels": [channel.to_dict() for channel in self.channels],
            "issues": [issue.to_dict() for issue in self.issues],
            "limitations": [
                "These are structural data-quality checks only, not source validation.",
                "No diagnosis, severity, confidence, or predictive-maintenance claim is made.",
                "No resampling, interpolation, filtering, unit conversion, or data repair was performed.",
            ],
        }


def _time_quality(time_s: np.ndarray) -> tuple[TimeAxisQuality, np.ndarray]:
    finite_mask = np.isfinite(time_s)
    finite_count = int(np.count_nonzero(finite_mask))
    nonfinite_count = int(time_s.size - finite_count)

    if time_s.size == 0:
        return (
            TimeAxisQuality(
                sample_count=0,
                finite_time_count=0,
                nonfinite_time_count=0,
                start_s=None,
                end_s=None,
                duration_s=None,
                strictly_increasing=False,
                duplicate_time_count=0,
                decreasing_time_count=0,
                nonpositive_interval_count=0,
            ),
            np.asarray([], dtype=float),
        )

    start_s = float(time_s[0]) if finite_mask[0] else None
    end_s = float(time_s[-1]) if finite_mask[-1] else None
    duration_s = end_s - start_s if start_s is not None and end_s is not None else None

    comparable_pairs = finite_mask[:-1] & finite_mask[1:]
    intervals = time_s[1:][comparable_pairs] - time_s[:-1][comparable_pairs]
    duplicate_count = int(np.count_nonzero(intervals == 0.0))
    decreasing_count = int(np.count_nonzero(intervals < 0.0))
    nonpositive_count = int(np.count_nonzero(intervals <= 0.0))
    strictly_increasing = bool(
        time_s.size >= 2 and nonfinite_count == 0 and np.all(intervals > 0.0)
    )
    return (
        TimeAxisQuality(
            sample_count=int(time_s.size),
            finite_time_count=finite_count,
            nonfinite_time_count=nonfinite_count,
            start_s=start_s,
            end_s=end_s,
            duration_s=duration_s,
            strictly_increasing=strictly_increasing,
            duplicate_time_count=duplicate_count,
            decreasing_time_count=decreasing_count,
            nonpositive_interval_count=nonpositive_count,
        ),
        intervals,
    )


def _sampling_quality(
    trace: SignalTrace,
    comparable_intervals: np.ndarray,
    sample_rate_rtol: float,
) -> SamplingQuality:
    if comparable_intervals.size:
        minimum = float(np.min(comparable_intervals))
        median = float(np.median(comparable_intervals))
        maximum = float(np.max(comparable_intervals))
        derived_rate = 1.0 / median if median > 0.0 else None
        relative_spread = (
            float(np.max(np.abs(comparable_intervals - median)) / median)
            if median > 0.0
            else None
        )
        uniform = bool(
            np.all(comparable_intervals > 0.0)
            and relative_spread is not None
            and relative_spread <= sample_rate_rtol
        )
    else:
        minimum = None
        median = None
        maximum = None
        derived_rate = None
        relative_spread = None
        uniform = False

    effective_error = (
        abs(trace.sample_rate_hz - derived_rate) / derived_rate
        if derived_rate is not None and derived_rate > 0.0
        else None
    )
    effective_consistent = (
        bool(effective_error <= sample_rate_rtol) if effective_error is not None else None
    )
    return SamplingQuality(
        sample_rate_hz=float(trace.sample_rate_hz),
        sample_rate_source=trace.sample_rate_source,
        declared_sample_rate_hz=trace.declared_sample_rate_hz,
        sample_rate_rtol=sample_rate_rtol,
        interval_count=int(comparable_intervals.size),
        minimum_interval_s=minimum,
        median_interval_s=median,
        maximum_interval_s=maximum,
        derived_sample_rate_hz=derived_rate,
        relative_interval_spread=relative_spread,
        uniform_within_tolerance=uniform,
        effective_rate_relative_error=effective_error,
        effective_rate_consistent=effective_consistent,
    )


def _channel_quality(channel: Any) -> ChannelQuality:
    values = np.asarray(channel.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    finite_count = int(finite_values.size)
    if finite_count:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
        standard_deviation = float(np.std(finite_values))
        constant = bool(finite_count >= 2 and minimum == maximum)
    else:
        minimum = None
        maximum = None
        mean = None
        standard_deviation = None
        constant = False

    lower_hits = 0
    upper_hits = 0
    if channel.clipping_bounds is not None:
        lower, upper = channel.clipping_bounds
        lower_hits = int(np.count_nonzero(finite_values <= lower))
        upper_hits = int(np.count_nonzero(finite_values >= upper))

    nan_count = int(np.count_nonzero(np.isnan(values)))
    positive_infinity_count = int(np.count_nonzero(np.isposinf(values)))
    negative_infinity_count = int(np.count_nonzero(np.isneginf(values)))
    return ChannelQuality(
        name=channel.name,
        unit=channel.unit,
        signal_type=channel.signal_type,
        source_column=channel.source_column,
        sample_count=int(values.size),
        finite_count=finite_count,
        nonfinite_count=int(values.size - finite_count),
        nan_count=nan_count,
        positive_infinity_count=positive_infinity_count,
        negative_infinity_count=negative_infinity_count,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=standard_deviation,
        constant=constant,
        clipping_bounds=channel.clipping_bounds,
        at_or_below_lower_bound_count=lower_hits,
        at_or_above_upper_bound_count=upper_hits,
        declared_bound_hit_count=lower_hits + upper_hits,
    )


def _issues(
    time: TimeAxisQuality,
    sampling: SamplingQuality,
    channels: tuple[ChannelQuality, ...],
) -> tuple[SignalQualityIssue, ...]:
    issues: list[SignalQualityIssue] = []
    if time.sample_count < 2:
        issues.append(
            SignalQualityIssue(
                code="insufficient_samples",
                message="At least two samples are required to assess the time axis.",
                count=time.sample_count,
            )
        )
    if time.nonfinite_time_count:
        issues.append(
            SignalQualityIssue(
                code="nonfinite_time",
                message="The time axis contains NaN or infinite values.",
                count=time.nonfinite_time_count,
            )
        )
    if time.duplicate_time_count:
        issues.append(
            SignalQualityIssue(
                code="duplicate_time",
                message="The time axis contains repeated timestamps.",
                count=time.duplicate_time_count,
            )
        )
    if time.decreasing_time_count:
        issues.append(
            SignalQualityIssue(
                code="decreasing_time",
                message="The time axis contains decreasing intervals.",
                count=time.decreasing_time_count,
            )
        )
    if time.sample_count >= 2 and not time.strictly_increasing:
        issues.append(
            SignalQualityIssue(
                code="nonmonotonic_time",
                message="Time values are not strictly increasing over all adjacent samples.",
            )
        )
    if sampling.interval_count and not sampling.uniform_within_tolerance:
        issues.append(
            SignalQualityIssue(
                code="irregular_sampling",
                message=(
                    "Adjacent time intervals are not uniform within the configured "
                    "relative tolerance."
                ),
            )
        )
    if sampling.effective_rate_consistent is False:
        issues.append(
            SignalQualityIssue(
                code="sample_rate_mismatch",
                message=(
                    "The trace sample rate is inconsistent with the median interval "
                    "derived from the time axis within the configured tolerance."
                ),
            )
        )

    for channel in channels:
        if channel.nonfinite_count:
            issues.append(
                SignalQualityIssue(
                    code="nonfinite_signal",
                    message="The channel contains NaN or infinite values.",
                    channel_name=channel.name,
                    count=channel.nonfinite_count,
                )
            )
        if channel.constant:
            issues.append(
                SignalQualityIssue(
                    code="constant_channel",
                    message="All finite samples in the channel have the same value.",
                    channel_name=channel.name,
                    count=channel.finite_count,
                )
            )
        if channel.declared_bound_hit_count:
            issues.append(
                SignalQualityIssue(
                    code="declared_bound_hit",
                    message=(
                        "Samples reach a declared clipping bound; this is an observation, "
                        "not proof of sensor saturation."
                    ),
                    channel_name=channel.name,
                    count=channel.declared_bound_hit_count,
                )
            )
    return tuple(issues)


def analyze_signal_quality(
    trace: SignalTrace,
    *,
    sample_rate_rtol: float | None = None,
) -> SignalQualityReport:
    """Analyze a trace without changing, repairing, or deriving new samples.

    If ``sample_rate_rtol`` is omitted, the value recorded by
    :class:`~imcm.data.trace.CsvImportSpec` is read from ``trace.metadata``;
    otherwise the module default is used.  Findings describe the supplied arrays
    and do not establish provenance, calibration, source validity, diagnosis, or
    fitness beyond these explicit structural checks.
    """
    if not isinstance(trace, SignalTrace):
        raise TypeError("trace must be a SignalTrace")
    if sample_rate_rtol is None:
        sample_rate_rtol = trace.metadata.get("sample_rate_rtol", DEFAULT_SAMPLE_RATE_RTOL)
    tolerance = _require_positive_finite("sample_rate_rtol", sample_rate_rtol)

    time, comparable_intervals = _time_quality(trace.time_s)
    sampling = _sampling_quality(trace, comparable_intervals, tolerance)
    channels = tuple(_channel_quality(channel) for channel in trace.channels)
    issues = _issues(time, sampling, channels)
    status: QualityStatus = "issues_detected" if issues else "no_detected_issues"
    return SignalQualityReport(
        time=time,
        sampling=sampling,
        channels=channels,
        issues=issues,
        status=status,
    )


__all__ = [
    "DEFAULT_SAMPLE_RATE_RTOL",
    "ChannelQuality",
    "QualityStatus",
    "SamplingQuality",
    "SignalQualityIssue",
    "SignalQualityReport",
    "TimeAxisQuality",
    "analyze_signal_quality",
]