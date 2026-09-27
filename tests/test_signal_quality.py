from __future__ import annotations

import json

import numpy as np
import pytest

from imcm.data import SignalChannel, SignalTrace
from imcm.validation import analyze_signal_quality


def _trace(**overrides) -> SignalTrace:
    time_s = np.asarray(overrides.pop("time_s", [0.0, 0.01, 0.02, 0.03]), dtype=float)
    if "channels" in overrides:
        channels = overrides.pop("channels")
    else:
        sample_count = time_s.size
        channels = {
            "current": np.asarray(
                [1.0 if index % 2 == 0 else -2.0 for index in range(sample_count)]
            ),
            "speed": np.asarray(
                [1400.0 + 10.0 * index for index in range(sample_count)]
            ),
        }
    defaults = {
        "time_s": time_s,
        "channels": channels,
        "units": {name: "A" if name == "current" else "rpm" for name in channels},
        "signal_types": {
            name: "current" if name == "current" else "speed" for name in channels
        },
        "sample_rate_hz": 100.0,
        "sample_rate_source": "declared",
        "declared_sample_rate_hz": 100.0,
        "metadata": {"sample_rate_rtol": 1.0e-3},
    }
    defaults.update(overrides)
    return SignalTrace.from_arrays(**defaults)


def test_generic_quality_adapter_reports_clean_structural_trace():
    trace = _trace()
    time_before = trace.time_s.copy()
    current_before = trace.channel("current").values.copy()

    report = analyze_signal_quality(trace)

    assert report.status == "no_detected_issues"
    assert report.has_issues is False
    assert report.issues == ()
    assert report.analysis_scope == "data_quality_only"
    assert report.data_modified is False
    assert report.time.strictly_increasing is True
    assert report.sampling.uniform_within_tolerance is True
    assert report.sampling.derived_sample_rate_hz == pytest.approx(100.0)
    assert report.sampling.effective_rate_consistent is True
    assert report.channel("current").nonfinite_count == 0
    assert report.channel("current").constant is False
    np.testing.assert_array_equal(trace.time_s, time_before)
    np.testing.assert_array_equal(trace.channel("current").values, current_before)


def test_quality_report_is_json_safe_and_contains_no_diagnostic_claim():
    report = analyze_signal_quality(_trace())
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["analysis_scope"] == "data_quality_only"
    assert payload["data_modified"] is False
    assert [channel["name"] for channel in payload["channels"]] == [
        "current",
        "speed",
    ]
    assert not {"diagnosis", "severity", "confidence"}.intersection(payload)
    assert any("not source validation" in note for note in payload["limitations"])


def test_quality_adapter_reports_irregular_time_rate_mismatch_and_nonmonotonicity():
    trace = _trace(
        time_s=np.asarray([0.0, 0.01, 0.02, 0.02, 0.05]),
        sample_rate_hz=250.0,
    )

    report = analyze_signal_quality(trace)
    issue_codes = {issue.code for issue in report.issues}

    assert report.status == "issues_detected"
    assert {
        "duplicate_time",
        "nonmonotonic_time",
        "irregular_sampling",
        "sample_rate_mismatch",
    }.issubset(issue_codes)
    assert report.time.duplicate_time_count == 1
    assert report.sampling.uniform_within_tolerance is False
    assert report.sampling.effective_rate_consistent is False


def test_quality_adapter_counts_nonfinite_constant_and_declared_bound_hits():
    time_s = np.asarray([0.0, 0.01, 0.02, 0.03])
    trace = SignalTrace(
        time_s=time_s,
        channels=(
            SignalChannel(
                name="current",
                values=np.asarray([1.0, np.nan, np.inf, -np.inf]),
                unit="A",
                source_column="current",
                signal_type="current",
            ),
            SignalChannel(
                name="constant",
                values=np.asarray([2.0, 2.0, 2.0, 2.0]),
                unit="V",
                source_column="constant",
                signal_type="voltage",
            ),
            SignalChannel(
                name="bounded",
                values=np.asarray([-1.0, 0.0, 1.0, 0.5]),
                unit="V",
                source_column="bounded",
                signal_type="voltage",
                clipping_bounds=(-1.0, 1.0),
            ),
        ),
        sample_rate_hz=100.0,
        sample_rate_source="declared",
        declared_sample_rate_hz=100.0,
    )

    report = analyze_signal_quality(trace)
    codes = {(issue.code, issue.channel_name) for issue in report.issues}

    assert ("nonfinite_signal", "current") in codes
    assert ("constant_channel", "constant") in codes
    assert ("declared_bound_hit", "bounded") in codes
    assert report.channel("current").nan_count == 1
    assert report.channel("current").positive_infinity_count == 1
    assert report.channel("current").negative_infinity_count == 1
    assert report.channel("bounded").declared_bound_hit_count == 2


def test_nonfinite_time_is_reported_without_repair():
    trace = _trace(time_s=np.asarray([0.0, np.nan, 0.02]))
    original = trace.time_s.copy()

    report = analyze_signal_quality(trace)

    assert report.time.nonfinite_time_count == 1
    assert report.time.strictly_increasing is False
    assert any(issue.code == "nonfinite_time" for issue in report.issues)
    np.testing.assert_equal(trace.time_s, original)


def test_one_sample_trace_is_reported_as_insufficient():
    trace = _trace(
        time_s=np.asarray([0.0]),
        channels={"current": np.asarray([1.0])},
    )

    report = analyze_signal_quality(trace)

    assert report.time.sample_count == 1
    assert report.sampling.interval_count == 0
    assert [issue.code for issue in report.issues] == ["insufficient_samples"]