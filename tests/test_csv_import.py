from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from imcm.data import (
    CANONICAL_UNITS,
    CsvImportError,
    CsvImportSpec,
    ExternalSignalProvenance,
    ImportedSignal,
    SignalChannelSpec,
    SignalTrace,
    load_signal_csv,
)


def _spec(
    *,
    sample_rate_hz: float | None = None,
    sample_rate_rtol: float = 1.0e-3,
) -> CsvImportSpec:
    return CsvImportSpec(
        time_column="time_s",
        channels=(
            SignalChannelSpec(
                name="phase_a_current",
                column="current_a",
                unit="A",
                signal_type="current",
                clipping_bounds=(-10.0, 10.0),
            ),
            SignalChannelSpec(
                name="radial_acceleration",
                column="acceleration",
                unit="m/s^2",
                signal_type="vibration",
            ),
        ),
        sample_rate_hz=sample_rate_hz,
        sample_rate_rtol=sample_rate_rtol,
    )


def _write_csv(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def test_canonical_unit_contract_is_explicit_and_does_not_convert():
    assert CANONICAL_UNITS == frozenset(
        {"A", "V", "m/s^2", "rad/s", "rpm", "N*m", "s", "Hz", "dimensionless"}
    )

    with pytest.raises(ValueError, match="unsupported unit"):
        SignalChannelSpec(name="current", column="current", unit="mA")

    with pytest.raises(ValueError, match="automatic time conversion"):
        CsvImportSpec(
            time_column="time_ms",
            channels=(SignalChannelSpec("current", "current_a", "A"),),
            time_unit="ms",  # type: ignore[arg-type]
        )


def _external_provenance(*, unit: str = "A") -> ExternalSignalProvenance:
    return ExternalSignalProvenance(
        source_name="direct construction test",
        source_sha256=hashlib.sha256(b"test source").hexdigest(),
        imported_at_utc="2026-09-27T00:00:00+00:00",
        time_column="time_s",
        channel_mapping={"current": "current"},
        units={"current": unit},
        sample_rate_hz=100.0,
        sample_rate_source="declared",
        declared_sample_rate_hz=100.0,
        row_count=2,
    )


def test_external_provenance_enforces_canonical_units():
    with pytest.raises(ValueError, match="unsupported unit"):
        _external_provenance(unit="mA")


def test_imported_signal_rejects_provenance_trace_mismatch():
    trace = SignalTrace.from_arrays(
        time_s=[0.0, 0.01],
        channels={"current": [1.0, 2.0]},
        units={"current": "A"},
        sample_rate_hz=100.0,
        sample_rate_source="declared",
        declared_sample_rate_hz=100.0,
    )

    with pytest.raises(ValueError, match="provenance units must match"):
        ImportedSignal(trace=trace, provenance=_external_provenance(unit="V"))


def test_load_signal_csv_integrates_trace_and_external_provenance(tmp_path):
    csv_path = _write_csv(
        tmp_path / "recording.csv",
        "time_s,current_a,acceleration\n"
        "0.00,1.0,0.1\n"
        "0.01,2.0,0.2\n"
        "0.02,3.0,0.3\n"
        "0.03,4.0,0.4\n",
    )

    imported = load_signal_csv(
        csv_path,
        _spec(),
        source_name="operator supplied recording",
        imported_at_utc="2026-09-27T00:00:00+00:00",
    )

    assert isinstance(imported, ImportedSignal)
    assert imported.trace.channel_names == (
        "phase_a_current",
        "radial_acceleration",
    )
    np.testing.assert_allclose(imported.trace.time_s, [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(
        imported.trace.channel("phase_a_current").values,
        [1.0, 2.0, 3.0, 4.0],
    )
    assert imported.trace.sample_rate_hz == pytest.approx(100.0)
    assert imported.trace.sample_rate_source == "derived_from_time"
    assert imported.trace.metadata["sample_rate_relative_spread"] == pytest.approx(0.0)
    assert imported.provenance.source_name == "operator supplied recording"
    assert imported.provenance.source_sha256 == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert imported.provenance.channel_mapping == {
        "phase_a_current": "current_a",
        "radial_acceleration": "acceleration",
    }
    assert imported.provenance.units == {
        "phase_a_current": "A",
        "radial_acceleration": "m/s^2",
    }
    assert imported.provenance.provenance_class == "external_unvalidated"
    assert imported.provenance.validation_status == "unvalidated_external_input"
    assert imported.provenance.experimental_validation is False
    assert json.loads(json.dumps(imported.to_summary_dict()))["trace"][
        "sample_count"
    ] == 4


def test_declared_rate_is_preserved_and_not_repaired(tmp_path):
    csv_path = _write_csv(
        tmp_path / "mislabelled-rate.csv",
        "time_s,current_a,acceleration\n"
        "0,0,0\n"
        "0.01,1,1\n"
        "0.02,2,2\n",
    )

    imported = load_signal_csv(csv_path, _spec(sample_rate_hz=250.0))

    assert imported.trace.sample_rate_hz == 250.0
    assert imported.trace.declared_sample_rate_hz == 250.0
    assert imported.trace.sample_rate_source == "declared"
    assert imported.trace.metadata["derived_sample_rate_hz"] == pytest.approx(100.0)


def test_nonfinite_signal_values_are_retained_for_quality_analysis(tmp_path):
    csv_path = _write_csv(
        tmp_path / "nonfinite-signal.csv",
        "time_s,current_a,acceleration\n"
        "0,0,NaN\n"
        "0.01,Inf,1\n"
        "0.02,-Inf,2\n",
    )

    imported = load_signal_csv(csv_path, _spec())

    np.testing.assert_equal(
        imported.trace.channel("phase_a_current").values,
        [0.0, np.inf, -np.inf],
    )
    assert np.isnan(imported.trace.channel("radial_acceleration").values[0])


@pytest.mark.parametrize(
    ("text", "expected_code"),
    [
        ("", "empty_file"),
        ("time_s,current_a,acceleration\n", "insufficient_rows"),
        (
            "time_s,current_a\n0,1\n0.01,2\n",
            "missing_column",
        ),
        (
            "time_s,current_a,acceleration,extra\n0,1,2,3\n0.01,2,3,4\n",
            "unexpected_column",
        ),
        (
            "time_s,current_a,acceleration\n0,1,2\n0,2,3\n",
            "nonmonotonic_time",
        ),
        (
            "time_s,current_a,acceleration\n0,1,2\nnan,2,3\n",
            "nonfinite_time",
        ),
        (
            "time_s,current_a,acceleration\n0,,2\n0.01,2,3\n",
            "empty_value",
        ),
        (
            "time_s,current_a,current_a\n0,1,2\n0.01,2,3\n",
            "duplicate_header",
        ),
    ],
)
def test_csv_import_errors_are_structured(tmp_path, text, expected_code):
    csv_path = _write_csv(tmp_path / "invalid.csv", text)

    with pytest.raises(CsvImportError) as caught:
        load_signal_csv(csv_path, _spec())

    assert caught.value.code == expected_code
    assert caught.value.path == str(csv_path)
    assert str(caught.value).startswith(f"{expected_code}:")