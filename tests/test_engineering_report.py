"""Tests for the read-only engineering-report builder and renderers.

These tests exercise the approved reporting milestone only. They consume the
existing stored results directory, never run a simulation, never open an NPZ
archive, and never write into ``results``. Determinism is asserted by
rebuilding the report and comparing canonical hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from imcm.reporting.engineering_report import (
    REPORT_MANIFEST_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    EngineeringReportError,
    ReportInputError,
    build_engineering_report,
    write_engineering_report,
)
from imcm.reporting.report_html import render_html, write_html
from imcm.reporting.report_pdf import render_pdf

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIXED_TIME = "2024-01-01T00:00:00+00:00"

RAW_SCRIPT = "<" + "script>alert(1)<" + "/script>.csv"
# Build the expected escaped form without a literal entity, so editor or
# formatter HTML-entity decoding cannot silently change the expectation.
ESCAPED_SCRIPT = "".join(["&lt" + ";", "script", "&gt" + ";"])


def _build(**kwargs):
    return build_engineering_report(RESULTS_DIR, generated_at_utc=FIXED_TIME, **kwargs)


def _external_record(source_name: str = "plant_scope.csv", *, valid: bool = True) -> dict:
    provenance = {
        "source_type": "external_csv",
        "provenance_class": "external_unvalidated",
        "validation_status": "unvalidated_external_input",
        "experimental_validation": False,
        "source_name": source_name,
        "source_sha256": "a" * 64,
        "imported_at_utc": "2024-01-01T00:00:00+00:00",
    }
    if not valid:
        provenance["provenance_class"] = "something_else"
    return {
        "summary": {
            "provenance": provenance,
            "trace": {
                "sample_count": 10,
                "duration_s": 1.0,
                "sample_rate_hz": 10.0,
                "channels": [{"name": "i_a", "unit": "A", "signal_type": "current"}],
            },
        },
        "quality_report": {"analysis_scope": "data_quality_only", "status": "ok"},
    }


def _digest_tree(path: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digests[item.relative_to(path).as_posix()] = hashlib.sha256(
                item.read_bytes()
            ).hexdigest()
    return digests


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def test_build_consumes_all_stored_conditions() -> None:
    report = _build()
    ids = [condition.condition_id for condition in report.conditions]
    assert ids == ["healthy", "fault_01", "fault_02", "fault_03", "fault_04", "fault_05"]
    assert report.schema_version == REPORT_SCHEMA_VERSION
    assert report.report_id.startswith("report-")


def test_build_marks_conditions_simulated_and_partial() -> None:
    report = _build()
    assert report.source_data_status() == {"simulated": "partial", "external": "none_supplied"}
    for condition in report.conditions:
        assert condition.source_status.source_kind == "simulated_study"
        assert condition.source_status.state in {"partial", "complete"}
        assert condition.provenance.parameter_source == "literature_example"


def test_build_never_loads_waveform_archives() -> None:
    report = _build()
    # The builder must reference only presentation-safe artifacts; it never
    # opens an NPZ archive.  A stored output-file *name* may legitimately appear
    # in copied manifest provenance, but no waveform record is loaded.
    for record in report.input_files:
        assert not record.path.endswith(".npz")
        assert record.content_kind in {"json", "image", "md"}
    for condition in report.conditions:
        for figure in condition.figures:
            assert figure.content_kind == "image"
    for artifact in report.overview_artifacts:
        assert artifact.content_kind == "image"


def test_stored_consistency_checks_are_copied_not_rerun() -> None:
    report = _build()
    assert report.stored_all_checks_pass is True
    assert len(report.stored_consistency_checks) >= 8
    for check in report.stored_consistency_checks:
        assert check["status"] == "pass"


def test_limitations_include_scope_boundaries() -> None:
    report = _build()
    text = "\n".join(report.limitations)
    assert "no new simulation" in text.lower()
    assert "no experimental validation" in text or "simulated" in text.lower()
    assert "ML" in text or "diagnosis" in text.lower() or "severity" in text.lower()


def test_report_id_is_deterministic_and_content_addressed() -> None:
    first = _build()
    second = _build()
    assert first.report_id == second.report_id
    assert first.to_dict() == second.to_dict()


def test_generated_at_utc_is_reproducible_not_datetime_now() -> None:
    report = _build()
    assert report.generated_at_utc == FIXED_TIME


def test_missing_overview_is_a_hard_input_error(tmp_path: Path) -> None:
    with pytest.raises(ReportInputError):
        build_engineering_report(tmp_path, generated_at_utc=FIXED_TIME)


def test_invalid_timestamp_is_rejected() -> None:
    with pytest.raises(EngineeringReportError):
        build_engineering_report(RESULTS_DIR, generated_at_utc="2024-01-01")
    with pytest.raises(EngineeringReportError):
        build_engineering_report(RESULTS_DIR, generated_at_utc="")


# ---------------------------------------------------------------------------
# External imported records
# ---------------------------------------------------------------------------


def test_valid_external_record_is_classified_external_unvalidated() -> None:
    report = _build(external_sources=[_external_record()])
    assert report.source_data_status()["external"] == "complete"
    record = report.external_sources[0]
    assert record.source_kind == "external_unvalidated_signal"
    assert record.state == "complete"
    assert record.source_name == "plant_scope.csv"


def test_external_record_without_required_tags_is_invalid() -> None:
    report = _build(external_sources=[_external_record(valid=False)])
    record = report.external_sources[0]
    assert record.state == "invalid"
    assert any("unvalidated external provenance" in note for note in record.notes)


def test_external_quality_scope_must_be_data_quality_only() -> None:
    bad = _external_record()
    bad["quality_report"] = {"analysis_scope": "diagnosis"}
    report = _build(external_sources=[bad])
    assert report.external_sources[0].state == "invalid"


def test_external_sources_default_to_none_supplied() -> None:
    report = _build()
    assert report.external_sources == ()
    assert report.source_data_status()["external"] == "none_supplied"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_html_is_self_contained_and_flags_simulated_data() -> None:
    html = render_html(_build())
    assert html.startswith("<!doctype html>")
    assert "SIMULATED STUDY DATA" in html
    assert "data:image/png;base64," in html
    assert "EngineeringReport" not in html


def test_html_escapes_external_source_names() -> None:
    record = _external_record(source_name=RAW_SCRIPT)
    html = render_html(_build(external_sources=[record]))
    assert RAW_SCRIPT not in html
    assert ESCAPED_SCRIPT in html


def test_html_reports_absent_external_sources() -> None:
    html = render_html(_build())
    assert "no external signal record was supplied" in html.lower()


def test_pdf_is_a_valid_document(tmp_path: Path) -> None:
    target = tmp_path / "report.pdf"
    render_pdf(_build(), target)
    data = target.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.rstrip().endswith(b"%%EOF")


def test_pdf_rendering_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    render_pdf(_build(), first)
    render_pdf(_build(), second)
    assert first.read_bytes() == second.read_bytes()


def test_pdf_refuses_existing_path_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "report.pdf"
    render_pdf(_build(), target)
    with pytest.raises(FileExistsError):
        render_pdf(_build(), target)


def test_html_writer_refuses_existing_path_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "report.html"
    write_html(_build(), target)
    with pytest.raises(FileExistsError):
        write_html(_build(), target)


# ---------------------------------------------------------------------------
# Output orchestration
# ---------------------------------------------------------------------------


def test_write_produces_all_four_artifacts(tmp_path: Path) -> None:
    artifacts = write_engineering_report(_build(), tmp_path)
    assert set(artifacts.paths) == {"json", "html", "pdf", "manifest"}
    for path in artifacts.paths.values():
        assert path.is_file()
    manifest = json.loads(artifacts.paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == REPORT_MANIFEST_SCHEMA_VERSION
    assert {item["format"] for item in manifest["outputs"]} == {"json", "html", "pdf"}


def test_write_is_deterministic_across_runs(tmp_path: Path) -> None:
    first = write_engineering_report(_build(), tmp_path / "a")
    second = write_engineering_report(_build(), tmp_path / "b")
    assert first.hashes == second.hashes


def test_write_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    write_engineering_report(_build(), tmp_path)
    with pytest.raises(FileExistsError):
        write_engineering_report(_build(), tmp_path)


def test_write_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(EngineeringReportError):
        write_engineering_report(_build(), tmp_path, formats=("json", "docx"))


# ---------------------------------------------------------------------------
# Report generation must not mutate the source bundle
# ---------------------------------------------------------------------------


def test_report_generation_does_not_modify_results(tmp_path: Path) -> None:
    before = _digest_tree(RESULTS_DIR)
    write_engineering_report(_build(), tmp_path)
    after = _digest_tree(RESULTS_DIR)
    assert before == after