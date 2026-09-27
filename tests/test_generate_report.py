"""Tests for the deterministic engineering-report CLI generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from imcm.reporting.generate_report import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIXED_TIMESTAMP = "2024-01-01T00:00:00+00:00"


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_generates_all_four_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"

    code = main([
        "--output-dir", str(output_dir),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code == 0

    expected_files = {
        "imcm_engineering_report.json",
        "imcm_engineering_report.html",
        "imcm_engineering_report.pdf",
        "imcm_engineering_report_manifest.json",
    }
    actual_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert expected_files.issubset(actual_files)

    # Manifest checks
    manifest_data = json.loads((output_dir / "imcm_engineering_report_manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["generated_at_utc"] == FIXED_TIMESTAMP
    assert {o["format"] for o in manifest_data["outputs"]} == {"json", "html", "pdf"}


def test_cli_is_strictly_deterministic(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    code_a = main([
        "--output-dir", str(run_a),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code_a == 0

    code_b = main([
        "--output-dir", str(run_b),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code_b == 0

    filenames = [
        "imcm_engineering_report.json",
        "imcm_engineering_report.html",
        "imcm_engineering_report.pdf",
        "imcm_engineering_report_manifest.json",
    ]

    for fname in filenames:
        hash_a = _digest_file(run_a / fname)
        hash_b = _digest_file(run_b / fname)
        assert hash_a == hash_b, f"Artifact {fname} is not byte-for-byte deterministic across runs"


def test_cli_fails_safely_when_output_dir_equals_results_dir(tmp_path: Path) -> None:
    code = main([
        "--output-dir", str(RESULTS_DIR),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code == 1


def test_cli_refuses_to_overwrite_existing_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"

    # First invocation succeeds
    code1 = main([
        "--output-dir", str(output_dir),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code1 == 0

    # Second invocation refuses to overwrite and exits nonzero
    code2 = main([
        "--output-dir", str(output_dir),
        "--results-dir", str(RESULTS_DIR),
        "--generated-at-utc", FIXED_TIMESTAMP,
    ])
    assert code2 == 1


def test_cli_requires_generated_at_utc(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"

    with pytest.raises(SystemExit) as excinfo:
        main([
            "--output-dir", str(output_dir),
            "--results-dir", str(RESULTS_DIR),
        ])
    assert excinfo.value.code != 0
