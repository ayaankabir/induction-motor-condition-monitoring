"""Command-line interface to generate deterministic engineering report artifacts.

Generates:
- imcm_engineering_report.json
- imcm_engineering_report.html
- imcm_engineering_report.pdf
- imcm_engineering_report_manifest.json

Usage:
    python -m imcm.reporting.generate_report \\
        --output-dir build/report/ \\
        --generated-at-utc "2024-01-01T00:00:00+00:00"

Options:
    --output-dir        Required. Directory where report artifacts are written.
    --generated-at-utc  Required. Explicit ISO-8601 UTC timestamp for reproducibility.
    --results-dir       Optional. Path to results directory (defaults to results/).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from imcm.reporting.engineering_report import (
    EngineeringReportError,
    build_engineering_report,
    write_engineering_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="imcm-report",
        description=(
            "Generate read-only engineering report artifacts (JSON, HTML, PDF, manifest) "
            "from existing simulation results."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Target directory where generated report files will be written.",
    )
    parser.add_argument(
        "--generated-at-utc",
        type=str,
        required=True,
        help="Explicit ISO-8601 UTC timestamp (e.g. '2024-01-01T00:00:00+00:00').",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing stored result summaries and figures (default: results/).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir == results_dir:
        sys.stderr.write(
            f"Error: output-dir cannot be the same as results-dir ({results_dir})\n"
        )
        return 1

    try:
        report = build_engineering_report(
            results_dir=results_dir,
            generated_at_utc=args.generated_at_utc,
        )
        artifacts = write_engineering_report(
            report=report,
            output_dir=output_dir,
            formats=("json", "html", "pdf", "manifest"),
            overwrite=False,
        )
    except FileExistsError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except EngineeringReportError as err:
        sys.stderr.write(f"Report generation error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Unexpected error: {err}\n")
        return 1

    for fmt, path in sorted(artifacts.paths.items()):
        print(f"Generated {fmt}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
