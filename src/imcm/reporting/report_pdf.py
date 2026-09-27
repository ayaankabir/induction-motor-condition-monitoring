"""Deterministic PDF rendering for an existing engineering-report model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from imcm.reporting.engineering_report import (
    EngineeringReport,
    EngineeringReportError,
    ReportCondition,
    ReportFile,
)


_PAGE_WIDTH = 8.27
_PAGE_HEIGHT = 11.69
_FIXED_PDF_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)
_FIXED_TEXT_WIDTH = 105


def _display(value: Any) -> str:
    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (int, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _paragraphs(text: Any, width: int = _FIXED_TEXT_WIDTH) -> list[str]:
    rendered = _display(text)
    lines: list[str] = []
    for paragraph in rendered.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False) or [""])
    return lines or [""]


def _add_text_page(
    pdf: PdfPages,
    title: str,
    lines: Sequence[str],
    *,
    subtitle: str | None = None,
) -> None:
    """Write a text page, paginating deterministically without recursion."""

    remaining = list(lines)
    page_title = title
    is_first = True
    while True:
        figure = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
        figure.patch.set_facecolor("white")
        y = 0.94
        figure.text(0.07, y, page_title, fontsize=16, fontweight="bold", color="#102a43", va="top")
        y -= 0.045
        if is_first and subtitle:
            for line in _paragraphs(subtitle):
                figure.text(0.07, y, line, fontsize=9, color="#52606d", va="top")
                y -= 0.018
            y -= 0.012
        emitted = 0
        for line in remaining:
            if y < 0.07:
                break
            figure.text(0.07, y, line, fontsize=8.5, color="#17202a", va="top", family="DejaVu Sans")
            y -= 0.022
            emitted += 1
        remaining = remaining[emitted:]
        figure.text(
            0.07,
            0.035,
            "Read-only report • no new scientific calculation • no experimental validation",
            fontsize=7,
            color="#52606d",
        )
        pdf.savefig(figure, metadata={"CreationDate": _FIXED_PDF_DATE, "ModDate": _FIXED_PDF_DATE})
        plt.close(figure)
        if not remaining:
            return
        # Guarantee forward progress even for pathological zero-capacity pages.
        if emitted == 0:
            figure = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
            figure.patch.set_facecolor("white")
            figure.text(0.07, 0.94, page_title, fontsize=16, fontweight="bold", color="#102a43", va="top")
            figure.text(0.07, 0.90, remaining[0], fontsize=8.5, color="#17202a", va="top", family="DejaVu Sans")
            pdf.savefig(figure, metadata={"CreationDate": _FIXED_PDF_DATE, "ModDate": _FIXED_PDF_DATE})
            plt.close(figure)
            remaining = remaining[1:]
            if not remaining:
                return
        page_title = f"{title} (continued)"
        is_first = False


def _add_table_page(
    pdf: PdfPages,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> None:
    figure = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    figure.patch.set_facecolor("white")
    figure.text(0.05, 0.95, title, fontsize=15, fontweight="bold", color="#102a43", va="top")
    if not rows:
        figure.text(0.05, 0.90, "No rows available.", fontsize=9, color="#9b2c2c")
        pdf.savefig(figure, metadata={"CreationDate": _FIXED_PDF_DATE, "ModDate": _FIXED_PDF_DATE})
        plt.close(figure)
        return
    # Keep cells bounded so the output remains a readable, deterministic table.
    header = [textwrap.shorten(_display(cell), width=42, placeholder="…") for cell in headers]
    body = [
        [textwrap.shorten(_display(cell), width=42, placeholder="…") for cell in row]
        for row in rows
    ]
    axis = figure.add_axes([0.05, 0.08, 0.90, 0.80])
    axis.axis("off")
    table = axis.table(
        cellText=body,
        colLabels=header,
        colWidths=[1 / len(headers)] * len(headers),
        loc="upper left",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    for (row_index, _column_index), cell in table.get_celld().items():
        is_header = row_index == 0
        cell.get_text().set_fontweight("bold" if is_header else "normal")
        cell.get_text().set_color("white" if is_header else "#17202a")
        cell.set_facecolor("#145da0" if is_header else ("#f5f7fa" if row_index % 2 else "white"))
        cell.set_edgecolor("#d9e2ec")
    figure.text(0.05, 0.035, "Stored values only; missing values are not treated as zero.", fontsize=7, color="#52606d")
    pdf.savefig(figure, metadata={"CreationDate": _FIXED_PDF_DATE, "ModDate": _FIXED_PDF_DATE})
    plt.close(figure)


def _local_file(report: EngineeringReport, file: ReportFile) -> Path | None:
    candidate = (report.root / file.path).resolve()
    root = report.root.resolve()
    try:
        if not candidate.is_relative_to(root):
            return None
    except AttributeError:  # pragma: no cover
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
    return candidate if candidate.is_file() else None


def _add_figure_page(pdf: PdfPages, report: EngineeringReport, file: ReportFile, title: str) -> None:
    path = _local_file(report, file)
    figure = plt.figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT))
    figure.patch.set_facecolor("white")
    figure.text(0.05, 0.95, title, fontsize=14, fontweight="bold", color="#102a43", va="top")
    if path is None:
        figure.text(0.05, 0.90, f"Figure not available: {file.path}", fontsize=10, color="#9b2c2c")
    else:
        try:
            image = plt.imread(path)
            axis = figure.add_axes([0.07, 0.18, 0.86, 0.68])
            axis.imshow(image)
            axis.axis("off")
            figure.text(0.07, 0.11, file.path, fontsize=7, color="#52606d")
        except (OSError, ValueError) as exc:
            figure.text(0.05, 0.90, f"Figure could not be read: {file.path} ({type(exc).__name__})", fontsize=10, color="#9b2c2c")
    figure.text(0.07, 0.035, "Existing stored figure; no new plot was generated by this report.", fontsize=7, color="#52606d")
    pdf.savefig(figure, metadata={"CreationDate": _FIXED_PDF_DATE, "ModDate": _FIXED_PDF_DATE})
    plt.close(figure)


def _condition_lines(condition: ReportCondition) -> list[str]:
    explanation = condition.explanation or {}
    triage = condition.triage or {}
    lines = [
        f"Condition ID: {condition.condition_id}",
        f"Name: {condition.name}",
        f"Classification: {condition.classification}",
        f"Source status: {condition.source_status.state}",
        f"Summary: {condition.summary_source}",
        f"Affected signal: {condition.affected_signal}",
        f"Expected signature: {condition.expected_signature}",
        f"Mechanism: {condition.mechanism}",
        "",
        "Stored measured values:",
    ]
    lines.extend(f"  {key}: {_display(value)}" for key, value in condition.measured.items())
    lines.extend(["", "Existing explanation:"])
    for key in (
        "classification_category",
        "classification_label",
        "physical_mechanism",
        "why_classified",
    ):
        lines.append(f"  {key}: {_display(explanation.get(key))}")
    evidence = explanation.get("supporting_evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, Mapping):
                lines.append(
                    "  evidence — "
                    f"{item.get('metric_name')}: observed={_display(item.get('observed_value'))}, "
                    f"baseline={_display(item.get('baseline_value'))}, "
                    f"{item.get('interpretation')}"
                )
    lines.extend(["", "Existing deterministic triage:"])
    for key in ("status", "condition_id", "classification_label", "why", "triggered_rule_ids"):
        lines.append(f"  {key}: {_display(triage.get(key))}")
    if condition.limitations:
        lines.extend(["", "Stored limitations:"])
        lines.extend(f"  - {item}" for item in condition.limitations)
    return lines


def _triage_lines(report: EngineeringReport) -> list[str]:
    lines = [
        "These are the existing source-data-only rule results. They are not a new "
        "diagnosis, classifier, severity assessment, or forecast.",
        "",
    ]
    for condition in report.conditions:
        triage = condition.triage
        lines.append(condition.name)
        if not isinstance(triage, Mapping):
            lines.append("  Existing deterministic triage result unavailable.")
            lines.append("")
            continue
        for key in (
            "status",
            "condition_id",
            "classification_category",
            "classification_label",
            "why",
            "triggered_rule_ids",
            "ambiguities",
        ):
            lines.append(f"  {key}: {_display(triage.get(key))}")
        rules = triage.get("rules")
        if isinstance(rules, list) and rules:
            lines.append("  rules:")
            for item in rules:
                if isinstance(item, Mapping):
                    lines.append(
                        f"    {item.get('rule_id')}: {item.get('status')} "
                        f"(condition={item.get('condition_id')})"
                    )
        lines.append("")
    return lines


def _provenance_lines(report: EngineeringReport) -> list[str]:
    lines = [
        "Source provenance is copied from stored manifests. Missing values remain not recorded; "
        "the current checkout commit is not substituted.",
        "",
    ]
    for condition in report.conditions:
        provenance = condition.provenance
        lines.append(condition.name)
        lines.append(f"  parameter_hash: {provenance.parameter_hash or 'not recorded'}")
        lines.append(f"  parameter_source: {provenance.parameter_source or 'not recorded'}")
        lines.append(f"  solver_settings_source: {provenance.solver_settings_source or 'not recorded'}")
        lines.append(f"  git_commit: {provenance.git_commit or 'not recorded'}")
        lines.append(f"  source_timestamp: {provenance.timestamp or 'not recorded'}")
        lines.append(f"  run_id: {provenance.run_id or 'not recorded'}")
        lines.append(f"  run_config_hash: {provenance.run_config_hash or 'not recorded'}")
        lines.append(f"  manifest: {condition.source_status.provenance_path or 'not recorded'}")
        lines.append(f"  run_config: {condition.source_status.run_config_path or 'not recorded'}")
        lines.append("")
    lines.append("Stored consistency checks (not rerun by this report):")
    for check in report.stored_consistency_checks:
        lines.append(
            f"  {check.get('name')}: {check.get('status')} — {check.get('detail')}"
        )
    lines.append(f"Stored all-checks-pass flag: {_display(report.stored_all_checks_pass)}")
    return lines


def render_pdf(
    report: EngineeringReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render the report to a deterministic multi-page PDF.

    The function is intentionally presentation-only. It reads existing raster
    artifacts, never NPZ traces, and uses fixed PDF dates/metadata.
    """

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    project = report.project
    status = report.source_data_status()

    with PdfPages(
        target,
        metadata={
            "Title": _display(project.get("title")),
            "Author": "IMCM engineering-report generator",
            "Subject": "Read-only presentation of stored project data",
            "Creator": "imcm.reporting.report_pdf",
            "Producer": "imcm.reporting.report_pdf",
            "CreationDate": _FIXED_PDF_DATE,
            "ModDate": _FIXED_PDF_DATE,
        },
    ) as pdf:
        cover = [
            f"Report schema: {report.schema_version}",
            f"Report ID: {report.report_id}",
            f"Report-generation timestamp: {report.generated_at_utc or 'not recorded'}",
            f"Report-generator revision: {report.report_git_commit or 'not recorded'}",
            f"Source bundle: {project.get('source_bundle')}",
            f"Simulated source status: {status['simulated']}",
            f"External source status: {status['external']}",
            "",
            "SIMULATED STUDY DATA — literature-example parameters are not measurements; "
            "there is no experimental validation.",
            "This report performs no new simulation, feature extraction, or scientific calculation.",
        ]
        _add_text_page(pdf, "1. Cover and report identity", cover)

        scope = [
            f"Project: {_display(project.get('title'))}",
            f"Model / parameter: {_display(project.get('model_name'))}",
            f"Scenario: {_display(project.get('scenario_name'))}",
            f"Park convention: {_display(project.get('park_convention'))}",
            f"Supply frequency: {_display(project.get('supply_frequency_hz'))} Hz",
            f"Load: {_display(project.get('load_type'))}, {_display(project.get('load_torque_nm'))} N m",
            f"Solver: {_display(project.get('solver'))}",
            f"max_step: {_display(project.get('max_step_s'))} s",
            f"rtol: {_display(project.get('rtol'))}; atol: {_display(project.get('atol'))}",
            f"t_end: {_display(project.get('t_end_s'))} s; output_dt: {_display(project.get('output_dt_s'))} s",
            f"Initial condition: {_display(project.get('initial_condition'))}",
        ]
        _add_text_page(pdf, "2. Scope, classification, and study metadata", scope)

        _add_table_page(
            pdf,
            "3. Condition/study register",
            ["Order", "Condition", "ID", "Classification", "Status", "Summary"],
            [
                [
                    index,
                    condition.name,
                    condition.condition_id,
                    condition.classification,
                    condition.source_status.state,
                    condition.summary_source,
                ]
                for index, condition in enumerate(report.conditions, start=1)
            ],
        )
        _add_text_page(
            pdf,
            "4. Key existing metrics",
            ["Stored healthy baseline:", _display(project.get("healthy_baseline", "See stored overview condition record."))]
            + [
                f"{condition.name}: {_display(condition.measured)}"
                for condition in report.conditions
            ],
        )

        for index, condition in enumerate(report.conditions, start=1):
            _add_text_page(
                pdf,
                f"5.{index} Condition details and existing evidence — {condition.name}",
                _condition_lines(condition),
            )
            for figure in condition.figures:
                _add_figure_page(pdf, report, figure, f"Stored figure — {figure.path}")

        _add_text_page(
            pdf,
            "6. Existing deterministic triage results",
            _triage_lines(report),
        )
        _add_text_page(
            pdf,
            "7. Provenance and reproducibility",
            _provenance_lines(report),
        )
        _add_text_page(
            pdf,
            "8. Limitations and scope",
            [str(item) for item in report.limitations],
        )
        appendix = [
            "Existing overview artifacts:",
            *[
                f"  {item.path} (sha256={item.sha256 or 'not recorded'})"
                for item in report.overview_artifacts
            ],
            "",
            "Input file digests:",
            *[
                f"  {item.path} | {item.content_kind} | available={item.exists} | sha256={item.sha256 or 'not recorded'}"
                for item in report.input_files
            ],
            "",
            "Raw NPZ waveform samples are not included and were not loaded for analysis.",
        ]
        if report.external_sources:
            appendix.append("")
            appendix.append("External imported metadata:")
            for item in report.external_sources:
                appendix.extend(
                    [
                        f"  {item.source_name or item.source_id}: {item.source_kind}; status={item.state}",
                        f"  source_sha256={item.source_sha256 or 'not recorded'}; imported_at={item.imported_at_utc or 'not recorded'}",
                    ]
                )
        else:
            appendix.extend(["", "No external signal record was supplied; no CSV was loaded."])
        _add_text_page(pdf, "9. Source artifacts and reproducibility appendix", appendix)
        for figure in report.overview_artifacts:
            _add_figure_page(pdf, report, figure, f"Stored overview figure — {figure.path}")

    return target


def write_pdf(
    report: EngineeringReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Compatibility alias for the deterministic PDF writer."""

    return render_pdf(report, path, overwrite=overwrite)


__all__ = ["render_pdf", "write_pdf"]