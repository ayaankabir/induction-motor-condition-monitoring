"""Deterministic, self-contained HTML rendering for an engineering report."""

from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from imcm.reporting.engineering_report import (
    EngineeringReport,
    EngineeringReportError,
    ReportCondition,
    ReportFile,
)


_STYLE = """
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #52606d;
  --line: #d9e2ec;
  --panel: #f5f7fa;
  --accent: #145da0;
  --warning-bg: #fff4d6;
  --warning-ink: #6b4e00;
  --external-bg: #eef5ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #fff;
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1120px; margin: 0 auto; padding: 32px 42px 64px; }
h1, h2, h3 { color: #102a43; line-height: 1.2; }
h1 { font-size: 30px; margin: 0 0 8px; }
h2 { border-bottom: 2px solid var(--accent); margin-top: 42px; padding-bottom: 6px; }
h3 { margin-top: 28px; }
p { margin: 8px 0 14px; }
.subtitle { color: var(--muted); font-size: 16px; margin-bottom: 18px; }
.banner { border-left: 6px solid #b7791f; background: var(--warning-bg); color: var(--warning-ink); padding: 14px 18px; margin: 18px 0; }
.external-banner { background: var(--external-bg); border-left-color: var(--accent); color: #102a43; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
.meta > div, .card { background: var(--panel); border: 1px solid var(--line); padding: 12px 14px; }
.meta dt, .card dt { color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; }
.meta dd, .card dd { margin: 3px 0 0; overflow-wrap: anywhere; }
table { border-collapse: collapse; margin: 12px 0 20px; width: 100%; }
th, td { border: 1px solid var(--line); padding: 7px 9px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eaf2f8; color: #102a43; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: #f5f7fa; border: 1px solid var(--line); padding: 12px; overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
.small { color: var(--muted); font-size: 13px; }
ul, ol { margin-top: 6px; }
figure { margin: 18px 0 26px; }
figure img { display: block; max-width: 100%; height: auto; border: 1px solid var(--line); }
figcaption { color: var(--muted); font-size: 13px; margin-top: 5px; }
.status-complete { color: #276749; font-weight: 600; }
.status-partial, .status-invalid, .status-unavailable { color: #9b2c2c; font-weight: 600; }
details { border: 1px solid var(--line); margin: 10px 0; padding: 8px 12px; }
summary { cursor: pointer; font-weight: 600; }
.missing { color: #9b2c2c; font-style: italic; }
footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; margin-top: 48px; padding-top: 12px; }
@media print {
  main { max-width: none; padding: 18px; }
  h2 { break-after: avoid; }
  table, figure, details { break-inside: avoid; }
  .banner { break-inside: avoid; }
}
""".strip()


def _value(value: Any) -> str:
    """Render a stored value without interpreting it as HTML."""

    if value is None:
        return "Not reported"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format(value, ".12g")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _html(value: Any) -> str:
    return escape(_value(value), quote=True)


def _json_block(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return f"<pre><code>{escape(text)}</code></pre>"


def _definition_list(items: Sequence[tuple[str, Any]]) -> str:
    rows = "".join(
        f"<div><dt>{escape(str(label))}</dt><dd>{_html(value)}</dd></div>"
        for label, value in items
    )
    return f'<dl class="meta">{rows}</dl>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_html(cell)}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _local_file(report: EngineeringReport, file: ReportFile) -> Path | None:
    candidate = (report.root / file.path).resolve()
    root = report.root.resolve()
    try:
        if not candidate.is_relative_to(root):
            return None
    except AttributeError:  # pragma: no cover - Python 3.11+ guarantee
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
    return candidate if candidate.is_file() else None


def _figure(report: EngineeringReport, file: ReportFile) -> str:
    path = _local_file(report, file)
    if path is None:
        return (
            f'<figure><div class="missing">Figure not available: '
            f"{escape(file.path)}</div><figcaption>{escape(file.path)}</figcaption></figure>"
        )
    try:
        data = path.read_bytes()
    except OSError:
        return (
            f'<figure><div class="missing">Figure could not be read: '
            f"{escape(file.path)}</div><figcaption>{escape(file.path)}</figcaption></figure>"
        )
    encoded = base64.b64encode(data).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" '
        f'alt="Stored engineering result figure: {escape(file.path)}">'
        f"<figcaption>{escape(file.path)}</figcaption></figure>"
    )


def _status(value: str) -> str:
    return f'<span class="status-{escape(value)}">{escape(value)}</span>'


def _condition_overview(condition: ReportCondition) -> str:
    status = _status(condition.source_status.state)
    return (
        "<article class=\"card\">"
        f"<h3>{escape(condition.name)}</h3>"
        f"<p><strong>ID:</strong> {escape(condition.condition_id)}</p>"
        f"<p><strong>Classification:</strong> {escape(condition.classification)}</p>"
        f"<p><strong>Source status:</strong> {status}</p>"
        f"<p><strong>Summary:</strong> <code>{escape(condition.summary_source)}</code></p>"
        f"<p><strong>Affected signal:</strong> {escape(condition.affected_signal)}</p>"
        f"<p>{escape(condition.mechanism)}</p>"
        "</article>"
    )


def _render_explanation(condition: ReportCondition) -> str:
    explanation = condition.explanation
    if explanation is None:
        return '<p class="missing">Existing explanation unavailable.</p>'
    parts = [
        f"<h3>Existing evidence and explanation — {escape(condition.name)}</h3>",
        _definition_list(
            [
                ("Classification category", explanation.get("classification_category")),
                ("Classification label", explanation.get("classification_label")),
                ("Physical mechanism", explanation.get("physical_mechanism")),
                ("Why classified", explanation.get("why_classified")),
            ]
        ),
    ]
    evidence = explanation.get("supporting_evidence")
    if isinstance(evidence, list) and evidence:
        parts.append(
            _table(
                ["Metric", "Observed", "Baseline", "Unit", "Role", "Interpretation"],
                [
                    [
                        item.get("metric_name"),
                        item.get("observed_value"),
                        item.get("baseline_value"),
                        item.get("unit"),
                        item.get("diagnostic_role"),
                        item.get("interpretation"),
                    ]
                    for item in evidence
                    if isinstance(item, Mapping)
                ],
            )
        )
    for title, key in (
        ("Discrimination versus confounders", "discrimination_vs_confounders"),
        ("Standing limitations", "standing_limitations"),
    ):
        values = explanation.get(key)
        if isinstance(values, list) and values:
            parts.append(f"<h4>{escape(title)}</h4><ul>")
            parts.extend(f"<li>{escape(str(item))}</li>" for item in values)
            parts.append("</ul>")
    return "".join(parts)


def _render_triage(condition: ReportCondition) -> str:
    triage = condition.triage
    if triage is None:
        return '<p class="missing">Existing deterministic triage result unavailable.</p>'
    parts = [
        f"<h3>Existing deterministic triage — {escape(condition.name)}</h3>",
        _definition_list(
            [
                ("Status", triage.get("status")),
                ("Condition", triage.get("condition_id")),
                ("Category", triage.get("classification_category")),
                ("Label", triage.get("classification_label")),
                ("Why", triage.get("why")),
                ("Triggered rules", triage.get("triggered_rule_ids")),
                ("Ambiguities", triage.get("ambiguities")),
            ]
        ),
    ]
    evidence = triage.get("supporting_evidence")
    if isinstance(evidence, list) and evidence:
        parts.append(
            _table(
                ["Metric", "Observed", "Baseline", "Unit", "Role", "Interpretation"],
                [
                    [
                        item.get("metric_name"),
                        item.get("observed_value"),
                        item.get("baseline_value"),
                        item.get("unit"),
                        item.get("diagnostic_role"),
                        item.get("interpretation"),
                    ]
                    for item in evidence
                    if isinstance(item, Mapping)
                ],
            )
        )
    rules = triage.get("rules")
    if isinstance(rules, list) and rules:
        parts.append(
            _table(
                ["Rule", "Condition", "Status", "Why"],
                [
                    [
                        item.get("rule_id"),
                        item.get("condition_id"),
                        item.get("status"),
                        item.get("why"),
                    ]
                    for item in rules
                    if isinstance(item, Mapping)
                ],
            )
        )
    return "".join(parts)


def render_html(report: EngineeringReport) -> str:
    """Render the complete report as deterministic, self-contained HTML."""

    payload = report.to_dict()
    project = report.project
    status = report.source_data_status()
    status_note = (
        "The simulated source bundle is complete."
        if status["simulated"] == "complete"
        else "The simulated source bundle is partial; missing provenance is listed explicitly."
    )
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(str(project.get('title', 'Engineering report')))}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body><main>",
        "<header>",
        f"<h1>{escape(str(project.get('title', 'Engineering report')))}</h1>",
        '<p class="subtitle">Static engineering report generated from existing stored/project records.</p>',
        "</header>",
        '<section class="banner"><strong>SIMULATED STUDY DATA</strong> — '
        "The current project results are simulated. Literature-example parameters "
        "are not measurements, and there is no experimental validation.</section>",
        "<section><h2>1. Cover and report identity</h2>",
        _definition_list(
            [
                ("Report schema", report.schema_version),
                ("Report ID", report.report_id),
                ("Report-generation timestamp", report.generated_at_utc or "not recorded"),
                ("Report-generator revision", report.report_git_commit or "not recorded"),
                ("Source bundle", project.get("source_bundle")),
            ]
        ),
        "</section>",
        "<section><h2>2. Scope and data-source classification</h2>",
        f"<p>{escape(status_note)}</p>",
        _definition_list(
            [
                ("Simulated source status", status["simulated"]),
                ("External source status", status["external"]),
                ("Project data origin", project.get("data_origin")),
                ("Parameter provenance", project.get("parameter_provenance")),
                ("Stored overview label", project.get("stored_overview_label")),
            ]
        ),
        '<p class="small">This report adds no diagnosis, severity, confidence, health index, prediction, ranking, ML, or predictive-maintenance layer.</p>',
        "</section>",
        "<section><h2>3. Project and study metadata</h2>",
        _definition_list(
            [
                ("Model / parameter name", project.get("model_name")),
                ("Scenario", project.get("scenario_name")),
                ("Park convention", project.get("park_convention")),
                ("Supply frequency", project.get("supply_frequency_hz")),
                ("Load", f"{project.get('load_type', 'not reported')} / {project.get('load_torque_nm', 'not reported')} N m"),
                ("Solver", project.get("solver")),
                ("Max step", project.get("max_step_s")),
                ("Relative tolerance", project.get("rtol")),
                ("Absolute tolerance", project.get("atol")),
                ("Integration end", project.get("t_end_s")),
                ("Output interval", project.get("output_dt_s")),
                ("Initial condition", project.get("initial_condition")),
            ]
        ),
        "</section>",
        "<section><h2>4. Condition/study register</h2>",
        _table(
            ["Order", "Condition", "ID", "Classification", "Source status", "Summary"],
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
        ),
        "</section>",
        "<section><h2>5. Key existing metrics</h2>",
        "<h3>Stored healthy baseline</h3>",
        _json_block(project.get("healthy_baseline", "The healthy baseline is retained in the stored overview condition record.")),
    ]

    for condition in report.conditions:
        parts.append(
            f"<h3>{escape(condition.name)} — stored measured values</h3>"
            + _json_block(condition.measured)
        )
    parts.append("</section>")

    parts.append("<section><h2>6. Condition details and existing evidence</h2>")
    for condition in report.conditions:
        parts.append(_condition_overview(condition))
        parts.append("<h3>Expected signature</h3>")
        parts.append(f"<p>{escape(condition.expected_signature)}</p>")
        parts.append(_render_explanation(condition))
        for figure in condition.figures:
            parts.append(_figure(report, figure))
    parts.append("</section>")

    parts.append("<section><h2>7. Existing deterministic triage results</h2>")
    parts.append(
        '<p class="small">These are the existing source-data-only rule results. '
        "They are not a new diagnosis, classifier, severity assessment, or forecast.</p>"
    )
    for condition in report.conditions:
        parts.append(_render_triage(condition))
    parts.append("</section>")

    parts.append("<section><h2>8. Provenance and reproducibility</h2>")
    parts.append(
        "<p>Source timestamps and source git commits below are copied from stored manifests. "
        "A missing value is shown as not recorded; the current checkout commit is not substituted.</p>"
    )
    for condition in report.conditions:
        provenance = condition.provenance
        parts.append(f"<h3>{escape(condition.name)}</h3>")
        parts.append(
            _definition_list(
                [
                    ("Parameter hash", provenance.parameter_hash or "not recorded"),
                    ("Parameter source", provenance.parameter_source or "not recorded"),
                    ("Solver settings source", provenance.solver_settings_source or "not recorded"),
                    ("Git commit", provenance.git_commit or "not recorded"),
                    ("Source timestamp", provenance.timestamp or "not recorded"),
                    ("Run ID", provenance.run_id or "not recorded"),
                    ("Run-config hash", provenance.run_config_hash or "not recorded"),
                    ("Manifest", condition.source_status.provenance_path or "not recorded"),
                    ("Run config", condition.source_status.run_config_path or "not recorded"),
                ]
            )
        )
        if provenance.solver_settings:
            parts.append(_json_block(provenance.solver_settings))
        if provenance.initial_condition:
            parts.append(_json_block(provenance.initial_condition))
        if provenance.notes:
            parts.append("<ul>")
            parts.extend(f"<li>{escape(note)}</li>" for note in provenance.notes)
            parts.append("</ul>")
    parts.append("<h3>Stored consistency checks (not rerun by this report)</h3>")
    if report.stored_consistency_checks:
        parts.append(
            _table(
                ["Check", "Status", "Detail"],
                [
                    [item.get("name"), item.get("status"), item.get("detail")]
                    for item in report.stored_consistency_checks
                ],
            )
        )
    else:
        parts.append('<p class="missing">No stored consistency checks were reported.</p>')
    parts.append(
        f"<p>Stored overview all-checks-pass flag: {_html(report.stored_all_checks_pass)}</p>"
    )
    parts.append("</section>")

    parts.append("<section><h2>9. Limitations and scope</h2><ul>")
    parts.extend(f"<li>{escape(str(item))}</li>" for item in report.limitations)
    parts.append("</ul></section>")

    parts.append("<section><h2>10. Source artifacts and reproducibility appendix</h2>")
    parts.append("<h3>Overview artifacts</h3>")
    for figure in report.overview_artifacts:
        parts.append(_figure(report, figure))
    if not report.overview_artifacts:
        parts.append('<p class="missing">No overview figure was available.</p>')
    parts.append("<h3>Input file digests</h3>")
    parts.append(
        _table(
            ["Path", "Kind", "Available", "SHA-256"],
            [
                [item.path, item.content_kind, item.exists, item.sha256 or "not recorded"]
                for item in report.input_files
            ],
        )
    )
    if report.external_sources:
        parts.append("<h3>External imported metadata</h3>")
        for item in report.external_sources:
            parts.append(
                _condition_card_external(report, item)
            )
    else:
        parts.append(
            '<div class="banner external-banner"><strong>External imported signals:</strong> '
            "no external signal record was supplied. No CSV was loaded by this report.</div>"
        )
    parts.append(
        "<p>Raw NPZ waveform samples are not included and were not loaded for analysis.</p>"
    )
    parts.append("</section>")
    parts.append(
        '<footer>Read-only engineering report. No source result files were modified by report generation.</footer>'
    )
    parts.append("</main></body></html>")
    return "\n".join(parts) + "\n"


def _condition_card_external(report: EngineeringReport, item) -> str:
    return (
        '<div class="banner external-banner">'
        f"<strong>External unvalidated source:</strong> {escape(item.source_name or item.source_id)}<br>"
        f"Source ID: {escape(item.source_id)}<br>"
        f"Status: {escape(item.state)}<br>"
        f"Source SHA-256: {escape(item.source_sha256 or 'not recorded')}<br>"
        f"Imported at: {escape(item.imported_at_utc or 'not recorded')}<br>"
        "This record is not simulated, experimental validation, or a diagnosis."
        "</div>"
        + _json_block(item.summary)
        + (_json_block(item.quality_report) if item.quality_report is not None else "")
    )


def write_html(
    report: EngineeringReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write deterministic HTML, refusing an existing path by default."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing report artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(report), encoding="utf-8", newline="\n")
    return target


__all__ = ["render_html", "write_html"]