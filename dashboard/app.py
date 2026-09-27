"""Read-only presentation of existing simulation summaries; no experiments run."""

import json
import math
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "fault_overview_summary.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def load_json(path):
    """Return data and an actionable message without creating missing files."""
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle), None
    except (OSError, ValueError) as exc:
        return None, f"Cannot read {path.relative_to(ROOT)}: {exc}"


def local_path(reference):
    """Resolve repository paths, including absolute paths from an older checkout."""
    if not isinstance(reference, str) or not reference.strip():
        return None
    try:
        path = Path(reference)
        if path.is_absolute():
            try:
                path = path.relative_to(ROOT)
            except ValueError:
                # Generated summaries may retain another checkout's absolute path.
                if "results" not in path.parts:
                    return None
                path = Path(*path.parts[path.parts.index("results"):])
        elif len(path.parts) == 1:
            path = Path("results") / path
        resolved = (ROOT / path).resolve()
        return resolved if resolved.is_relative_to(ROOT) else None
    except (OSError, ValueError):
        return None


def display_value(value):
    if value is None:
        return "Not reported"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def show_data(value):
    """Preserve arbitrary JSON fields and avoid mixed-type table columns."""
    if isinstance(value, dict) and value:
        st.table([
            {"Field": str(key), "Value": display_value(item)}
            for key, item in value.items()
        ])
    elif isinstance(value, list) and value:
        st.json(value)
    elif value is None or value == {} or value == []:
        st.info("No data reported.")
    else:
        st.write(value)


def leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def show_plots(data):
    """Display only existing local raster plots referenced in the supplied JSON."""
    seen = set()
    found = False
    for label, reference in leaves(data):
        if not isinstance(reference, str):
            continue
        if Path(reference).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        found = True
        path = local_path(reference)
        if path is None:
            st.warning(f"Plot reference is not repository-local: {label}")
            continue
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            st.info(f"Plot not available: {relative}")
            continue
        try:
            st.image(path.read_bytes(), caption=f"{label} — {relative}")
        except Exception as exc:
            # A corrupt or unsupported plot must not hide the rest of the report.
            st.warning(f"Cannot display {relative}: {exc}")
    if not found:
        st.caption("No raster plot references are reported in this data.")


def check_fields(data):
    """Discover stored checks without inferring validation from unrelated booleans."""
    if isinstance(data, dict):
        for key, value in data.items():
            if any(word in key.lower() for word in ("check", "validation")):
                yield key, value
            elif isinstance(value, (dict, list)):
                for child, item in check_fields(value):
                    yield f"{key}.{child}", item
    elif isinstance(data, list):
        for index, value in enumerate(data):
            for child, item in check_fields(value):
                yield f"[{index}].{child}", item


def condition_records(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [
            {"key": key, **item} if isinstance(item, dict)
            else {"key": key, "summary": item}
            for key, item in value.items()
        ]
    return []


def main():
    st.set_page_config(page_title="Motor Condition Dashboard", layout="wide")
    st.title("Induction Motor Condition Monitoring")
    st.caption(
        "Read-only • simulated traces • literature-example parameters • "
        "not experimentally validated"
    )

    data, error = load_json(SUMMARY)
    if error:
        st.warning(error)
    if not isinstance(data, dict):
        if error is None:
            st.warning("The overview JSON must contain an object.")
        data = {}

    conditions = condition_records(data.get("conditions"))
    names = [
        str(item.get("name") or item.get("key") or f"Condition {index + 1}")
        for index, item in enumerate(conditions)
    ]

    # Read existing referenced summaries only; never launch generators or load NPZs.
    sources = {}
    references = data.get("generated_from", [])
    if not isinstance(references, list):
        references = []
    references = references + [
        item["summary_source"] for item in conditions if "summary_source" in item
    ]
    for reference in references:
        if not isinstance(reference, str) or reference in sources:
            continue
        path = local_path(reference)
        if path is None or path.suffix.lower() != ".json":
            sources[reference] = (None, "Unsupported or non-local summary reference.")
        else:
            sources[reference] = load_json(path)

    overview, explorer, comparison, validation, about = st.tabs([
        "Overview",
        "Condition Explorer",
        "Cross-Condition Comparison",
        "Consistency & Validation",
        "About / Scope",
    ])

    with overview:
        st.header("Overview")
        st.caption("Source: results/fault_overview_summary.json")
        st.metric("Available conditions", len(conditions))
        show_data({
            key: value for key, value in data.items()
            if key not in {"conditions", "healthy_baseline", "artifacts"}
        })
        st.subheader("Healthy baseline")
        show_data(data.get("healthy_baseline"))
        with st.expander("Complete overview JSON"):
            st.json(data)

    with explorer:
        st.header("Condition Explorer")
        if not conditions:
            st.info("No conditions available. Supply the existing overview JSON.")
        else:
            index = st.selectbox(
                "Condition", range(len(conditions)),
                format_func=lambda item: names[item],
            )
            condition = conditions[index]
            st.subheader(names[index])
            show_data(condition)
            st.caption(
                "Fields named 'measured' are extracted from simulations, "
                "not physical measurements. Missing values are not zero."
            )
            show_plots(condition)
            reference = condition.get("summary_source")
            if isinstance(reference, str) and reference in sources:
                summary, source_error = sources[reference]
                st.subheader("Source summary")
                if source_error:
                    st.info(source_error)
                else:
                    show_data(summary)
                    show_plots(summary)

    with comparison:
        st.header("Cross-Condition Comparison")
        st.caption(
            "Only identical stored metric names are aligned; no unit conversion, "
            "missing-value substitution, or diagnostic ranking is performed. "
            "Different conditions may use different signals or analysis windows."
        )
        measured = [
            dict(leaves(item.get("measured", {})))
            if isinstance(item.get("measured"), dict) else {}
            for item in conditions
        ]
        metrics = sorted({key for item in measured for key in item})
        if not metrics:
            st.info("No measured metrics available for comparison.")
        else:
            selected = st.multiselect("Metrics", metrics, default=metrics[:5])
            rows = [
                {
                    "Condition": f"{index + 1}. {names[index]}",
                    **{key: display_value(values.get(key)) for key in selected},
                }
                for index, values in enumerate(measured)
            ]
            st.dataframe(rows)
            numeric = [
                key for key in metrics
                if any(
                    isinstance(values.get(key), (int, float))
                    and not isinstance(values.get(key), bool)
                    and math.isfinite(values[key])
                    for values in measured
                )
            ]
            if numeric:
                metric = st.selectbox("Numeric metric to plot", numeric)
                chart_rows = [
                    {"Condition": f"{index + 1}. {names[index]}",
                     "Value": values[metric]}
                    for index, values in enumerate(measured)
                    if isinstance(values.get(metric), (int, float))
                    and not isinstance(values.get(metric), bool)
                    and math.isfinite(values[metric])
                ]
                st.bar_chart(chart_rows, x="Condition", y="Value")
                st.caption(
                    f"Stored metric: {metric}. Only finite numeric values are plotted; "
                    "text and missing values are omitted."
                )
        st.subheader("Stored artifact plots")
        show_plots(data.get("artifacts", {}))
        with st.expander("Artifact references"):
            show_data(data.get("artifacts"))

    with validation:
        st.header("Consistency & Validation")
        st.info(
            "These are stored simulation checks, not checks rerun by this dashboard "
            "and not evidence of experimental validation."
        )
        checks = list(check_fields(data))
        if not checks:
            st.info("No overview checks or validation fields reported.")
        for label, value in checks:
            st.subheader(label)
            show_data(value)
        st.subheader("Available source summaries and checks")
        if not sources:
            st.info("No source summaries referenced.")
        for reference, (summary, source_error) in sources.items():
            with st.expander(reference):
                if source_error:
                    st.info(source_error)
                    continue
                source_checks = list(check_fields(summary))
                if not source_checks:
                    st.caption("No explicitly named checks or validation fields.")
                for label, value in source_checks:
                    st.write(label)
                    show_data(value)
                st.write("Complete stored summary")
                st.json(summary)

    with about:
        st.header("About / Scope")
        st.write(
            "Presentation of controlled reduced-order motor simulation studies only. "
            "This dashboard does not run simulations, modify results, train a "
            "classifier, or diagnose a physical machine."
        )
        st.write(
            "Literature-example parameters are not identified machine measurements. "
            "A proxy is not a bar-resolved or experimentally calibrated fault model. "
            "Condition-specific classifications and limitations are retained in "
            "the explorer. No digital-twin or predictive-maintenance claim is made."
        )
        st.subheader("Stored provenance")
        show_data(data.get("provenance"))
        st.subheader("Stored scope note (verbatim)")
        show_data(data.get("scope_note"))
        st.caption(
            "The stored scope note describes the source report; this dashboard "
            "adds a read-only viewer, not additional model or validation capability."
        )
        st.write(
            "All paths resolve relative to the repository containing this file. "
            "Legacy absolute results paths are relocated to this repository's "
            "results directory. External paths and remote plots are not opened. "
            "Only referenced local raster images are rendered; raw trace archives "
            "are not loaded. Missing or unreadable files remain unavailable."
        )


if __name__ == "__main__":
    main()