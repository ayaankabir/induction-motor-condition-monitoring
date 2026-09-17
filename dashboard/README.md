# Read-only condition dashboard

Run from the repository root using the existing virtual environment:

```sh
.venv/bin/streamlit run dashboard/app.py
```

The dashboard reads `results/fault_overview_summary.json` and available referenced condition summaries and local artifact plots. Paths are resolved relative to the repository, including legacy absolute artifact paths under `results/`. Missing or unreadable files are reported without generating replacements.

## Sections

1. Overview
2. Condition Explorer
3. Cross-Condition Comparison
4. Consistency & Validation
5. About / Scope

Condition fields, comparison metrics, and stored checks are discovered from the JSON. Missing measurements are not treated as zero. Comparisons align identical stored field names; they do not infer equivalent metrics or convert units.

This viewer does not run simulations or tests, modify results, or load raw trace archives. Displayed checks are previously recorded results, not newly executed validation. Data are simulated with literature-example parameters, not experimental measurements. Fault proxies retain their limitations; the dashboard makes no real-machine diagnostic, digital-twin, or predictive-maintenance claims.