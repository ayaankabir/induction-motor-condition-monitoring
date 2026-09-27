# Experiments

Numbered simulation studies and deterministic engineering report workflows.
All simulation outputs are **simulated** unless a later dataset is added with provenance.

## Condition Taxonomy

The repository preserves a strict scientific distinction between true faults and operational/supply confounders:

- **Healthy baseline:** DOL start from rest, nominal Krause parameters.
- **Fault 01:** Stator resistance imbalance (+10% phase-A resistance; high-resistance connection proxy).
- **Condition 02:** Increased mechanical load (50% higher constant load torque; operational change confounder).
- **Condition 03:** Supply voltage unbalance (phase-C at 0.9 p.u.; power supply confounder).
- **Fault 04:** Simulated bearing outer-race vibration signature (BPFO vibration channel; electrical currents bit-for-bit healthy).
- **Fault 05:** Rotor electrical asymmetry proxy for broken-bar behavior ($2sf_s$ modulation; not bar-resolved, not severity-calibrated).

## Running Simulation Studies

Activate the virtual environment before running:

```bash
source .venv/bin/activate
```

### Healthy start-up baseline

```bash
python experiments/run_healthy_startup.py
```

Writes `results/healthy_startup.npz`, `results/healthy_startup_summary.json`,
`results/healthy_startup_provenance.json`, and start-up PNG figures. Parameters used
are the literature-example machine, not laboratory measurements.

### Matched condition studies

Each script runs a matched healthy/fault pair with identical solver settings and writes
a `.npz` trace archive, a summary JSON, and a comparison PNG under `results/`:

- `python experiments/run_fault_01_stator_resistance_imbalance.py`
- `python experiments/run_fault_02_increased_mechanical_load.py`
- `python experiments/run_fault_03_supply_voltage_unbalance.py`
- `python experiments/run_fault_04_bearing_outer_race.py`
- `python experiments/run_fault_05_rotor_asymmetry.py`

### Condition overview summary

Generate the unified overview comparison and cross-condition table:

```bash
python experiments/run_fault_overview.py
```

Writes `results/fault_overview_summary.json`, `results/fault_overview_comparison.png`,
and `results/fault_overview_table.md`.

## Deterministic Engineering Report Workflow

To compile comprehensive read-only engineering report artifacts from the tracked
simulation summaries without re-running ODE integrations:

```bash
python -m imcm.reporting.generate_report \
  --output-dir build/report/ \
  --generated-at-utc "2026-01-01T00:00:00+00:00" \
  --results-dir results/
```

This CLI deterministic command generates four artifacts in the target directory:
- `imcm_engineering_report.json`
- `imcm_engineering_report.html`
- `imcm_engineering_report.pdf`
- `imcm_engineering_report_manifest.json`
