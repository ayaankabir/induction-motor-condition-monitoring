# Experiments

Numbered simulation studies. Outputs are **simulated** unless a later dataset
is added with provenance.

## Healthy start-up (phase 1)

```bash
source .venv/bin/activate
python experiments/run_healthy_startup.py
```

Writes `results/healthy_startup.npz`, `results/healthy_startup_summary.json`,
and PNG figures. Parameters used are the literature-example machine, not
laboratory measurements.

## Fault studies

Each `run_fault_0N_*.py` script runs a matched healthy/fault pair with
identical solver settings and writes an `.npz` trace archive, a summary JSON,
and a comparison PNG under `results/`. All outputs are **simulated**.

- `run_fault_01_stator_resistance_imbalance.py` — +10% phase-A stator resistance.
- `run_fault_02_increased_mechanical_load.py` — 50% higher constant load torque.
- `run_fault_03_supply_voltage_unbalance.py` — phase-C supply at 0.9 p.u.
- `run_fault_04_bearing_outer_race.py` — BPFO vibration channel (electrical
  traces bit-for-bit healthy).
- `run_fault_05_rotor_asymmetry.py` — rotor electrical asymmetry **proxy**
  for broken-bar-related behaviour (3 s runs; sidebands near
  `f_s(1 -/+ 2s)`). Not a physically complete or bar-resolved broken-bar
  model; not severity-calibrated; no experimental validation.
