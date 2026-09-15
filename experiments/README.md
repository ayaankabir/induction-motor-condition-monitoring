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
