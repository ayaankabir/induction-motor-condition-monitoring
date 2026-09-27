# Induction Motor Condition Monitoring Using a Reduced-Order Model and Electrical Signal Processing

Student engineering project. **Simulation-first, not a digital twin.**
No experimental measurements exist in this repository.

## Project Structure

This repository is the Python/backend side of the project, containing the induction-motor simulation, signal analysis, diagnostics, API, tests, and engineering reporting. The current portfolio-facing dashboard is maintained in a separate Next.js frontend repository.

## What this project is

A modular Python workspace to:

1. Integrate a **fifth-order two-axis (dq) squirrel-cage induction machine** using a single classical Krause Park convention.
2. Reconstruct **healthy** stator currents from that model with explicit stated assumptions.
3. Simulate a **small, selected set of fault and operating conditions** using physically motivated parameter or circuit modifications:
   - **Healthy baseline:** DOL startup from rest to steady state
   - **Fault 01:** Stator resistance imbalance (+10% phase A; high-resistance connection proxy)
   - **Condition 02:** Increased mechanical load (+50% constant torque; operational change confounder)
   - **Condition 03:** Supply voltage unbalance (phase C at 0.9 p.u.; supply-side confounder)
   - **Fault 04:** Simulated bearing outer-race vibration signature (BPFO vibration channel; electrical currents bit-for-bit healthy)
   - **Fault 05:** Rotor electrical asymmetry proxy for broken bar behavior ($2sf_s$ modulation; simulation proxy, not bar-resolved)
4. Apply **electrical signal processing** (primarily motor current signature analysis and envelope analysis) with claims matched to model fidelity.
5. Provide structured provenance metadata, diagnostic triage logic, a read-only Streamlit dashboard, a local REST API, and deterministic engineering-report generation (JSON, HTML, PDF, manifest).

## What this project is not

- Not FEA or a winding-function replica of a specific motor
- Not a complete commercial digital twin or turnkey condition-monitoring product
- Not a source of fabricated lab data
- **Not experimentally validated** — all machine traces are ODE simulations
- Not an identified model of a real motor; machine parameters are an illustrative literature benchmark

## Locked decisions (modeling & physics)

| Topic | Choice |
| --- | --- |
| Park / torque | Classical **Krause** ($2/3$) transform and $\tfrac{3}{2}$ torque. Not power-invariant. Do not mix conventions. |
| Plant | Fifth-order (4 electrical + 1 mechanical) flux model in the synchronous ($dq$) frame; squirrel cage ($v_{qr}=v_{dr}=0$). |
| Supply | Balanced sinusoids, **50 Hz**, 400 V line-line RMS, star (or explicit asymmetric supply in Condition 03); **no inverter**. |
| Load | Constant load torque ($T_L=15\,\mathrm{N\cdot m}$ nominal; stepped to $22.5\,\mathrm{N\cdot m}$ in Condition 02). |
| Parameters | Illustrative **4 kW, 400 V, 50 Hz, 4-pole** literature example — **not measured**. |
| Integrator | Adaptive **RK45**, `max_step = 10^{-4}\,\mathrm{s}`. |
| Initial condition | **Start from rest** (zero flux, $\omega_m=0$), direct-on-line voltage application. |
| Broken rotor bar | **No bar-resolved discrete cage model.** Documented **simulation-only proxy** (Fault 05, rotor electrical asymmetry); see [`docs/fault_05_rotor_asymmetry.md`](docs/fault_05_rotor_asymmetry.md). |
| Bearing vibration | Outer-race ball-pass frequency proxy generator (Fault 04) producing an auxiliary synthetic vibration channel. |
| Scope | Simulation-only unless real data are added with provenance. No experimental-validation claims. |

Full equations, identities, tests, and limitations: [`docs/modeling-plan.md`](docs/modeling-plan.md).
Scientific rules: [`AGENTS.md`](AGENTS.md).

## Package Architecture

```
src/imcm/
├── analysis/     # Spectral analysis, FFT, and MCSA sideband extraction
├── data/         # Provenance schemas, trace containers, and CSV importer
├── faults/       # Fault catalog and Fault 04 bearing vibration proxy generator
├── models/       # Parameters, Krause ODEs, flux maps, and fault modifications
├── processing/   # Envelope demodulation (Hilbert transform)
├── reporting/    # Engineering report generator, HTML/PDF renderers, triage, and provenance
├── signals/      # Krause abc <-> qd0 reference frame transformations
└── validation/   # Power balance, equivalent circuit, signal quality, and metrics
```

## Results & Reproducibility Tracking

The repository tracks lightweight summary and provenance metadata under `results/`:

- **7 tracked summary JSON files:**
  - `healthy_startup_summary.json`
  - `fault_01_stator_resistance_imbalance_summary.json`
  - `fault_02_increased_mechanical_load_summary.json`
  - `fault_03_supply_voltage_unbalance_summary.json`
  - `fault_04_bearing_outer_race_summary.json`
  - `fault_05_rotor_asymmetry_summary.json`
  - `fault_overview_summary.json`
- **2 tracked provenance manifests:**
  - `healthy_startup_provenance.json`
  - `fault_01_stator_resistance_imbalance_provenance.json`

Heavy simulation trace archives (`.npz`), plots (`.png`), and generated reports (`.html`, `.pdf`) are excluded via `.gitignore` to keep the repository lightweight and portable. Any missing run-configuration or additional provenance files are intentionally not backfilled by rerunning simulations — this preserves repository portability without claiming complete historical provenance for every run. All simulation results can be reproduced by running the corresponding experiment scripts.

## Installation & Testing

```bash
# Optional: create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install editable package with dev dependencies
pip install -e ".[dev]"

# Run test suite
python -m pytest tests/
```

## Running Simulations

```bash
# Run healthy start-up baseline
python experiments/run_healthy_startup.py

# Run condition studies
python experiments/run_fault_01_stator_resistance_imbalance.py
python experiments/run_fault_02_increased_mechanical_load.py
python experiments/run_fault_03_supply_voltage_unbalance.py
python experiments/run_fault_04_bearing_outer_race.py
python experiments/run_fault_05_rotor_asymmetry.py

# Run unified overview comparison
python experiments/run_fault_overview.py
```

## Generating Deterministic Engineering Reports

Compile standalone engineering reports from existing simulation results:

```bash
python -m imcm.reporting.generate_report \
  --output-dir build/report/ \
  --generated-at-utc "2026-01-01T00:00:00+00:00" \
  --results-dir results/
```

This generates:
- `imcm_engineering_report.json`
- `imcm_engineering_report.html`
- `imcm_engineering_report.pdf`
- `imcm_engineering_report_manifest.json`

## Interactive Dashboard & API

`dashboard/app.py` is the original read-only Streamlit dashboard for development. The current portfolio UI is the separate Next.js frontend maintained in [motor-simulation-data-comparison](https://github.com/ayaankabir/motor-simulation-data-comparison).

```bash
# Launch read-only Streamlit dashboard
streamlit run dashboard/app.py

# Launch local REST API service
uvicorn api.main:app --reload
```

## Academic Use & Citations

For coursework and research notes. Cite Krause, Ong, and MCSA survey literature. Do not copy restricted datasets into `data/` without provenance.
