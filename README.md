# Induction Motor Condition Monitoring Using a Reduced-Order Model and Electrical Signal Processing

Student engineering project. **Simulation-first, not a digital twin.**
No experimental measurements exist in this repository.

## What this project is

A modular Python workspace to:

1. Integrate a **fifth-order two-axis (dq) squirrel-cage induction machine** using a **single** classical Krause Park convention.
2. Reconstruct **healthy** stator currents from that model (phase 1, implemented).
3. Later add a **small set of physically motivated fault representations**, including only a **documented proxy** if broken-bar-like spectra are studied.
4. Apply **electrical signal processing** (mainly MCSA) with claims matched to model fidelity.

## What this project is not

- Not FEA or a winding-function replica of a specific motor
- Not a complete condition-monitoring product
- Not a source of fabricated lab data
- **Not experimentally validated** — traces are ODE simulations
- Fault simulation and a dashboard are **not** implemented yet

## Locked decisions (healthy plant)

| Topic | Choice |
| --- | --- |
| Park / torque | Classical **Krause** \(2/3\) transform and \(\tfrac{3}{2}\) torque. Not power-invariant. Do not mix. |
| Plant | Healthy fifth-order flux model only |
| Supply | Balanced sinusoids, **50 Hz**, 400 V line-line RMS, star; **no inverter** |
| Load | **Constant** \(T_L=15\,\mathrm{N\cdot m}\) (assumed scenario) |
| Parameters | Illustrative **4 kW, 400 V, 50 Hz, 4-pole** literature example — **not measured** |
| Integrator | Adaptive **RK45**, `max_step = 10^{-4}\,\mathrm{s}` |
| Initial condition | **Start from rest** (zero flux, \(\omega_m=0\)), direct-on-line voltage |
| Broken rotor bar | **Not implemented.** Future work may use a labeled proxy or coupled circuits |
| Scope | Simulation-only |

Full equations, identities, tests, and limitations: [`docs/modeling-plan.md`](docs/modeling-plan.md).

Agent rules: [`AGENTS.md`](AGENTS.md).

## Current repository status

Phase 1 **healthy plant** is implemented:

- Documented assumed machine and operating scenario in `imcm.models`
- Krause Park transforms and linear flux maps (unit-tested)
- Time-domain RK45 integration of the fifth-order healthy ODEs
- Start-up figures and compressed traces under `results/` after you run the experiment script

Empty `data/` does **not** contain measurements. Files in `results/` are **simulated**.

## Proposed architecture

```
supply v_abc(t)          T_L = constant
        │                      │
        ▼                      ▼
┌─────────────────────────────────────┐
│  fifth-order dq healthy plant       │  ← implemented (RK45, start from rest)
│  Krause synchronous frame           │
└─────────────────┬───────────────────┘
                  │
                  ▼
        i_abc, ω_m, T_e   (label: simulated, healthy)
                  │
                  ▼
        later: fault wrappers, MCSA, reports
```

| Package | Responsibility |
| --- | --- |
| `imcm.models` | Parameters, scenario, flux map, healthy ODEs |
| `imcm.signals` | Krause \(abc\leftrightarrow qd0\) |
| `imcm.validation` | Power balance and equivalent-circuit checks |
| `imcm.reporting` | Static start-up figures (not a dashboard) |
| `imcm.faults` | Planned identifiers only; **no fault physics** |
| `imcm.processing` | Spectra (later) |

## Modeling approach (short)

**Healthy machine:** fifth-order flux-linkage model in the **synchronous** \(dq\) frame;
squirrel-cage \(v_{qr}=v_{dr}=0\); linear \(L_s,L_r,L_m\); inertia and viscous friction.

**Why this order:** electrical currents are the sensors. Third-order and first-order
models cannot support this project. FEA and per-bar circuits are higher fidelity than
we will claim.

**Why Krause, not power-invariant:** matches Krause/Ong bookkeeping; balanced supply
maps to \(v_{qs}=V_s\), \(v_{ds}=0\); torque uses \(\tfrac{3}{2}\). Changing convention
requires rewriting transforms, torque, and power tests together.

**Faults:** not in this milestone. When added, change the equations; do not paste
MCSA sidebands onto healthy currents. A two-axis rotor-asymmetry model is a **proxy**,
not a complete broken-bar simulation.

## How to install, test, and run the healthy simulation

From the repository root:

```bash
source .venv/bin/activate   # if the local venv exists
pip install -e ".[dev]"
python -m pytest
python experiments/run_healthy_startup.py
```

The experiment writes:

- `results/healthy_startup.npz` — simulated time series
- `results/healthy_startup_summary.json` — solver metadata and sanity numbers
- `results/healthy_speed.png`, `healthy_torque.png`, `healthy_currents.png`,
  `healthy_i_qd.png`, `healthy_slip.png`

Do not look in `data/` for measurements; there are none.

## Development roadmap

| Phase | Goal | Exit criterion |
| --- | --- | --- |
| **0 — Foundation** | Docs, conventions, assumed parameters, equation tests | Done |
| **1 — Healthy plant** | Time-domain fifth-order model, \(abc\) currents | RK45 start-up; slip/torque vs equivalent circuit; balanced currents |
| **2 — Numerical hygiene** | Step size, power balance, transform tests on simulated traces | Tests fail if energy or frames break |
| **3 — Voltage unbalance** | Confounder; same healthy ODEs | Negative-sequence current; labeled **supply**, not winding damage |
| **4 — Stator \(R\) unbalance** | Unequal phase resistances | High-resistance connection, **not** turn fault |
| **5 — Rotor asymmetry (optional)** | Documented proxy **or** reduced coupled-circuit | Sidebands near \(f_s(1\pm 2s)\) if proxy; **never** “bar count” |
| **6 — MCSA pipeline** | Windowed spectra, slip-aware bins | Features from simulated traces + metadata only |
| **7 — Optional** | Public dataset or lab plan **if** they exist | Never invent traces |
| **8 — Optional** | Report UI | Visualization of **already computed** results |

Do not start phase \(n+1\) until phase \(n\) is verified. **Do not add faults until approved.**

## Assumptions

- Linear magnetics, sinusoidal MMF, uniform air gap
- Single equivalent rotor cage, stator-referred
- Ideal 50 Hz three-phase sinusoids (PWM deferred)
- Constant inertia, viscous friction, **constant load torque**
- Classical Krause \(qd0\) scaling
- Parameters from a published illustrative machine, not a measured lab motor
- Direct-on-line start from rest (large inrush is expected in the model)

## Remaining uncertainties

- Whether experimental current data will ever exist
- Proxy vs coupled-circuit cage **if** broken-bar work is requested later
- Optional fan-type load comparison after constant-\(T_L\) verification
- Optional near-steady flux initial condition for monitoring-window studies

## License / academic use

For coursework and research notes. Cite Krause, Ong, and MCSA survey literature
when the report is written. Do not copy restricted datasets into `data/` without provenance.
