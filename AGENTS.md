# Agent instructions

This repository is a student engineering project, not a commercial digital twin.

**Title:** Induction Motor Condition Monitoring Using a Reduced-Order Model and Electrical Signal Processing

## Objective

Build a **scientifically defensible** pipeline that:

1. Simulates a **reduced-order squirrel-cage induction motor** (not FEA, not a full winding-function digital twin).
2. Generates **healthy** three-phase electrical signals from that model, with every assumption stated.
3. Simulates a **small, selected set of fault and operating conditions** using physically motivated parameter or circuit modifications:
   - Healthy baseline
   - Fault 01: stator resistance imbalance
   - Condition 02: increased mechanical load / operating-condition change
   - Condition 03: supply voltage unbalance / supply confounder
   - Fault 04: simulated bearing outer-race vibration signature
   - Fault 05: rotor electrical asymmetry proxy
4. Applies **electrical signal processing** (primarily motor current signature analysis) to those signals.
5. Reports what the method **can and cannot** claim through structured provenance, triage, and deterministic engineering reports.

## Mandatory scientific honesty

- Do **not** fabricate experimental data, nameplate measurements, or lab results.
- Do **not** claim a complete digital twin, FEA-equivalent accuracy, or validated plant diagnosis.
- Do **not** present textbook or literature parameter sets as if they were measured on a specific physical motor unless the user later provides those measurements.
- Label every dataset as `simulated`, `literature-example`, or `experimental` (only when real measurements exist).
- If a fault is represented by a **proxy** (for example rotor resistance asymmetry instead of discrete bar circuits), say so in code comments, docs, and UI copy.
- Always distinguish true internal faults from operational or supply confounders (e.g., increased load, supply voltage unbalance).
- Prefer a correct, limited model over a flashy, unjustified one.

## Locked decisions (do not silently replace)

- **Park / torque:** classical Krause \(qd0\) (factor \(2/3\)) and \(\tfrac{3}{2}\) in torque and power. **Not** power-invariant. Never mix conventions.
- **Healthy plant:** fifth-order (4 electrical + 1 mechanical) flux model, synchronous frame, squirrel cage \(v_{qr}=v_{dr}=0\).
- **Supply:** balanced 50 Hz sinusoids (or explicit asymmetric supply in Condition 03); inverter out of scope.
- **Load:** constant torque (stepped in Condition 02).
- **Integrator:** adaptive RK45 (`scipy.integrate.solve_ivp`, `method="RK45"`) with documented `max_step`.
- **Initial condition:** start from rest (zero fluxes, \(\omega_m=0\)). Direct-on-line voltage application.
- **Broken rotor bar:** represented as a documented two-axis **proxy** (Fault 05), not discrete bar circuits.
- **Scope:** simulation-only unless real data are added with provenance. No experimental-validation claims.

## Implementation rules

- Read this file and `docs/modeling-plan.md` before changing models, faults, or claims.
- Keep code **modular**: parameters, healthy plant, fault wrappers, integration, signal processing, and reporting stay in separate packages.
- Numerical solvers, reference-frame transforms, and feature extractors must be unit-testable.
- Default motor parameters are a **published literature example**. They are starting values for development, not identified machine data.

## Language and structure

- Python 3.11+ compatible scientific stack (`numpy`, `scipy`, `matplotlib`, `pytest`).
- Package lives under `src/imcm/`.
- Tests live under `tests/`.
- Equations, assumptions, and limitations live under `docs/`.

## When in doubt

Stop and ask the user. Remaining open choices are listed in `docs/modeling-plan.md` §14 (later data, later BRB fidelity, optional fan-type load).
