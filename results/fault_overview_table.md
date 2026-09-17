# Simulated condition overview — healthy vs fault comparison

All values are **simulated** on the illustrative 4 kW, 400 V, 50 Hz, 4-pole **literature-example** machine (constant 15.0 N m load unless stated, DOL start from rest, RK45, `max_step=1e-4 s`). There is **no experimental validation**; nothing here is a diagnosis, classifier, or predictive-maintenance claim.

## 1. Shared simulated metrics

| Metric | Healthy | Fault 01 | Cond. 02 | Fault 03 | Fault 04 | Fault 05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final rotor speed (r/min) | = healthy | 1464.61 * | 1445.61 | 1461.62 * | = healthy (bit-identical) | 1461.84 * |
| Final slip (p.u.) | 0.02376 | 0.02359 | 0.03626 | 0.02559 | = healthy (bit-identical) | 0.02544 |
| Phase RMS i_a/i_b/i_c (A) | (5.565, 5.565, 5.565) | (5.532, 5.630, 5.535) | not studied | (7.007, 6.412, 3.733) | = healthy (bit-identical) | not studied |
| Current unbalance (%) | 1.43e-06 | 1.759 | 0 (balanced by construction) | 57.27 | = healthy (bit-identical) | not studied |
| Negative-sequence current (A) | 4.651e-08 | 0.06444 | not studied | 1.91 | = healthy (bit-identical) | not studied |
| Torque ripple RMS (N m) | 6.27e-06 | 0.1901 | not studied | 5.417 | = healthy (bit-identical) | 0.05443 |
| Power-balance residual (rel.) | 9.44e-15 | 3.98e-08 | 2.48e-14 | 8.88e-07 | 9.44e-15 | 4.07e-13 |

\* Final speed/slip for Faults 01/03/05 are derived as `healthy + stored difference` (those summaries store differences only). "Bit-identical" cells: the study does not modify the electrical plant (verified elementwise for Fault 04). "Not studied": the metric is not the studied channel and is not stored in that study's summary.

## 2. Primary simulated signature per condition

| Condition | Affected signal | Expected signature | Measured (healthy -> condition) |
| --- | --- | --- | --- |
| Fault 01 — stator resistance imbalance (+10% phase A) | Stator currents (sequence components), qd currents, torque ripple | Nonzero negative-sequence current, phase current unbalance, and qd/torque ripple at twice supply frequency. | negative-sequence current: 4.651e-08 A -> 0.06444 A (1.158% of positive sequence) |
| Condition 02 — increased mechanical load (+50%, 15.0 -> 22.5 N m) | Speed, slip, torque, current magnitude (i_qs, phase RMS) | Lower speed, higher slip, proportionally higher mean torque and current; currents remain balanced. | slip: 0.02376 -> 0.03626 (+50% load); mean i_qs: 5.223 A -> 7.782 A |
| Fault 03 — supply voltage unbalance (phase C at 0.9 p.u.) | Supply voltages, phase currents, torque ripple at 2*omega_e | Voltage unbalance factor ~3.45%, large current unbalance, negative-sequence current, second-harmonic torque ripple. | current unbalance: 1.433e-06% -> 57.27%; negative sequence: 1.91 A (34.27%) at 3.448% VUF |
| Fault 04 — bearing outer-race fault (BPFO vibration signature) | Simulated accelerometer channel only (m/s^2) | Envelope-spectrum lines at BPFO (~87.5 Hz at the simulated steady speed) and its harmonics; healthy channel is exactly zero by construction. | envelope amplitude at BPFO (87.49 Hz): 0 -> 0.1527 m/s^2 (2 x BPFO: 0.09547 m/s^2); electrical traces bit-identical to healthy |
| Fault 05 — rotor electrical asymmetry, severity 0.10 (broken-bar proxy) | Stator current spectrum (sidebands), qd modulation tone, small torque ripple | Sidebands near f_s(1 - 2s) ~ 47.6 Hz and f_s(1 + 2s) ~ 52.4 Hz; synchronous-frame tone at 2 s f_s ~ 2.38 Hz; mean operating point essentially unchanged. | phase-a sidebands at f_s(1-/+2s): 0.0002624 -> 0.2243 A at 47.62 Hz and 0.2282 A at 52.38 Hz; 2 s f_s tone 0.4524 A |

## 3. Classification: physically modelled vs proxy vs confounder

| Condition | Classification |
| --- | --- |
| Healthy baseline (balanced 50 Hz supply, 15.0 N m load) | Reference baseline — no fault |
| Fault 01 — stator resistance imbalance (+10% phase A) | Physically modelled in the reduced-order plant (proxy for a high-resistance connection/joint; NOT a turn fault) |
| Condition 02 — increased mechanical load (+50%, 15.0 -> 22.5 N m) | Operating-condition change, NOT an internal motor fault (confounder for current-magnitude and slip features) |
| Fault 03 — supply voltage unbalance (phase C at 0.9 p.u.) | Supply-side confounder, physically modelled supply; motor equations and parameters unchanged (NOT winding damage) |
| Fault 04 — bearing outer-race fault (BPFO vibration signature) | Simulation-only vibration-channel signature (bearing kinematics + assumed resonance); electrical plant bit-for-bit healthy — NOT a motor-model fault, NOT an MCSA bearing claim |
| Fault 05 — rotor electrical asymmetry, severity 0.10 (broken-bar proxy) | Simulation-only proxy for broken-bar-related behaviour (rotor-frame axis resistance split); NOT bar-resolved, NOT severity-calibrated, NOT a real-machine diagnosis |

## 4. Automated consistency checks (regenerated outputs, current code)

| Check | Status | Detail |
| --- | --- | --- |
| `provenance_tags` | pass | every summary tags parameters as literature_example and traces as simulated |
| `healthy_columns_consistent` | pass | max |healthy phase RMS (fault_01) - (fault_03)| = 0.000e+00 A (all studies regenerated with the current code) |
| `fault_04_electrical_untouched` | pass | fault_04 npz: healthy_i_abc == fault_i_abc elementwise |
| `traces_finite` | pass | all npz arrays finite |
| `power_balance_healthy` | pass | worst healthy relative residual = 9.44e-15 (< 1e-10) |
| `power_balance_faulted` | pass | worst faulted relative residual = 8.88e-07 (< 1e-05; np.gradient artifact, converges as dt^2) |
| `fault_03_signature` | pass | VUF = 3.448% (expected ~3.448%), I2 = 1.910 A |
| `fault_04_signature` | pass | BPFO = 87.49 Hz, measured envelope peak at 87.59 Hz, healthy envelope line = 0 (idealised: zero vibration floor) |
| `fault_05_signature` | pass | sideband bins at f_s(1-2s_ref) = 47.62 Hz; fault/healthy amplitude ratio = 9e+02 |

## 5. Standing limitations (apply to every row above)

- Parameters are a published illustrative example, **not** a measured motor; all traces are deterministic ODE outputs with no noise model, so every separation shown is idealised.
- Unbalance / negative-sequence / current-magnitude metrics are **not unique discriminants**: load (Cond. 02), supply (Fault 03), and stator resistance (Fault 01) all move them. The studies are controlled contrasts, not diagnosis.
- Healthy torque-ripple floors (~1e-5 N m and below) are numerical, not physical.
- No GUI, no ML classifier, no predictive-maintenance or digital-twin claim is made anywhere in this repository.

Per-condition detail: docs/fault_01_*.md, docs/fault_02_*.md, docs/fault_03_*.md, docs/fault_04_*.md, docs/fault_05_*.md; narrative overview: docs/fault_summary.md.
