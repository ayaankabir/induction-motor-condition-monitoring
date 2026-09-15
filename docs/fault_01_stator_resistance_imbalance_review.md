# Fault 01 implementation: strict code and scientific review

## Scope

This is a review of the implemented phase-A +10% stator-resistance imbalance.
It evaluates internal mathematical consistency and controlled simulation
behavior only. It does not assess agreement with experimental motor data.

## Findings

| # | Review item | Classification | Evidence and conclusion |
| --- | --- | --- | --- |
| 1 | Healthy-model preservation | **PASS** | The disabled configuration returns `(Rs, Rs, Rs)` at `src/imcm/models/stator_resistance.py:15-22`. Equal phase values take the original scalar drop path at `fifth_order_dq.py:99-110`. `tests/test_fault_01_stator_resistance_imbalance.py:34-45` compares speed, `qd` currents, torque, and reconstructed phase currents against the healthy case at `rtol=0`, `atol=1e-12`. Existing healthy tests remain meaningful because the default scenario uses the disabled configuration (`operating_scenario.py:73-99`). |
| 2 | Explicit phase resistances | **PASS** | `StatorResistanceConfig` stores three multipliers at `operating_scenario.py:17-32`. Fault 01 explicitly sets `(1.10, 1.0, 1.0)` at `:102-123`; multiplication by the healthy `Rs=1.405` produces `(1.5455, 1.405, 1.405)` ohm. This is checked at `tests/...imbalance.py:22-31`. It does not change scalar `params.r_s`. |
| 3 | abc-to-`qd` resistance projection | **PASS** | `resistance_matrix_qd` at `stator_resistance.py:25-39` implements the existing Krause rows, with phase shifts `(0, -2π/3, +2π/3)` and factor `2/3`, restricted to `i0=0`. It is `K_qd diag(Ra,Rb,Rc) K_qd^{-1}` on that subspace. The matrix changes with `theta` unless all phases are equal, as physically expected. Direct abc-transform checks at three angles gave maximum error `8.9e-16`; the code test verifies the same construction at `tests/...imbalance.py:48-56`. |
| 4 | Electrical equations, signs, and units | **PASS** | The projected resistance drop is subtracted in the stator flux ODEs at `fifth_order_dq.py:104-110`, matching the healthy voltage-equation sign. Resistance times current is volts, so units are correct. Equal values reduce to `R i` exactly (`stator_resistance.py:30-31` and `fifth_order_dq.py:104-106`). Rotor, torque, and mechanics equations remain unchanged. |
| 5 | RMS and sequence metrics | **MINOR CONCERN** | Both cases use the same asserted output grid (`fault_metrics.py:68-80`) and one `t_start`, so comparisons are matched. RMS imbalance is explicit at `:35-42`; torque ripple removes the mean at `:62-65`. The phase convention is correct: with the repository's A-B-C voltages, `I1=(Ia+a Ib+a² Ic)/3` and `I2=(Ia+a² Ib+a Ic)/3` at `:51-58`. However, the `t >= 0.8` mask includes both 0.8 and 1.0 seconds: 2001 samples span ten cycles but duplicate the endpoint phase in a simple arithmetic mean. This is small here but makes the stated “integer-cycle” estimate not an exact discrete-period average. Prefer a half-open window or integration weighting in a future refinement. |
| 6 | Fault-aware power balance | **PASS** | `power_balance.py:38-66` calculates unequal stator copper loss as `(3/2) i_qd^T R_qd i_qd` and retains rotor copper loss with the correct Krause scaling. All terms are watts: electrical input, copper loss, `Te*omega_m`, and magnetic-energy rate. The Fault 01 test checks late-window residual `<1e-6` at `tests/...imbalance.py:71-75`; the observed residual was `3.98e-8`. The balanced T-equivalent-circuit assertion is not run on the fault case. |
| 7 | Numerical behavior and convergence | **MINOR CONCERN** | Fault runs are finite and phase currents remain finite (`tests/...imbalance.py:59-68`); no NaN, Inf, discontinuity, or RK45 failure was observed. Re-running with `max_step=1e-5`, `rtol=1e-9`, `atol=1e-11` reproduced default fault metrics: unbalance `1.7847271656%`, negative sequence `0.06697154838 A`, torque ripple `0.1900506905 N m`, and final-speed difference `0.2412816264 r/min`, unchanged at displayed precision. This convergence check is currently manual rather than an automated regression test. |
| 8 | Physical interpretation | **PASS** | The observed increase from `0.0430%` to `1.7847%` RMS current unbalance and from `0.00278 A` to `0.06697 A` negative sequence is plausible for a controlled phase-fixed resistance increase. Small mean speed change is appropriately not over-interpreted: the model holds supply/load fixed and the resistance perturbation primarily changes electrical balance. Physical scope and non-identifiability are accurately limited at `docs/fault_01_stator_resistance_imbalance.md:72-100`. |
| 9 | Test quality | **MINOR CONCERN** | Tests do verify behavior, not merely execution: exact balanced regression (`:34-45`), abc projection (`:48-56`), finite three-wire fault response and changed current/sequence metrics (`:59-68`), and fault-aware power balance (`:71-75`). They do not yet automate multi-angle projection coverage, metric convergence under tightened solver settings, or the half-open/integration-weighted metric-window convention. These are useful hardening additions, not evidence of a current model defect. |

## Required fixes

No **IMPORTANT ISSUE** or blocking fix was found. The two minor concerns are
test/estimator hardening items: use an unambiguous non-duplicated metric window,
and automate the already successful tolerance/`max_step` convergence check.

## Test record

The requested project-environment test command was run after this review:

```bash
.venv/bin/python -m pytest -q
```

Result: **27 passed**.

## Commit readiness

The implementation is ready for a Git commit from the standpoint of electrical
signs, units, healthy preservation, numerical stability, and stated scientific
scope. The two minor metric/test refinements above are recommended follow-up
work, not prerequisites for this controlled simulation implementation.
