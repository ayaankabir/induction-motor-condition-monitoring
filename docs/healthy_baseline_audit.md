# Healthy induction-motor baseline: scientific audit

## Scope and method

This is a source-level, simulation-consistency audit of the healthy baseline.
It assesses the stated reduced-order model and its internal checks only; it does
not compare the illustrative parameter set or simulated traces with experimental
motor data.  No implementation changes are proposed or made here.

## Findings

| # | Audit item | Classification | Evidence and conclusion |
| --- | --- | --- | --- |
| 1 | Park convention and inverse | **PASS** | `src/imcm/signals/park.py:14-34` implements the classical Krause amplitude-invariant `qd0` transform with factor \(2/3\); `:37-48` is its matching inverse, including \(f_0\). `tests/test_park.py:7-13` verifies arbitrary three-component round trip, and `tests/test_healthy_simulation.py:73-81` verifies the same round trip on the simulated currents. The model uses the matching \(3/2\) power/torque scaling. |
| 2 | Phase-current reconstruction | **PASS** | Reconstruction sets \(i_0=0\) and applies the same inverse transform at `src/imcm/models/fifth_order_dq.py:130-133`. This is consistent with the balanced, isolated-neutral assumption. `tests/test_healthy_simulation.py:67-70` verifies \(i_a+i_b+i_c=0\), and `:73-81` recovers the original \(i_{qs},i_{ds}\). |
| 3 | Electrical angular frequency and reference angle | **PASS** | `src/imcm/models/operating_scenario.py:26-33` defines \(\omega_e=2\pi f\) and the correct star phase peak. The plant uses it in the voltage/rotor-speed terms at `fifth_order_dq.py:92-102`; reconstruction uses \(\theta=\omega_e t\) at `:130-133`. The balanced-voltage identity is verified in `tests/test_park.py:16-26` and `test_healthy_simulation.py:84-90`. |
| 4 | Mechanical-speed units | **PASS** | The state is mechanical angular speed in rad/s: electrical rotor speed is correctly \(\omega_r=(P/2)\omega_m\) at `fifth_order_dq.py:60-67` and `:95`. The reporting-only conversion to r/min is \(60\omega_m/(2\pi)\) at `:69-71`. Synchronous-speed metadata uses \(120f/P\) at `parameters.py:70-72`. |
| 5 | Slip definition and sign | **PASS** | `fifth_order_dq.py:64-67` implements \(s=(\omega_e-\omega_r)/\omega_e\), with \(\omega_r=(P/2)\omega_m\). Thus below-synchronous motoring has positive slip, as checked at `tests/test_healthy_simulation.py:93-103`. A temporary negative slip is correctly an overspeed under this convention. |
| 6 | Electromagnetic torque sign and units | **PASS** | `src/imcm/models/flux_map.py:49-57` uses \(T_e=(3/2)(P/2)(\lambda_{ds}i_{qs}-\lambda_{qs}i_{ds})\). Weber-ampere is N m, and the positive sign is used consistently in the mechanical equation. The equivalent current form is implemented at `:60-68` and algebraically checked by `tests/test_flux_map.py:20-28`. |
| 7 | Mechanical equation and load interpretation | **MINOR CONCERN** | The equation at `fifth_order_dq.py:106-110` is \(J\dot\omega_m=T_e-T_L-B\omega_m\), which is internally sign-consistent and its steady balance is checked at `tests/test_healthy_simulation.py:93-103`. However, `T_L` is a fixed positive signed torque even at zero or negative speed. With zero initial flux this gives a brief reverse acceleration. This is a deliberate scenario simplification, documented in `docs/healthy_baseline_review.md:80-87,101-106`; it is not a defect in the electrical signs, but it is not a passive-load/stiction law. |
| 8 | Balanced, approximately 50 Hz steady currents | **PASS** | Zero sequence is excluded in reconstruction (`fifth_order_dq.py:132-133`), phase balance is tested at `tests/test_healthy_simulation.py:67-70`, and the late 0.4 s window is checked by both FFT-peak and zero-crossing estimators at `:135-148`. `src/imcm/validation/frequency.py:8-28,31-46` provides the two independent numerical estimators. |
| 9 | Approximately DC synchronous-frame currents | **PASS** | A synchronous supply frame makes the balanced supply DC by construction; the plant uses constant \(v_{qs}\), \(v_{ds}=0\) at `fifth_order_dq.py:92-94`. The late-window standard-deviation tests at `tests/test_healthy_simulation.py:106-111` verify that \(i_{qs}\) and \(i_{ds}\) settle approximately to constants. |
| 10 | Power-balance check | **PASS** | `src/imcm/validation/power_balance.py:28-55` uses the Krause-consistent input power, copper losses, electromagnetic mechanical conversion \(T_e\omega_m\), and derivative of the correctly scaled linear magnetic co-energy. Its late-window mean residual check at `tests/test_healthy_simulation.py:114-117` is logically valid for steady-state internal consistency. It does not validate parameters or neglected physics. |
| 11 | Equivalent-circuit check | **PASS** | `src/imcm/validation/equivalent_circuit.py:25-56` is a steady-state RMS T-equivalent circuit using the same stator-referred parameters, supply frequency, and slip. Comparing its torque/current with the late nearly steady ODE window at `tests/test_healthy_simulation.py:120-132` is logically valid as an algebraic cross-check. It is not an independent experimental validation, and it should not be applied to the startup transient. |

## Cross-check of documentation and experiment entry point

`experiments/run_healthy_startup.py:37-65` uses the audited default 1 s RK45
simulation, takes the late \(t\geq0.8\) s window, and reports speed, slip,
torque balance, power balance, and equivalent-circuit values with their correct
roles.  Its provenance statement at `:67-70` agrees with
`docs/healthy_baseline_review.md:91-106`: the traces are simulated and the
parameters are literature-example assumptions, not measurements.

## Overall conclusion

The healthy baseline passes the requested internal scientific-consistency audit.
There are no important issues in the transform, units, speed/slip convention,
torque convention, reconstruction, or stated steady-state validation logic.
The one minor concern is the fixed signed constant-load law through standstill
and reverse motion; it should remain documented as a limitation until a future,
explicitly approved load-law change.

## Test record

The project test suite was run after this audit with:

```bash
.venv/bin/python -m pytest -q
```

Result: **22 passed**.
