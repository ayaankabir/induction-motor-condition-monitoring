# Condition 02: increased mechanical load (operating-condition change)

## Status and boundary

This is a **controlled operating-condition change**, not a confirmed internal
motor fault. It does **not** alter the motor model equations, motor parameters,
supply, Park convention, or stator resistance. Only the applied constant
mechanical load torque changes. All traces are **simulated**; parameters remain
`literature_example` values. There is no experimental validation in this
repository.

## Controlled parameter definition

| Quantity | Healthy baseline | Condition 02 |
| --- | ---: | ---: |
| Load type | `constant_torque` | `constant_torque` |
| Load torque T_L | 15.0 N m | **22.5 N m** |

The change is exactly +50 percent:

    T_L(C02) = 1.5 * T_L(healthy) = 1.5 * 15.0 = 22.5 N m.

22.5 N m is about 0.84 of the assumed rated torque (about 26.71 N m). This is a
heavily loaded but not speculative "overload" claim; the value is an assumed
numerical scenario, not a measured dynamometer setting.

## Unchanged inputs

- **Supply:** balanced 50 Hz sinusoids, 400 V line-line RMS, star, no inverter.
- **Park convention:** classical Krause qd0, factor 2/3, 3/2 torque, named
  `krause_classical_2_3`.
- **Motor parameters:** the same illustrative 4 kW, 400 V, 50 Hz, 4-pole
  literature-example machine.
- **Stator resistance:** healthy and balanced, `enabled=False`,
  `multipliers_abc=(1.0, 1.0, 1.0)`.
- **Initial condition:** start from rest (zero flux, zero mechanical speed), DOL.

Only `load=LoadConfig(load_type="constant_torque", torque_nm=22.5)` differs from
the approved healthy scenario.

## Simulation protocol

Matched healthy and Condition 02 runs use identical settings: 1.0 s duration,
RK45 with `max_step=1e-4 s`, `rtol=1e-6`, `atol=1e-8`, `1e-4 s` output sampling.
Reporting uses the late window from 0.8 s to 1.0 s.

## Expected effects of a higher constant load

- **Torque:** steady electromagnetic torque rises to meet the larger load, since
  T_e is approximately T_L plus a small friction term.
- **Speed:** the rotor slows; a larger torque is required, so speed drops.
- **Slip:** slip increases because rotor speed falls while synchronous speed is
  fixed.
- **Current:** the machine draws more current to produce more torque, so |i_qs|
  and RMS phase currents increase, roughly tracking the load.
- **Power balance:** stator input rises; the Krause power identity
  P_in = copper + T_e*omega_m + dW_mag/dt still holds because no fault term is
  introduced.

## Simulated observations

From `experiments/run_fault_02_increased_mechanical_load.py`:

| Metric | Healthy (15.0 N m) | Condition 02 (22.5 N m) |
| --- | ---: | ---: |
| Load torque | 15.0 N m | 22.5 N m |
| Final speed | 1464.37 r/min | 1445.61 r/min |
| Final slip | 0.02376 | 0.03626 |
| Late mean T_e | 15.46 N m | 22.95 N m |
| Late mean i_qs | 5.22 A | 7.78 A |
| Power-balance residual | 9.4e-15 | 2.5e-14 |

Speed falls by about 18.75 r/min and slip rises by about 0.0125, while the
machine draws more current and torque to carry the larger load. Currents remain
finite and balanced (i_a + i_b + i_c = 0). The saved comparison plot overlays
phase currents, qd currents, torque, speed, and slip.

## Physical interpretation

Increasing the mechanical load at fixed supply is a normal operating-condition
change. The electrical signature (higher current magnitude, lower speed, higher
slip) resembles what an increasing mechanical load produces on a real machine.
Because supply, parameters, and winding condition are unchanged, any current or
slip change here is attributable to load, not to a winding or rotor defect.

## Assumptions and limitations

- Parameters and torque values are illustrative literature-example assumptions,
  not a measured motor or test-bench condition.
- The model is linear and fundamental-frequency only: no saturation, slotting,
  thermal drift, PWM, or noise.
- Constant load torque; fan-type load proportional to speed squared is deferred.
- Lumped inertia and viscous friction; no torsional dynamics.
- A load change, a supply change, and an internal fault can each raise current;
  this single controlled simulation does **not** establish fault uniqueness or
  enable a classifier.

## Evidence status

These results are **simulation evidence** for this exact operating-condition
change only. They are not experimental validation or proof of real-world
condition classification. That would require measurements with independently
recorded supply, load, and fault provenance.