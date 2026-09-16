# Fault 03: supply-voltage unbalance (supply-side confounder)

## Status and boundary

This is a **controlled supply-side confounder**, not a confirmed internal motor
fault and not winding damage. It does **not** alter the motor model equations,
motor parameters, stator resistance, load torque, Park convention, or supply
frequency. Only the applied three-phase supply voltages change. All traces are
**simulated**; parameters remain `literature_example` values. There is no
experimental validation in this repository.

A supply-voltage unbalance is a grid/feeder condition. It is deliberately kept
separate from Fault 01 (a simulated winding-resistance imbalance) so that a
supply confounder is never mislabelled as stator damage.

## Controlled parameter definition

| Quantity | Healthy baseline | Fault 03 |
| --- | ---: | ---: |
| Phase-A peak voltage | 1.0 p.u. | 1.0 p.u. |
| Phase-B peak voltage | 1.0 p.u. | 1.0 p.u. |
| Phase-C peak voltage | 1.0 p.u. | **0.9 p.u.** |
| Phase spacing | 120 deg | 120 deg (unchanged) |

The per-phase multipliers are `(1.0, 1.0, 0.9)` of the balanced phase peak
`V_peak = 400 * sqrt(2) / sqrt(3) = 326.599 V`. A 5 percent case is available by
passing `(1.0, 1.0, 0.95)` to
`fault_03_supply_voltage_unbalance_scenario`.

## Unchanged inputs

- **Supply frequency and waveform:** 50 Hz balanced-sinusoid model, 400 V
  line-line RMS, star, no inverter.
- **Park convention:** classical Krause qd0, factor 2/3, 3/2 torque, named
  `krause_classical_2_3`.
- **Motor parameters:** the same illustrative 4 kW, 400 V, 50 Hz, 4-pole
  literature-example machine.
- **Stator resistance:** healthy and balanced, `enabled=False`,
  `multipliers_abc=(1.0, 1.0, 1.0)`.
- **Load:** constant 15.0 N m, identical to the healthy baseline.
- **Initial condition:** start from rest (zero flux, zero mechanical speed), DOL.

Only `supply.voltage_unbalance` differs from the approved healthy scenario.

## Supply-voltage model

The applied phase voltages keep the locked 120-degree spacing but scale each
phase peak independently:

    v_a(t) = m_a * V_peak * cos(omega_e t)
    v_b(t) = m_b * V_peak * cos(omega_e t - 2*pi/3)
    v_c(t) = m_c * V_peak * cos(omega_e t + 2*pi/3)

with `(m_a, m_b, m_c) = (1.0, 1.0, 0.9)`. The balanced control path
(`m = (1, 1, 1)`, or the disabled default) returns the exact approved scalars
`v_qs = V_peak`, `v_ds = 0`, `v_0 = 0`, so the healthy, Fault 01, and Fault 02
results are bit-for-bit unchanged.

The unbalanced set is transformed to the synchronous frame with the same Park
convention. Because the phase peaks differ, the applied voltage contains a
**positive-sequence** and a **negative-sequence** component. The machine is a
three-wire system, so the zero-sequence voltage does not drive current; the
plant responds only to the positive- and negative-sequence components. The
negative-sequence voltage drives a negative-sequence current, which is the
physical origin of the extra loss, torque ripple, and current unbalance.

## Simulation protocol

Matched healthy and Fault 03 runs use identical settings: 1.0 s duration, RK45
with `max_step=1e-4 s`, `rtol=1e-6`, `atol=1e-8`, `1e-4 s` output sampling.
Reporting uses the late window from 0.8 s to 1.0 s.

## Expected effects of supply-voltage unbalance

- **Voltage unbalance factor:** nonzero. For `(1, 1, 0.9)` the standard
  `|V2| / |V1| * 100` is about 3.448 percent.
- **Negative-sequence current:** nonzero, driven by the negative-sequence
  voltage through the negative-sequence impedance.
- **Current unbalance:** phase currents become unequal; the phase with the
  reduced voltage carries less current while the other phases carry more.
- **Torque ripple:** a second-harmonic (2*omega_e) torque ripple appears because
  the positive- and negative-sequence fields interact.
- **Speed/slip:** a small change, since the mean air-gap torque still balances
  the load.
- **Power balance:** the Krause power identity
  P_in = copper + T_e*omega_m + dW_mag/dt still holds because no fault term is
  introduced.

## Simulated observations

From `experiments/run_fault_03_supply_voltage_unbalance.py`:

| Metric | Healthy | Fault 03 (phase C 0.9 p.u.) |
| --- | ---: | ---: |
| Supply voltage unbalance | ~0 | 3.448 % |
| Phase RMS currents (a, b, c) | 5.565, 5.565, 5.565 A | 7.007, 6.412, 3.733 A |
| Current unbalance | ~0 | 57.27 % |
| Negative-sequence current | ~0 | 1.910 A (34.27 %) |
| Torque ripple (RMS) | ~0 | 5.417 N m |
| Final speed | 1464.37 r/min | 1461.62 r/min |
| Final slip | 0.02376 | 0.02559 |
| Power-balance residual | 9.4e-15 | 8.9e-07 |

The phase-C current drops while phases A and B rise, the negative-sequence
current becomes clearly nonzero, and a second-harmonic torque ripple appears.
The power-balance residual is a `np.gradient` discretization artifact that
converges as `dt^2` (2.2e-7 at `output_dt=5e-5`), not a model error.

## Physical interpretation

A supply-voltage unbalance is a common real-world confounder. It produces a
negative-sequence current, extra stator loss, current unbalance, and torque
ripple that can resemble the signature of an internal winding fault. Because
the motor parameters, stator resistance, and load are unchanged here, any
current unbalance or torque ripple is attributable to the supply, not to a
winding or rotor defect. This is exactly why the supply condition must be
recorded independently before any fault claim is made.

## Assumptions and limitations

- Parameters and voltage values are illustrative literature-example assumptions,
  not a measured motor or test-bench condition.
- The model is linear and fundamental-frequency only: no saturation, slotting,
  thermal drift, PWM, or noise.
- The unbalance is a static per-phase magnitude scaling; phase-angle unbalance
  and time-varying sags are deferred.
- Constant load torque; fan-type load proportional to speed squared is deferred.
- A supply change, a load change, and an internal fault can each raise current
  and unbalance; this single controlled simulation does **not** establish fault
  uniqueness or enable a classifier.

## Evidence status

These results are **simulation evidence** for this exact supply-voltage
unbalance only. They are not experimental validation or proof of real-world
condition classification. That would require measurements with independently
recorded supply, load, and fault provenance.
