# Fault 01: implemented phase-A stator-resistance imbalance

## Implemented model behavior

Fault 01 extends the approved fifth-order, synchronous Krause `qd` flux model
with an explicit per-phase stator-resistance configuration. The healthy scenario
uses

\[
(R_{sa},R_{sb},R_{sc})=(1.405,1.405,1.405)\ \Omega.
\]

The controlled fault scenario
`fault_01_phase_a_stator_resistance_plus_10pct` uses

\[
(R_{sa},R_{sb},R_{sc})=(1.5455,1.405,1.405)\ \Omega.
\]

The implementation does **not** alter scalar `R_s` to represent the fault.
It projects \(\operatorname{diag}(R_{sa},R_{sb},R_{sc})\) into the existing
zero-sequence-free Krause frame. For phase-A increment \(\Delta R\), the
resulting resistance term is

\[
\mathbf R_{qd}(\theta)=R_s\mathbf I+
\frac{2}{3}\Delta R
\begin{bmatrix}
\cos^2\theta & \sin\theta\cos\theta\\
\sin\theta\cos\theta & \sin^2\theta
\end{bmatrix},\qquad \theta=\omega_et.
\]

This matrix replaces only the stator resistive voltage drop in
`src/imcm/models/fifth_order_dq.py`. Rotor equations, flux-current map, Park
transform, torque expression, and mechanical equation are unchanged. When the
fault is disabled, or when all phase resistances equal the healthy value, the
implementation takes the original scalar-resistance path exactly.

Fault-aware power balance is explicit: stator copper loss uses
\(\tfrac32\mathbf i_{qd}^{\mathsf T}\mathbf R_{qd}\mathbf i_{qd}\).
The healthy scalar T-equivalent-circuit validation is retained for healthy
operation only and is not asserted for Fault 01.

## Controlled simulation conditions

Healthy and Fault 01 use identical 1.0 s DOL simulations: balanced 400 V
line-line RMS, 50 Hz supply; 15 N m constant load; zero flux and zero mechanical
speed initial state; RK45 with `max_step=1e-4 s`, `rtol=1e-6`, `atol=1e-8`; and
`1e-4 s` output sampling. The only changed input is phase-A resistance.

## Simulated observations

The matched experiment (`experiments/run_fault_01_stator_resistance_imbalance.py`)
uses the late 0.8--1.0 s window. With the illustrative parameters, it produced:

| Metric | Healthy | Fault 01 (+10% phase A) |
| --- | ---: | ---: |
| Phase RMS currents, A (a, b, c) | (5.5652, 5.5666, 5.5643) | (5.5315, 5.6309, 5.5337) |
| Current unbalance | 0.0430% | 1.7847% |
| Negative-sequence magnitude | 0.00278 A | 0.06697 A |
| Negative/positive sequence | 0.0500% | 1.2034% |
| Electromagnetic torque-ripple RMS | 0.0000063 N m | 0.1901 N m |
| Final speed difference, fault minus healthy | -- | +0.2413 r/min |
| Final slip difference, fault minus healthy | -- | -0.0001609 |

The fault therefore produces a clear, measurable simulated current asymmetry,
negative sequence, and `qd`/torque ripple while changing final speed and slip
only slightly. The saved comparison plot overlays phase currents, `qd` currents,
electromagnetic torque, speed, and slip for the matched cases.

## Physical interpretation

This is an interpretable proxy for an elevated-resistance phase-A connection,
joint, or effective winding resistance. A phase-fixed asymmetry appears as
time-varying coupling in a synchronous reference frame, so periodic `qd` and
torque components are expected. It is not an inter-turn-short model and should
not be described as one.

## Assumptions and limitations

- Parameters and load are illustrative literature-example assumptions, not a
  measured motor or test condition.
- The model is linear and fundamental-frequency only: no saturation, slotting,
  thermal dynamics, detailed terminals, winding turns, PWM, or zero-sequence
  current path is represented.
- The scenario assumes isolated-neutral, three-wire operation, so
  \(i_a+i_b+i_c=0\).
- The negative-sequence metric uses an RMS phasor estimate over ten complete
  50 Hz cycles. It is a controlled simulated indicator, not a classifier.
- Supply unbalance, load variation, temperature effects, and measurement noise
  can also cause current unbalance; this model does not establish uniqueness.

## Evidence status

The results above are **simulation evidence** for this exact parameter change.
They are not experimental validation or proof of real-world fault-detection
capability. Such validation would require measurements from a known-resistance
condition with independently recorded supply, load, temperature, and fault
provenance.
