# Fault 01 design: stator-resistance imbalance in phase A

## Status and boundary

This is a **model design** for the first controlled fault scenario. It does not
implement or simulate the fault, change the approved healthy plant, or claim a
fault-detection result. Future traces must be labelled `simulated`; the motor
parameters remain `literature_example` values.

The intended mechanism is a high-resistance phase-A stator winding or terminal
connection within the fundamental-frequency, linear model. It is **not** a model
of an inter-turn short, insulation failure, temperature field, or specific motor.

## Controlled parameter definition

The healthy model uses the balanced per-phase resistance \(R_s=1.405\ \Omega\):

\[
\mathbf R_{s,abc}^{H}=\operatorname{diag}(1.405,\ 1.405,\ 1.405)\ \Omega.
\]

The proposed fault changes only phase A by +10%:

\[
R_{sa}=1.10R_s=1.5455\ \Omega,\qquad R_{sb}=R_{sc}=1.405\ \Omega,
\]

\[
\mathbf R_{s,abc}^{F}=\operatorname{diag}(1.5455,\ 1.405,\ 1.405)\ \Omega.
\]

The 0.1405 ohm increment is a controlled sensitivity level, not a measured
fault severity. Raising scalar `params.r_s` by 10% would be wrong: it raises all
three phases equally and cannot represent a phase-A imbalance.

## Mathematical insertion point

The healthy stator equations at `src/imcm/models/fifth_order_dq.py:99-100` use
the scalar drops \(R_si_{qs}\) and \(R_si_{ds}\). A future fault wrapper should
replace only \(R_s[i_{qs},i_{ds}]^\mathsf T\) with the phase-resistance matrix
projected into the existing synchronous `qd` subspace. With the current Krause
transform, \(\theta=\omega_et\), \(i_0=0\), and
\(\Delta R=R_{sa}-R_s=0.1405\ \Omega\),

\[
\mathbf R_{qd}(\theta)=R_s\mathbf I+
\frac{2}{3}\Delta R
\begin{bmatrix}
\cos^2\theta & \sin\theta\cos\theta\\
\sin\theta\cos\theta & \sin^2\theta
\end{bmatrix}.
\]

The future stator equations are therefore

\[
\begin{bmatrix}\dot\lambda_{qs}\\\dot\lambda_{ds}\end{bmatrix}
=
\begin{bmatrix}v_{qs}\\v_{ds}\end{bmatrix}
-\mathbf R_{qd}(\theta)
\begin{bmatrix}i_{qs}\\i_{ds}\end{bmatrix}
+\begin{bmatrix}-\omega_e\lambda_{ds}\\+\omega_e\lambda_{qs}\end{bmatrix}.
\]

The matrix has DC and \(2\omega_e\) terms in the synchronous frame, as expected
for a phase-fixed asymmetry. Rotor equations, flux-current map, torque equation,
and mechanical equation are unchanged. The formulation retains the existing
isolated-neutral three-wire condition \(i_a+i_b+i_c=0\); no zero-sequence path
may be added without an explicit neutral/grounding model.

## Physical interpretation

Increased series resistance plausibly represents an elevated-resistance terminal,
joint, or effective winding resistance. It increases copper loss and unbalances
phase voltage drops; under balanced three-wire supply it can yield unbalanced
currents and negative sequence. It does not represent a shorted turn, which
requires coupled circuits and may alter inductance/local flux beyond this model.

## Controlled comparison protocol

Healthy and fault cases must differ only in \(\mathbf R_{s,abc}\), with identical:

- balanced 50 Hz, 400 V line-line RMS ideal supply and phase convention;
- constant \(T_L=15\) N m load torque;
- zero fluxes and \(\omega_m(0)=0\);
- RK45: `max_step=1e-4 s`, `rtol=1e-6`, `atol=1e-8`, output interval `1e-4 s`;
- 1.0 s duration; and
- reporting windows, including \(t\in[0.8,1.0]\) s (ten 50 Hz cycles).

Record case ID, all phase resistances, solver configuration, initial-condition
label, and provenance. Use the same parameter object apart from an explicit
fault-case resistance descriptor.

## Signals and proposed simulated indicators

Compare full startup traces and matched late-window summaries for phase currents,
`qd` currents, electromagnetic torque \(T_e\) (not net torque), rotor speed,
and slip. Define the following over the same integer-cycle late window:

| Indicator | Definition | Model-level use |
| --- | --- | --- |
| RMS current imbalance | \((\max(I_{a,rms},I_{b,rms},I_{c,rms})-\min(\cdot))/\operatorname{mean}(I_{a,rms},I_{b,rms},I_{c,rms})\) | Compare fault with healthy numerical floor. |
| Negative-sequence magnitude | From 50 Hz phasors and \(a=e^{j2\pi/3}\), \(I_2=(I_a+a^2I_b+aI_c)/3\); optionally report \(|I_2|/|I_1|\), \(I_1=(I_a+aI_b+a^2I_c)/3\). | Quantifies fundamental asymmetry. |
| Torque-ripple RMS | \(\sqrt{\operatorname{mean}[(T_e-\operatorname{mean}T_e)^2]}\). | Compare electromechanical pulsation. |
| Speed deviation | \(\Delta\bar n=\bar n_F-\bar n_H\), plus RMS of \(n_F-n_H\) on the common grid. | Separates mean shift from ripple. |

Phasor extraction requires a documented convention and an integer number of
cycles. In the fault case `qd` currents should acquire periodic components rather
than remain ideally DC; that is a model response, not a classification result.

## Meaningful simulation evidence

A meaningful result would show, after implementation and convergence checks:

1. zero fault increment reproduces healthy states and signals within a preset
   numerical tolerance;
2. +10% phase A produces repeatable nonzero RMS imbalance and negative sequence
   above the healthy numerical floor; and
3. torque, speed, and slip differences use the same late window and remain
   consistent with fault-aware power balance.

That is evidence only for this controlled linear simulation. It would not prove
real-world fault detection, representative fault severity, resistance
identifiability amid supply/load/temperature variation, generalization across
motors, or discrimination from supply-voltage unbalance. Experimental validation
needs measurements with independent fault and operating-condition provenance.

## Required tests before implementation

1. Validate phase resistances: positive entries; A = 1.5455 ohm; B/C = 1.405
   ohm; zero increment yields three equal healthy values.
2. Test the `qd` resistance projection against direct abc-to-`qd` transformation
   of \(\mathbf R_{s,abc}\mathbf i_{abc}\) for random \(\theta\) and
   zero-sequence-free currents. Check symmetry and positive dissipation.
3. Regression-test zero increment against healthy states, currents, torque,
   speed, and slip within solver tolerance; preserve all healthy tests.
4. Test fault phase-current reconstruction and \(i_a+i_b+i_c=0\).
5. Test that late healthy negative sequence/current imbalance stay at the
   numerical floor and the +10% case is nonzero. Set thresholds after a
   convergence study, not visual inspection.
6. Make power validation fault aware: use
   \(\tfrac32\mathbf i_{qd}^\mathsf T\mathbf R_{qd}\mathbf i_{qd}\), not the
   scalar resistance loss at `src/imcm/validation/power_balance.py:40-43`.
7. Do not use the current balanced scalar T-equivalent-circuit check as a
   fault-case validator; extend it for unbalance or mark it healthy-only.
8. Repeat matched cases with tighter RK45 tolerances and smaller `max_step`, and
   assert convergence of all four indicators.

## Evidence hierarchy

- **Model design (this document):** defines an interpretable parameter change
  and a mathematically consistent insertion point.
- **Simulation evidence (future):** requires implementation, regression tests,
  matched comparisons, and convergence checks.
- **Experimental validation (not present):** requires real-machine measurements
  with independently documented fault and operating conditions.
