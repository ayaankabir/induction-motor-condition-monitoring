# Healthy baseline technical review

## Purpose and scope

This is a simulation-only review of the healthy, fifth-order squirrel-cage
induction-motor model.  The plant has four synchronous-frame electrical flux
states and one mechanical-speed state.  It uses the classical amplitude-invariant
Krause `qd0` transform (factor `2/3`, with matching `3/2` torque and power
factors), a balanced sinusoidal supply, and a squirrel-cage rotor
(`v_qr = v_dr = 0`).  It is a reduced-order healthy-machine model, not FEA, a
bar-resolved cage model, or a digital twin.

No fault model, machine-learning diagnosis, dashboard, inverter model, or
hardware integration is included in this baseline.

## Assumed machine and operating data

The parameter set is `illustrative_4kW_400V_50Hz_4pole`, with provenance
`literature_example`; it is not identified from a particular machine.

| Quantity | Assumed value | Unit |
| --- | ---: | --- |
| Stator resistance, \(R_s\) | 1.405 | ohm |
| Rotor resistance (stator-referred), \(R_r\) | 1.395 | ohm |
| Stator leakage inductance, \(L_{ls}\) | 0.005839 | H |
| Rotor leakage inductance, \(L_{lr}\) | 0.005839 | H |
| Magnetizing inductance, \(L_m\) | 0.1722 | H |
| Pole count, \(P\) | 4 | -- |
| Rotor/load inertia, \(J\) | 0.0131 | kg m\(^2\) |
| Viscous friction, \(B\) | 0.002985 | N m s |
| Rated frequency (metadata) | 50 | Hz |
| Rated line-line voltage (metadata) | 400 | V RMS |
| Rated mechanical power (metadata) | 4000 | W |
| Rated speed (metadata) | 1430 | r/min |

The supply is an ideal balanced 50 Hz, 400 V line-line RMS, star-connected
three-phase source (phase-neutral peak \(\approx326.60\) V).  The mechanical
scenario applies an assumed constant signed load torque \(T_L=15\) N m.
The initial state is rest with all four flux linkages and mechanical speed zero;
full voltage is applied at \(t=0\) (DOL start).

## Numerical method

Integration uses `scipy.integrate.solve_ivp` with adaptive RK45,
`max_step = 1e-4 s`, `rtol = 1e-6`, and `atol = 1e-8`.  The normal simulation
duration is 1 s, with uniformly reported output at `1e-4 s` spacing.

A numerical sensitivity check repeated the simulation with `max_step = 1e-5 s`,
`rtol = 1e-9`, and `atol = 1e-11`.  The reported extrema were unchanged to the
displayed precision, so the startup features below are not an RK45 time-step or
output-sampling artifact.

## Interpretation of the startup response

- **DOL inrush current.** Applying the ideal supply to initially unfluxed
  inductances produces a magnetizing and leakage-current transient.  The chosen
  supply closing phase is implicit: phase-a voltage is at its positive peak at
  \(t=0\).  This is a switching transient, not a steady-state current prediction.
- **Initial reverse-speed transient.** Initially \(T_e=0\), but the model applies
  the positive signed 15 N m load immediately.  Thus
  \(\dot\omega_m(0)=(T_e-T_L-B\omega_m)/J=-1145.04\) rad/s\(^2\), and speed
  briefly reaches about -35.9 r/min.  This follows directly from the stated load
  law; it is not a transform or torque-sign error.
- **Temporary negative slip.** The torque impulse and low modeled damping produce
  a brief inertial overspeed of about 1514.2 r/min near 58.1 ms, giving minimum
  slip about -0.00949.  The electrical state is still transient at that instant;
  it is not a steady generating operating point.
- **Electromagnetic-torque overshoot.** The calculated \(T_e\) peaks near
  145.7 N m around 12.2 ms and subsequently damps.  A DOL, zero-flux switching
  transient and the linear, unsaturated inductance model explain its size and
  sensitivity.  In the reviewed code \(T_e\) itself does not become negative;
  a negative plotted quantity would instead be net accelerating torque or require
  checking the plotted data/label.
- **Final operating point.** Over \(t\geq0.8\) s, the mean speed is about
  1464.37 r/min, mean slip about 0.02376, and mean \(T_e\) about 15.458 N m.
  This is sensible motoring operation below the 1500 r/min synchronous speed.

## Torque definitions

- \(T_e\): electromagnetic air-gap torque from the electrical flux/current
  states; positive values accelerate the positive mechanical direction.
- \(T_L\): the imposed 15 N m signed load torque.  In the current equation it is
  assumed to oppose positive rotation even at zero or negative speed.
- Net accelerating torque: \(T_{net}=T_e-T_L-B\omega_m\).  It—not \(T_e\)—may
  be negative while the machine slows.  At steady state \(T_{net}\approx0\), so
  \(T_e\approx T_L+B\omega_m\), including the approximately 0.458 N m viscous
  torque here.

## Validation status and limitations

All **22 tests pass**.  They check the initial state, finite integration,
Krause transform/reconstruction identities, phase-current balance, motoring
slip and torque balance, synchronous-frame settling, power balance,
steady-state equivalent-circuit agreement, and late current frequency.

The checks establish internal numerical and algebraic consistency only.  The
results are illustrative because parameters and load are assumed literature
values, not measurements, and the supply closing angle and zero initial flux
are idealized.  They are not experimental validation.

Important limitations are linear magnetic inductances (no saturation), sinusoidal
MMF and uniform air gap, a single equivalent rotor cage, no slotting/eccentricity,
no inverter/PWM, no remanent flux, no shaft compliance, and no passive-load
stiction or direction-opposing load law.  In particular, a passive physical load
would normally require an explicit zero-speed/direction convention before the
brief reverse motion should be interpreted as physical behavior.
