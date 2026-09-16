# Fault 04: bearing outer-race fault (BPFO vibration signature)

## Status and boundary

Fault 04 is a **simulated rolling-element bearing outer-race fault** expressed
as a **vibration-sensor channel**, not a motor-model fault. The electrical
plant — supply, load, motor parameters, stator resistance, Park convention,
and the fifth-order dq ODEs — is **identical** to the approved healthy
baseline, so the stator currents are bit-for-bit healthy. The fault signature
lives entirely on a simulated accelerometer channel at the **BPFO**
characteristic frequency.

This is a **simulated condition-monitoring signature**. It is **not**:

- a measured bearing or a measured machine,
- a stator-current (MCSA) bearing signature,
- a confirmed real-machine diagnosis,
- experimentally validated in any way.

Motor parameters remain `literature_example`; bearing geometry is a published
6205-series literature example. There is no experimental data in this
repository.

## Physical meaning of BPFO

For a rolling-element bearing with the outer race stationary, the cage carries
the balls around the shaft at the cage frequency, and each ball passes over a
point on the outer race faster than the cage by the shaft-relative rotation.
When a **localised defect** (spall) sits on the outer race, every rolling
element strikes it once per pass, producing a periodic impact train at the
**Ball-Pass Frequency Outer race**:

\[
f_\text{BPFO}=\frac{N_b}{2}\,f_r\left(1-\frac{B_d}{P_d}\cos\varphi\right)
\]

| Symbol | Meaning | Value used here |
| --- | --- | --- |
| \(N_b\) | number of rolling elements | 9 |
| \(f_r\) | shaft rotational frequency (Hz) | from the simulated \(\omega_m(t)\) |
| \(B_d\) | ball (rolling-element) diameter | 7.94 mm |
| \(P_d\) | bearing pitch diameter | 39.04 mm |
| \(\varphi\) | contact angle | 0 deg (deep-groove, radial) |

For the default geometry the dimensionless order is
\(N_b(1-(B_d/P_d)\cos\varphi)/2 \approx 3.5868\) impacts per shaft revolution.
At the simulated steady speed of about 1464 r/min (24.4 Hz), BPFO is about
**87.5 Hz**. Because the outer race is stationary, the impulse train is **not
amplitude-modulated by the shaft rate** (unlike an inner-race defect); the
classic outer-race signature is an equal-amplitude impact train at BPFO with
harmonics in the envelope spectrum.

## Bearing parameters and provenance

| Parameter | Value | Provenance |
| --- | ---: | --- |
| `nb_balls` | 9 | Published 6205-series deep-groove ball bearing data (widely used in bearing-fault literature, e.g. the CWRU bearing-data descriptions). **Not measured here.** |
| `ball_diameter_m` | 7.94e-3 m | Same literature source. |
| `pitch_diameter_m` | 39.04e-3 m | Same literature source. |
| `contact_angle_deg` | 0.0 | Deep-groove radial bearing assumption. |
| `resonance_hz` | 2000.0 | Assumed structural resonance in a typical rolling-element bearing band. **Assumed, not identified from a transfer function.** |
| `resonance_decay_s` | 1.0e-3 s | Assumed lightly damped decay. |
| `amplitude_m_s2` | 1.0 | Arbitrary severity scale (m/s²); not calibrated to any sensor. |

Provenance tag: `literature_example_6205_series`. If a real bearing is ever
used, replace these values, retag provenance, and do not keep the literature
label.

## Signal-generation assumptions

Implemented in `imcm/faults/bearing.py`; configuration in
`imcm/models/operating_scenario.py` (`BearingFaultConfig`).

1. **Impact times from the simulated speed trace.** Impacts occur at equal
   increments of **shaft angle** (BPFO is a fixed multiple of shaft speed).
   The cumulative shaft angle is trapezoidal integration of the simulated
   `omega_m(t)` from the unchanged fifth-order plant, so the impact rate
   sweeps with the simulated DOL start-up exactly as the plant speed does.
2. **One damped sinusoid per impact.** Each impact excites an assumed
   structural resonance: `amplitude * exp(-t/tau) * sin(2*pi*f_res*t)`,
   `f_res = 2000 Hz`, `tau = 1 ms`. This is the standard high-frequency
   carrier that envelope analysis demodulates.
3. **No noise, no baseline vibration.** The reduced-order model has no
   slotting, load-torque ripple, or sensor-noise source, so the **healthy
   bearing channel is identically zero by construction**. The faulty channel
   is the deterministic impulse response train alone. This makes the
   simulated separation idealised: real machines always have a nonzero
   vibration floor.
4. **No electrical coupling.** Nothing is injected into `v_abc`, `i_abc`, or
   any ODE state. Real bearing faults *can* produce tiny stator-current
   signatures; this model deliberately does not claim that mechanism.

## Envelope-analysis features

Implemented in `imcm/processing/envelope.py` and
`imcm/validation/bearing_metrics.py`. Chain: Butterworth band-pass around the
resonance (1500–2500 Hz) → Hilbert envelope → mean removal → Hann window →
one-sided amplitude spectrum. Features over the 0.5–1.0 s window:

- envelope amplitude at BPFO and at 2×BPFO,
- interpolated envelope peak frequency near BPFO,
- analytic BPFO at the mean simulated shaft speed.

## What the model can and cannot prove

**Can (within the simulation):**

- demonstrate that an impulse train generated from bearing kinematics at BPFO
  produces envelope-spectrum lines at BPFO and its harmonics;
- show the impact-rate sweep during the simulated start-up;
- exercise a complete, unit-tested envelope-analysis pipeline.

**Cannot prove:**

- that a real bearing fault in a real machine produces these exact levels —
  amplitude is an arbitrary scale, not a calibrated severity;
- any stator-current bearing signature (the electrical model is untouched);
- fault severity, remaining useful life, or a diagnosis of any physical
  machine;
- robustness against noise, load variation, or transmission-path variation —
  the healthy channel here is exactly zero, which no real machine is.

## Simulation protocol

Matched healthy and Fault 04 runs: 1.0 s, RK45, `max_step=1e-4 s`,
`rtol=1e-6`, `atol=1e-8`, `1e-4 s` output sampling — identical to Faults
01–03. Envelope features use the 0.5–1.0 s window where the simulated speed
is nearly constant.

## Evidence status

These results are **simulation evidence** for this exact bearing geometry,
resonance assumption, and severity scale only. They are not experimental
validation, not a digital twin, and not a condition diagnosis of any real
motor. A real diagnosis would require measured vibration (and supply/load)
data with recorded provenance.