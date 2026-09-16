# Fault 05: rotor electrical asymmetry (broken-bar proxy)

## Status and boundary — read first

Fault 05 is a **simulation-only proxy** for broken-bar-related behaviour. It is
**not** a physically complete or bar-resolved electromagnetic broken rotor bar
model. It is **not** severity-calibrated to any number of broken bars, **not**
experimentally validated, and **not** a real-machine diagnosis. The motor
parameters remain `literature_example` values and every trace is `simulated`.

What the proxy *can* illustrate: that a rotor-fixed electrical asymmetry
appears in the synchronous frame as a resistance modulation at twice slip
frequency, producing stator-current components near the classic broken-bar
sideband locations `f_s (1 -/+ 2 s)`.

What it **cannot** claim: bar counts, bar identity, end-ring currents,
severity calibration, transient start-up fidelity of the modulation phase, or
any statement about a physical machine.

## Controlled parameter definition

| Quantity | Healthy baseline | Fault 05 (default) |
| --- | ---: | ---: |
| Rotor-frame axis resistances | `R_r, R_r` | `R_r (1 + 0.10)`, `R_r (1 - 0.10)` |
| Mean rotor resistance | `R_r = 1.395 Ω` | `R_r = 1.395 Ω` (unchanged) |
| Severity `δ` | 0 (disabled) | **0.10** (configurable, `0 <= δ < 1`) |
| Modulation phase rate | — | `2 s_ref ω_e` (constant reference slip) |

The severity is set via `fault_05_rotor_asymmetry_scenario(severity=...)`.
`RotorAsymmetryConfig.reference_slip` can pin the modulation rate explicitly;
by default it is solved from the healthy T-equivalent-circuit torque balance
at the scenario load.

## Unchanged inputs

- **Supply:** 50 Hz balanced sinusoids, 400 V line-line RMS, star, no inverter.
- **Park convention:** classical Krause qd0 (`krause_classical_2_3`), 3/2 torque.
- **Motor parameters:** the same illustrative 4 kW literature-example machine.
- **Stator resistance:** healthy and balanced.
- **Load:** constant 15.0 N m, identical to the healthy baseline.
- **Initial condition:** start from rest, DOL.
- **State vector:** still exactly five states — the locked fifth-order
  architecture is preserved.

Only the rotor-row resistive drops differ from the approved healthy plant.

## Proxy mechanism and equations

The single stator-referred cage resistance is split between two orthogonal
**rotor-frame** axes:

    R_high = R_r (1 + δ),   R_low = R_r (1 - δ),   mean = R_r exactly

Projecting this rotor-fixed split into the locked synchronous frame (the
rotor frame rotates at slip speed relative to it) gives the symmetric matrix

    R_sync(2φ) = R_r I + δ R_r [[cos 2φ,     sin 2φ],
                                [sin 2φ, -cos 2φ]]

with eigenvalues `R_r (1 ± δ)` and trace `2 R_r`. The rotor rows of the
fifth-order plant become

    p λ_qr = -drop_qr - (ω_e - ω_r) λ_dr
    p λ_dr = -drop_dr + (ω_e - ω_r) λ_qr

    drop_qr = R_r i_qr + δ R_r ( cos 2φ · i_qr + sin 2φ · i_dr )
    drop_dr = R_r i_dr + δ R_r ( sin 2φ · i_qr - cos 2φ · i_dr )

with `φ(t) = φ_0 + s_ref ω_e t`, so the modulation angle `2φ` advances at
`2 s_ref ω_e`. Stator voltage rows, flux map, torque, and mechanics are
untouched.

**Why sidebands appear:** a resistance modulation at `2 s f_s` in the
synchronous frame mixes with the fundamental, producing stator-current
components near `f_s (1 - 2 s)` and `f_s (1 + 2 s)` — the classic broken-bar
sideband locations. For the default operating point `s_ref ≈ 0.0238`, so the
sidebands sit near `47.6 Hz` and `52.4 Hz` and the synchronous-frame tone at
`2 s f_s ≈ 2.38 Hz`.

## Documented approximation (constant reference slip)

The physically exact modulation phase is the slip-angle integral
`∫ (ω_e - ω_r) dτ`, which would require a sixth ODE state and break the
locked fifth-order architecture. The proxy therefore advances `φ` at the
**constant** reference slip rate `s_ref ω_e`, where `s_ref` solves the healthy
T-equivalent-circuit torque balance `T_ec(s) = T_L + B ω_m(s)` on the stable
motoring branch. In the late (nearly steady) monitoring window the true slip
approaches `s_ref` and the modulation matches the physical mechanism there;
during the direct-on-line start-up it does not. This is a prescribed, not
emergent, modulation frequency.

## Simulation protocol

Matched healthy and Fault 05 runs use identical solver settings: RK45,
`max_step = 1e-4 s`, `rtol = 1e-6`, `atol = 1e-8`, `1e-4 s` output sampling.
Because the sidebands sit only `2 s f_s ≈ 2.4 Hz` from the 50 Hz carrier, the
experiment uses a **3 s** run and reports the window from 2.0 s to 3.0 s
(0.5 Hz raw bin spacing with Hann leakage suppression). Shorter runs remain
valid for the synchronous-frame `2 s f_s` tone.

## Expected fault signature (simulated)

- Synchronous-frame currents `i_qs, i_ds` gain a tone at `2 s_ref f_s`.
- Phase currents gain components near `f_s (1 -/+ 2 s_ref)`.
- Electromagnetic torque develops a small ripple at `2 s_ref f_s`.
- The mean operating point (speed, slip, mean torque) is essentially
  unchanged, because the mean rotor resistance is preserved by construction.
- The Krause power identity still holds with the anisotropic rotor copper
  loss `(3/2) i_rᵀ R_sync i_r`.

## Simulated observations

From `experiments/run_fault_05_rotor_asymmetry.py` (3 s runs, 2.0–3.0 s
window, severity 0.10, `s_ref = 0.023756`):

| Metric | Healthy | Fault 05 (severity 0.10) |
| --- | ---: | ---: |
| Phase-a fundamental (50 Hz) | 7.871 A | 7.868 A |
| Lower sideband `f_s(1-2s)` ≈ 47.62 Hz | 0.00026 A | **0.2243 A** |
| Upper sideband `f_s(1+2s)` ≈ 52.38 Hz | 0.00026 A | **0.2282 A** |
| Synchronous-frame tone at `2 s f_s` ≈ 2.376 Hz | ~0 (1e-20) | **0.4524 A** |
| Torque ripple (RMS, 2–3 s window) | ~0 (1e-15) | **0.0544 N m** |
| Final speed | 1464.37 r/min | 1461.84 r/min |
| Final slip | 0.023756 | 0.025442 |
| Power-balance residual (rel.) | 3.9e-15 | 4.1e-13 |

The sidebands sit at the analytic proxy locations `f_s (1 -/+ 2 s_ref)` to
within spectral resolution, the healthy case contains no component there
beyond numeric leakage, and the small mean slip rise (+0.0017) is the
expected second-order effect of the anisotropic rotor loss term.

## Limitations

- Not bar-resolved: no bar count, bar identity, or end-ring segments.
- Not severity-calibrated: `δ` is a dimensionless modelling knob, not a
  measured percentage of broken bars.
- Mean rotor resistance kept at `R_r`: a real broken bar also raises the mean
  resistance slightly (higher slip); the proxy deliberately isolates the
  asymmetry effect and does not model that shift as a calibrated effect.
- Constant reference slip: the modulation phase is not the true slip-angle
  integral during start-up.
- Linear magnetics, sinusoidal MMF, uniform air gap: no saturation or
  slotting interactions with the fault.
- No experimental validation in this repository; no real-machine diagnosis is
  claimed or implied.

## Evidence status

These results are **simulation evidence** for this exact proxy configuration
only. They demonstrate the mechanism, not a validated broken-bar diagnostic.
Any future comparison with measurements would require independently recorded
supply, load, and machine provenance.