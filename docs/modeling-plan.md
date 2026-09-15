# Modeling plan

This document is the technical source of truth for the **first implementation**.
It is **not** a claim that a time-domain motor simulation has been run, validated,
or compared with laboratory measurements.

**Current implementation status:** the **healthy** fifth-order time-domain plant
is implemented (adaptive RK45, start from rest). Fault models, inverter supply,
and experimental validation are **not** implemented. All traces are **simulated**.

---

## Locked engineering decisions (do not silently change)

| Topic | Decision |
| --- | --- |
| Modeling convention | **Classical Krause-style** \(qd0\) transform (amplitude-invariant, factor \(2/3\)). Not power-invariant. One convention everywhere. |
| Healthy plant | Fifth-order squirrel-cage two-axis flux model (4 electrical + 1 mechanical). |
| First milestone | **Healthy motor only.** No fault simulation. |
| Supply | Balanced three-phase **sinusoidal** voltages, **50 Hz**, star-connected line-neutral reconstruction from 400 V line-line RMS. **No inverter.** |
| Load | **Constant load torque** \(T_L\) (assumed numerical value; see §5). Fan-type \(\propto\omega_m^2\) is later optional. |
| Broken rotor bar | **Out of scope for the first milestone.** Any later two-axis proxy is a **proxy**, not a bar-resolved cage. |
| Data | **Simulation-only.** No fabricated experiments. Public datasets or a lab plan only if added later with provenance. |
| Parameters | Published **illustrative / literature-example** 4 kW, 400 V, 50 Hz, 4-pole set. **Not measured.** |
| Integrator | Adaptive **RK45** (`solve_ivp`), `max_step = 10^{-4}\,\mathrm{s}` (1/200 of a 50 Hz period). |
| Initial condition | **Start from rest:** \(\lambda_{qs}=\lambda_{ds}=\lambda_{qr}=\lambda_{dr}=0\), \(\omega_m=0\). Full voltage at \(t=0\) (DOL start). |

---

## 1. Project objective

Develop a **reduced-order**, electrically observable induction-machine model and,
after the healthy plant is verified, a signal-processing pipeline that can:

- reproduce **healthy** stator currents under the locked supply and load;
- later introduce **selected, physically motivated faults** as controlled deviations;
- extract **electrical features** whose expected locations are known from the literature;
- state **limitations** so results are not sold as a digital twin or as plant diagnosis.

The scientific target is a **defensible student-scale prototype**: model → signals →
features → interpretation. It is not FEA, not a winding-function digital twin, and
not a certified condition-monitoring product.

---

## 2. Why this reduced-order model

| Fidelity | Typical state count | Useful for electrical monitoring? | Role here |
| --- | --- | --- | --- |
| 2-D/3-D FEA | thousands | high if time-stepped | out of scope |
| Coupled circuits (per bar) | tens–hundreds | good for broken bars | later optional only |
| **Fifth-order two-axis \(dq\)** | **5** | **healthy currents, torque, speed** | **first plant** |
| Third-order (no stator transients) | 3 | poor stator current transients | comparison later, not the core |
| First-order mechanical | 1 | no electrical waveforms | unusable |

**Decision:** healthy plant = **fifth-order two-axis squirrel-cage model** (Krause / Ong),
flux-linkage states, **synchronous** reference frame \(\omega=\omega_e=2\pi f_s\).

This is “reduced-order” relative to FEA and per-bar circuits: sinusoidal MMF, uniform
air gap, single equivalent rotor cage, linear inductances. That is acceptable for
**balanced healthy operation**. It does **not** contain slot harmonics, saturation
harmonics, or broken-bar sidebands until a **separate, documented** extension is added.

A **flux** formulation is preferred over a pure current formulation because, with
constant inductances, the flux ODEs are usually better conditioned.

---

## 3. Park transform convention (locked)

### 3.1 Why classical Krause, not power-invariant

Two common conventions exist:

1. **Classical Krause / many US machinery textbooks (chosen):** Clarke/Park factor
   \(2/3\). Peak of a balanced sinusoid maps to a \(q\)-axis DC equal to that peak.
   Instantaneous three-phase power is
   \(P=\tfrac{3}{2}(v_{qs}i_{qs}+v_{ds}i_{ds}+2v_{0}i_{0})\).
   Electromagnetic torque carries the matching **\(\tfrac{3}{2}\)** factor.
2. **Power-invariant (orthonormal):** factor \(\sqrt{2/3}\). Then
   \(P=v_q i_q+v_d i_d+v_0 i_0\) and torque has **no** \(3/2\).

**We keep Krause** because:

- it matches the textbooks this project will cite (Krause; Ong);
- \(v_{qs}=V_s\), \(v_{ds}=0\) is a simple healthy-supply sanity check;
- mixing \(\sqrt{2/3}\) transforms with a \(3/2\) torque formula (or the reverse)
  silently breaks power balance.

Power-invariant scaling is not “more physical”; it is a different bookkeeping
choice. Changing it later would require rewriting **every** transform, torque,
and power test together.

**Do not mix conventions** in code, comments, or plots.

### 3.2 Krause \(abc\to qd0\) (stator)

Let \(\theta\) be the angle of the **\(q\)-axis** relative to the **phase-\(a\)** magnetic axis
(electrical radians). Following Krause:

\[
\begin{aligned}
f_{qs} &= \frac{2}{3}\Bigl[
  f_a\cos\theta + f_b\cos\bigl(\theta-\tfrac{2\pi}{3}\bigr) + f_c\cos\bigl(\theta+\tfrac{2\pi}{3}\bigr)
\Bigr] \\
f_{ds} &= \frac{2}{3}\Bigl[
  f_a\sin\theta + f_b\sin\bigl(\theta-\tfrac{2\pi}{3}\bigr) + f_c\sin\bigl(\theta+\tfrac{2\pi}{3}\bigr)
\Bigr] \\
f_{0s} &= \frac{1}{3}(f_a+f_b+f_c)
\end{aligned}
\]

Inverse:

\[
\begin{aligned}
f_a &= f_{qs}\cos\theta + f_{ds}\sin\theta + f_{0s} \\
f_b &= f_{qs}\cos\bigl(\theta-\tfrac{2\pi}{3}\bigr) + f_{ds}\sin\bigl(\theta-\tfrac{2\pi}{3}\bigr) + f_{0s} \\
f_c &= f_{qs}\cos\bigl(\theta+\tfrac{2\pi}{3}\bigr) + f_{ds}\sin\bigl(\theta+\tfrac{2\pi}{3}\bigr) + f_{0s}
\end{aligned}
\]

**Signs:** the \(d\)-row uses **\(+\sin\)**, as in Krause’s \(\mathbf{K}_s\). Do not substitute
an IEEE “\(d\)-axis on phase \(a\)” matrix without changing \(\theta\) and the voltage
speed-voltage terms together.

### 3.3 Synchronous frame used for the healthy plant

\[
\theta(t)=\omega_e t,\qquad \omega_e=2\pi f_s,\qquad f_s=50\,\mathrm{Hz}
\]

(The same \(\theta\) is used for voltages and currents.)

### 3.4 Balanced-supply identity (used as a test)

With peak phase voltage \(V_s\) and

\[
v_a=V_s\cos(\omega_e t),\quad
v_b=V_s\cos(\omega_e t-\tfrac{2\pi}{3}),\quad
v_c=V_s\cos(\omega_e t+\tfrac{2\pi}{3})
\]

the Krause transform with \(\theta=\omega_e t\) gives

\[
v_{qs}=V_s,\qquad v_{ds}=0,\qquad v_{0s}=0.
\]

---

## 4. Healthy-machine equations

Rotor quantities are **stator-referred**. Linear magnetics. Squirrel cage:
\(v_{qr}=v_{dr}=0\).

### 4.1 Notation

| Symbol | Meaning |
| --- | --- |
| \(p=d/dt\) | time derivative |
| \(P\) | number of **poles** (not pole-pairs) |
| \(\omega_m\) | mechanical rotor speed (rad/s) |
| \(\omega_r=(P/2)\omega_m\) | rotor **electrical** speed |
| \(\omega=\omega_e\) | reference-frame electrical speed (synchronous frame) |
| \(s=(\omega_e-\omega_r)/\omega_e\) | slip |

**Motor sign convention:** \(T_e>0\) accelerates the rotor in the positive \(\omega_m\)
direction. \(T_L>0\) is a **load** opposing that rotation.

### 4.2 Voltage equations (arbitrary frame; we set \(\omega=\omega_e\))

\[
\begin{aligned}
v_{qs} &= R_s i_{qs} + \omega\lambda_{ds} + p\lambda_{qs} \\
v_{ds} &= R_s i_{ds} - \omega\lambda_{qs} + p\lambda_{ds} \\
v_{qr} &= R_r i_{qr} + (\omega-\omega_r)\lambda_{dr} + p\lambda_{qr}=0 \\
v_{dr} &= R_r i_{dr} - (\omega-\omega_r)\lambda_{qr} + p\lambda_{dr}=0
\end{aligned}
\]

Rearranged flux ODEs (implementation form, after currents are obtained from fluxes):

\[
\begin{aligned}
p\lambda_{qs} &= v_{qs} - R_s i_{qs} - \omega\lambda_{ds} \\
p\lambda_{ds} &= v_{ds} - R_s i_{ds} + \omega\lambda_{qs} \\
p\lambda_{qr} &= - R_r i_{qr} - (\omega-\omega_r)\lambda_{dr} \\
p\lambda_{dr} &= - R_r i_{dr} + (\omega-\omega_r)\lambda_{qr}
\end{aligned}
\]

### 4.3 Flux linkages (linear)

\[
\begin{aligned}
\lambda_{qs} &= L_s i_{qs} + L_m i_{qr} \\
\lambda_{ds} &= L_s i_{ds} + L_m i_{dr} \\
\lambda_{qr} &= L_r i_{qr} + L_m i_{qs} \\
\lambda_{dr} &= L_r i_{dr} + L_m i_{ds}
\end{aligned}
\]

\[
L_s=L_{ls}+L_m,\qquad L_r=L_{lr}+L_m
\]

Let \(\Delta=L_s L_r-L_m^2\). Current inversion (used at every evaluation of the ODE):

\[
\begin{aligned}
i_{qs} &= (L_r\lambda_{qs}-L_m\lambda_{qr})/\Delta \\
i_{ds} &= (L_r\lambda_{ds}-L_m\lambda_{dr})/\Delta \\
i_{qr} &= (L_s\lambda_{qr}-L_m\lambda_{qs})/\Delta \\
i_{dr} &= (L_s\lambda_{dr}-L_m\lambda_{ds})/\Delta
\end{aligned}
\]

### 4.4 Torque and mechanics (Krause \(3/2\) factor)

\[
T_e=\frac{3}{2}\frac{P}{2}\bigl(\lambda_{ds} i_{qs}-\lambda_{qs} i_{ds}\bigr)
\]

Equivalent forms \(\tfrac{3}{2}\tfrac{P}{2} L_m(i_{qs}i_{dr}-i_{ds}i_{qr})\) must match
this \(\lambda i\) form under linear magnetics (algebraic test).

\[
J\frac{d\omega_m}{dt}=T_e-T_L-B\omega_m
\]

**First milestone:** \(T_L(t)=T_L\) **constant**.

### 4.5 State vector

\[
\mathbf{x}=\bigl[\lambda_{qs},\;\lambda_{ds},\;\lambda_{qr},\;\lambda_{dr},\;\omega_m\bigr]^\top
\]

Zero-sequence stator flux is omitted because a **balanced** star supply has \(v_{0s}=0\)
and we assume no zero-sequence path that we need to integrate for the healthy case.

### 4.6 Instantaneous power check (Krause)

Electrical input (no zero sequence):

\[
P_\text{in}=\frac{3}{2}(v_{qs}i_{qs}+v_{ds}i_{ds})
\]

In the limit of a correct model,
\(P_\text{in}\) equals copper loss + mechanical power \(T_e\omega_m\) + rate of change
of stored magnetic energy. After transients decay, stored-energy rate \(\approx 0\),
so this becomes a **steady-state power-balance test**.

---

## 5. Motor parameters (illustrative, not measured)

### 5.1 Quantities required by the ODE

| Symbol | Code name | Role |
| --- | --- | --- |
| \(R_s\) | `r_s` | stator copper drop and loss |
| \(R_r\) | `r_r` | rotor copper drop; sets slip for a given torque |
| \(L_{ls},L_{lr}\) | `l_ls`, `l_lr` | leakage; affect starting current and transient peaks |
| \(L_m\) | `l_m` | magnetizing current and flux |
| \(P\) | `n_poles` | electrical/mechanical speed ratio; torque constant |
| \(J\) | `inertia` | mechanical time constant |
| \(B\) | `viscous_friction` | steady-state \(T_e=T_L+B\omega_m\) |
| \(f_s,V_{ll}\) | rated supply metadata | \(\omega_e\) and \(V_s\) |

Derived: \(L_s,L_r\), pole-pairs \(P/2\), \(n_s=120 f_s/P\) (synchronous r/min).

### 5.2 Metadata (not ODE states, needed for honest reporting)

| Quantity | Role |
| --- | --- |
| Assumed rated mechanical power | per-unit torque and “is this overload?” |
| Assumed rated speed | estimate rated torque \(T_\text{rated}\approx P_\text{mech}/\omega_{m,\text{rated}}\) |
| Star connection | voltages are phase-to-neutral |
| Provenance tag | must be `literature_example` until a user supplies measurements |

### 5.3 Chosen example machine

**Name:** `illustrative_4kW_400V_50Hz_4pole`

**Provenance:** `literature_example`

These numbers match a **widely published 4 kW, 400 V, 50 Hz, 1430 r/min, 4-pole
squirrel-cage SI example** used in textbook/Simulink-style documentation
(MathWorks Asynchronous Machine SI default and copies thereof).

**They are not measurements from a motor in this project.** They are **assumed
starting values** so the equations have a consistent numerical instance.

| Parameter | Assumed value | SI unit |
| --- | --- | --- |
| \(R_s\) | 1.405 | Ω |
| \(R_r\) (stator-referred) | 1.395 | Ω |
| \(L_{ls}\) | 0.005839 | H |
| \(L_{lr}\) | 0.005839 | H |
| \(L_m\) | 0.1722 | H |
| Poles \(P\) | 4 | — |
| \(J\) | 0.0131 | kg·m² |
| \(B\) | 0.002985 | N·m·s |
| Rated frequency | 50 | Hz |
| Rated line-line RMS | 400 | V |
| Assumed rated mechanical power | 4000 | W |
| Assumed rated speed | 1430 | r/min |
| Stator connection | star | — |

Assumed rated torque (metadata only):

\[
T_{\text{rated,assumed}}=\frac{4000}{1430\cdot 2\pi/60}\approx 26.71\,\mathrm{N\cdot m}
\]

If a laboratory motor is obtained later, replace this set, retag provenance as
`user_supplied` or `identified`, and **do not** keep the literature label.

---

## 6. Supply (first milestone)

- Waveform: ideal balanced sinusoids, **no PWM**.
- Frequency: **50 Hz**.
- Magnitude: \(V_{ll,\mathrm{rms}}=400\,\mathrm{V}\).
- Phase-to-neutral peak:

\[
V_s=\sqrt{2}\,\frac{V_{ll,\mathrm{rms}}}{\sqrt{3}}\approx 326.599\,\mathrm{V}
\]

Inverter-fed operation is **out of scope** until a later, explicit milestone.

---

## 7. Load (first milestone)

\[
T_L=15.0\,\mathrm{N\cdot m}\quad\text{(constant; assumed scenario, not measured)}
\]

This is about \(0.56\,T_{\text{rated,assumed}}\): loaded enough for non-zero slip,
not an overload claim.

Fan-type \(T_L=k\omega_m^2\) may be compared **later**. It is not the first plant.

---

## 8. How healthy electrical signals are generated

Implemented in `imcm.models.fifth_order_dq.simulate_healthy`.

Procedure:

1. Load the literature-example parameters and the locked supply/load scenario.
2. Set the initial state to rest (unfluxed):
   \(\mathbf{x}(0)=\mathbf{0}\).
3. Apply balanced 50 Hz voltages immediately (direct-on-line). In the synchronous
   frame this is the constant pair \(v_{qs}=V_s\), \(v_{ds}=0\).
4. Integrate the five ODEs with **adaptive RK45**:
   - `scipy.integrate.solve_ivp(..., method="RK45")`
   - `rtol=10^{-6}`, `atol=10^{-8}`
   - **`max_step=1\times10^{-4}\,\mathrm{s}`** so the stepper cannot skip more
     than 1/200 of a 50 Hz cycle even when the error estimate would allow a
     larger step. Leakage time constants are on the order of \(L_{ls}/R_s\sim 4\,\mathrm{ms}\);
     this cap is smaller than that scale.
5. Invert fluxes \(\to\) currents, inverse Park \(\to i_a,i_b,i_c\).
6. Store traces with metadata: `label=healthy`, `provenance=simulated`,
   `parameter_provenance=literature_example`, \(T_L\), \(f_s\),
   Park convention `krause_classical_2_3`, solver name and `max_step`.

**No laboratory noise.** Traces are deterministic ODE outputs.

Direct-on-line start from rest produces a large inrush and a torque transient.
That is expected reduced-order physics, not a claim about a specific laboratory
DOL event.

---

## 9. Expected outputs (healthy simulator, when implemented)

Time series (SI):

- `t` — time (s)
- `v_abc`, `i_abc` — reconstructed phase quantities
- `v_qd`, `i_qd`, `lambda_qd` — synchronous-frame electrical states
- `omega_m`, `omega_r`, `slip`
- `tau_e`, `tau_l` (`tau_l` constant)

Metadata: model name, convention, parameter provenance, scenario id.

**Healthy qualitative checks (not experimental claims):**

- After transients, \(i_{qs},i_{ds}\) nearly constant in the synchronous frame
- \(i_{abc}\) nearly balanced sinusoids at 50 Hz
- Slip \(>0\) for motoring with \(T_L>0\)
- Steady \(T_e\approx T_L+B\omega_m\)
- Slip/torque consistent in order of magnitude with the steady-state equivalent circuit

The model will **not** produce slot harmonics, PWM harmonics, or \(f_s(1\pm 2s)\)
sidebands in the healthy milestone.

---

## 10. Faults (deferred)

### 10.1 First milestone

**None.** Do not implement unbalance, stator \(R\) asymmetry, or rotor asymmetry yet.

### 10.2 Later (documentation only)

| Fault | Honest representation | Must not claim |
| --- | --- | --- |
| Supply unbalance | Unequal \(v_{abc}\); **same** healthy motor ODEs | “stator damage” |
| Stator \(R\) unbalance | Unequal phase resistances; balanced \(dq\) plant **not** sufficient | turn-to-turn short |
| Broken rotor bar | Optional **proxy** (\(R_{qr}\neq R_{dr}\) or \(2\omega_r\) modulation) **or** a reduced coupled-circuit cage | physically complete bar-break; bar count; severity calibration |

**Broken-bar proxy \(\neq\) broken-bar physics.** Sideband *existence* near
\(f_s(1\pm 2s)\) can be illustrated by a proxy; **severity and bar identity cannot**.

Do not paste sinusoids at \(f_s(1\pm 2s)\) onto healthy currents.

---

## 11. Limitations

1. Sinusoidal MMF and uniform gap: no slotting, no most eccentricity signatures.
2. Single referred cage: no individual bars or end-ring segments.
3. Linear magnetics: no saturation distortion of magnetizing current.
4. Lumped \(J,B\): no torsional load oscillations.
5. Ideal 50 Hz sinusoids: no VFD switching.
6. Literature parameters \(\neq\) a calibrated digital twin.
7. No experimental validation in this repository.
8. Numerical step size, windowing, and start-up can create spurious spectra;
   tests must separate numerics from physics.

---

## 12. Tests defined for the equations (implementable without the ODE solver)

These tests lock the math **before** time integration.

| ID | Test | Pass criterion |
| --- | --- | --- |
| T1 | Park then inverse Park | Recovers \(abc\) (balanced and with zero-sequence) |
| T2 | Balanced 50 Hz voltages, \(\theta=\omega_e t\) | \(v_{qs}=V_s\), \(v_{ds}=0\), \(v_{0s}=0\) |
| T3 | Flux map then inversion | \(\mathbf{i}\to\boldsymbol{\lambda}\to\mathbf{i}\) identity |
| T4 | Torque \(\lambda i\) vs \(L_m i_s\times i_r\) | Agree for linear magnetics |
| T5 | Parameter provenance | Example machine is `literature_example`, not `identified` |
| T6 | Integrator | RK45 succeeds; states remain finite; start-from-rest IC |
| T7 | Steady physics | \(T_e\approx T_L+B\omega_m\); slip \(>0\); power balance; equivalent-circuit torque at the same slip |

---

## 13. Assumptions (first milestone)

- Linear, constant \(L_s,L_r,L_m\)
- Sinusoidal winding distribution; no space harmonics
- Uniform air gap; no eccentricity
- Single squirrel-cage equivalent; rotor referred to stator
- Isolated-neutral star; no integrated zero-sequence electrical state
- Ideal balanced 50 Hz voltage source
- Constant \(T_L\), constant \(J\), viscous friction only
- Classical Krause \(2/3\) Park and \(\tfrac{3}{2}\) torque
- Parameters and \(T_L\) are **assumed / literature-example**

---

## 14. Remaining uncertainties (not blocking the healthy plant)

Locked for this milestone: RK45 with `max_step=10^{-4}\,\mathrm{s}`; start from rest.

Still open:

1. Whether a later thesis chapter will add a **public current dataset** or a **lab** motor (replace parameters; do not fabricate).
2. Whether broken-bar work, if ever, stays a two-axis **proxy** or becomes a small coupled-circuit model.
3. Fan-type load comparison (optional, after constant-\(T_L\) verification).
4. Whether a near-steady initial-flux guess is added later for monitoring-window studies (start-from-rest remains the default).

---

## 15. Roadmap

See `README.md`. Rule: **no fault or MCSA claims before the healthy plant is verified.**
Do not implement fault ODEs until explicitly approved.
