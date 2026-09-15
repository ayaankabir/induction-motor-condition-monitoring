# Healthy induction-motor baseline

This note records the **approved healthy plant** before any fault modelling.
The model is **numerically checked and internally consistent, but not experimentally validated.**

All motor parameters below are **assumed illustrative / literature-example values**.
They are **not** laboratory measurements, identified nameplate data, or a digital twin of a specific machine.

Full equations: [`modeling-plan.md`](modeling-plan.md).

## Model scope

- Fifth-order squirrel-cage two-axis flux model (four electrical states + mechanical speed).
- Classical Krause \(qd0\) convention (factor \(2/3\); torque and power use \(\tfrac{3}{2}\)).
- Healthy machine only. No fault ODEs, no inverter, no dashboard.
- Outputs are **simulated** traces: \(i_{abc}\), \(\omega_m\), \(T_e\), fluxes, slip.

## Assumptions

- Linear, constant \(L_s,L_r,L_m\); sinusoidal MMF; uniform air gap.
- Single stator-referred cage; \(v_{qr}=v_{dr}=0\).
- Isolated-neutral star; no integrated zero-sequence electrical state.
- Constant inertia \(J\) and viscous friction \(B\).
- Direct-on-line start from rest (zero fluxes, \(\omega_m=0\)).

## Motor parameters

Set name: `illustrative_4kW_400V_50Hz_4pole`. Provenance: `literature_example`.

| Parameter | Assumed value | Unit |
| --- | --- | --- |
| \(R_s\) | 1.405 | Ω |
| \(R_r\) (stator-referred) | 1.395 | Ω |
| \(L_{ls}=L_{lr}\) | 0.005839 | H |
| \(L_m\) | 0.1722 | H |
| Poles \(P\) | 4 | — |
| \(J\) | 0.0131 | kg·m² |
| \(B\) | 0.002985 | N·m·s |
| Rated frequency (metadata) | 50 | Hz |
| Rated line-line RMS (metadata) | 400 | V |
| Rated mechanical power (metadata) | 4000 | W |
| Rated speed (metadata) | 1430 | r/min |

Derived metadata (not measured): synchronous speed \(1500\) r/min; assumed rated torque \(\approx 26.71\) N·m.

## Supply and load

- Balanced three-phase sinusoids, **50 Hz**, \(V_{ll,\mathrm{rms}}=400\) V, star. **No PWM.**
- Constant load torque \(T_L=15\) N·m (about \(0.56\) of the assumed rated torque). Assumed scenario, not a dynamometer setting.

## Numerical solver

- Adaptive RK45 (`scipy.integrate.solve_ivp`, `method="RK45"`).
- `max_step = 10^{-4}` s, `rtol = 10^{-6}`, `atol = 10^{-8}`.
- Typical start-up window: \(t\in[0,1]\) s.

## Validation checks (simulation only)

Unit tests check finite states; start-from-rest IC; \(i_a+i_b+i_c\approx 0\); Krause round-trip on simulated currents and voltages; motoring slip and \(T_e\approx T_L+B\omega_m\); power identity; T-equivalent circuit at the same slip; late-window current frequency near 50 Hz (FFT and zero-crossings). Synthetic 40/60 Hz tones are rejected by the frequency estimators.

These checks do **not** constitute experimental validation.

## Key simulated results (\(T_L=15\) N·m, literature-example parameters)

Steady window \(t\ge 0.8\) s (from `results/healthy_startup_summary.json` after running `experiments/run_healthy_startup.py`):

- Mean speed \(\approx 1464\) r/min (below 1500 r/min).
- Mean slip \(\approx 0.0238>0\).
- Mean \(T_e\approx 15.46\) N·m \(\approx T_L+B\omega_m\).
- Phase-\(a\) RMS \(\approx 5.57\) A.
- DOL inrush peaks on the order of \(80\) A; peak \(T_e\) on the order of \(146\) N·m.

Reproduce with:

```bash
python -m pytest
python experiments/run_healthy_startup.py
```

Generated PNGs/NPZ/JSON under `results/` are **not** committed; they are regenerated locally.

## Limitations

- No slotting, saturation, eccentricity, PWM, or bar-resolved cage.
- Low assumed \(J\) yields a fast start-up and a brief reverse-speed wiggle at \(t=0^+\).
- Parameters and \(T_L\) are assumed. Do not treat traces as plant measurements.
- Fault simulation is **not** part of this baseline.
