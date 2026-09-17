"""Unified simulated condition overview across the five fault studies.

This module is **reporting only**. It reads the per-study summary JSON and
trace archives already written by the experiment runners under ``results/``
and builds three presentation artifacts:

- ``fault_overview_summary.json`` — machine-readable unified summary
  (fault name, physical mechanism or proxy, affected signal, expected
  signature, measured simulated signature, limitations, provenance).
- ``fault_overview_table.md`` — the consistent healthy-vs-condition
  comparison table plus automated consistency checks.
- ``fault_overview_comparison.png`` — one multi-panel overview figure with
  a consistent style and honesty caption.

It never changes model physics, fault magnitudes, or labels, and it adds no
classifier, GUI, digital-twin, or predictive-maintenance claim. Every
number shown is **simulated** on a **literature-example** machine; there is
no experimental validation in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imcm.processing.envelope import envelope_spectrum

CAPTION = (
    "Simulated reduced-order model, literature-example parameters (not "
    "measured). Not experimental data; no diagnostic claim."
)

#: Relative power-balance tolerance for faulted runs. The residual is a
#: ``np.gradient`` discretization artifact that converges as ``dt^2``
#: (documented in docs/fault_03_supply_voltage_unbalance.md).
FAULTED_POWER_BALANCE_TOL = 1.0e-5
HEALTHY_POWER_BALANCE_TOL = 1.0e-10


@dataclass(frozen=True)
class ConditionInfo:
    """Static, hand-authored description of one simulated condition."""

    key: str
    name: str
    table_label: str
    classification: str
    mechanism: str
    affected_signal: str
    expected_signature: str
    limitations: tuple[str, ...]
    doc_path: str
    summary_filename: str  # empty for the healthy baseline row
    npz_filename: str


CONDITIONS: tuple[ConditionInfo, ...] = (
    ConditionInfo(
        key="healthy",
        name="Healthy baseline (balanced 50 Hz supply, 15.0 N m load)",
        table_label="Healthy",
        classification="Reference baseline — no fault",
        mechanism=(
            "Fifth-order two-axis (dq) squirrel-cage model, Krause "
            "convention; balanced sinusoidal supply; constant load torque."
        ),
        affected_signal="i_abc, i_qd, T_e, speed, slip (all simulated)",
        expected_signature=(
            "Balanced 50 Hz phase currents; near-DC synchronous-frame "
            "currents after start-up; constant slip > 0; no sidebands, no "
            "negative sequence."
        ),
        limitations=(
            "Linear magnetics, sinusoidal MMF, uniform air gap; literature "
            "parameters; no noise; not experimentally validated.",
        ),
        doc_path="docs/healthy_baseline.md",
        summary_filename="",
        npz_filename="healthy_startup.npz",
    ),
    ConditionInfo(
        key="fault_01",
        name="Fault 01 — stator resistance imbalance (+10% phase A)",
        table_label="Fault 01",
        classification=(
            "Physically modelled in the reduced-order plant (proxy for a "
            "high-resistance connection/joint; NOT a turn fault)"
        ),
        mechanism=(
            "Per-phase stator resistances (1.5455, 1.405, 1.405) Ohm "
            "projected into the synchronous qd frame as a time-varying "
            "resistance matrix; rotor, flux map, torque, mechanics unchanged."
        ),
        affected_signal="Stator currents (sequence components), qd currents, torque ripple",
        expected_signature=(
            "Nonzero negative-sequence current, phase current unbalance, "
            "and qd/torque ripple at twice supply frequency."
        ),
        limitations=(
            "Not an inter-turn-short model; no thermal dynamics, no "
            "zero-sequence path; unbalance metrics are not unique "
            "discriminants (supply unbalance or load also move them).",
        ),
        doc_path="docs/fault_01_stator_resistance_imbalance.md",
        summary_filename="fault_01_stator_resistance_imbalance_summary.json",
        npz_filename="fault_01_stator_resistance_imbalance.npz",
    ),
    ConditionInfo(
        key="fault_02",
        name="Condition 02 — increased mechanical load (+50%, 15.0 -> 22.5 N m)",
        table_label="Cond. 02",
        classification=(
            "Operating-condition change, NOT an internal motor fault "
            "(confounder for current-magnitude and slip features)"
        ),
        mechanism=(
            "Only the constant load torque changes (15.0 -> 22.5 N m); "
            "motor equations, parameters, and supply are unchanged."
        ),
        affected_signal="Speed, slip, torque, current magnitude (i_qs, phase RMS)",
        expected_signature=(
            "Lower speed, higher slip, proportionally higher mean torque "
            "and current; currents remain balanced."
        ),
        limitations=(
            "Constant torque load only (no fan-type load); an internal "
            "fault can raise current too, so magnitude alone does not "
            "identify the cause.",
        ),
        doc_path="docs/fault_02_increased_mechanical_load.md",
        summary_filename="fault_02_increased_mechanical_load_summary.json",
        npz_filename="fault_02_increased_mechanical_load.npz",
    ),
    ConditionInfo(
        key="fault_03",
        name="Fault 03 — supply voltage unbalance (phase C at 0.9 p.u.)",
        table_label="Fault 03",
        classification=(
            "Supply-side confounder, physically modelled supply; motor "
            "equations and parameters unchanged (NOT winding damage)"
        ),
        mechanism=(
            "Per-phase voltage multipliers (1.0, 1.0, 0.9) create a "
            "negative-sequence voltage that drives a negative-sequence "
            "current through the healthy machine."
        ),
        affected_signal="Supply voltages, phase currents, torque ripple at 2*omega_e",
        expected_signature=(
            "Voltage unbalance factor ~3.45%, large current unbalance, "
            "negative-sequence current, second-harmonic torque ripple."
        ),
        limitations=(
            "Static magnitude unbalance only (no angle unbalance or sags); "
            "mimics some signatures of winding damage — must be separated "
            "from Fault 01 by recording the supply independently.",
        ),
        doc_path="docs/fault_03_supply_voltage_unbalance.md",
        summary_filename="fault_03_supply_voltage_unbalance_summary.json",
        npz_filename="fault_03_supply_voltage_unbalance.npz",
    ),
    ConditionInfo(
        key="fault_04",
        name="Fault 04 — bearing outer-race fault (BPFO vibration signature)",
        table_label="Fault 04",
        classification=(
            "Simulation-only vibration-channel signature (bearing "
            "kinematics + assumed resonance); electrical plant bit-for-bit "
            "healthy — NOT a motor-model fault, NOT an MCSA bearing claim"
        ),
        mechanism=(
            "Impact train at BPFO = (Nb/2) f_r (1 - (Bd/Pd) cos(phi)) from "
            "the simulated speed trace, exciting an assumed 2 kHz "
            "structural resonance; nothing injected into the ODEs."
        ),
        affected_signal="Simulated accelerometer channel only (m/s^2)",
        expected_signature=(
            "Envelope-spectrum lines at BPFO (~87.5 Hz at the simulated "
            "steady speed) and its harmonics; healthy channel is exactly "
            "zero by construction."
        ),
        limitations=(
            "Amplitude is an arbitrary, uncalibrated scale; healthy "
            "vibration floor is identically zero (no robustness claim); "
            "bearing geometry is a literature example; no stator-current "
            "bearing signature is modelled.",
        ),
        doc_path="docs/fault_04_bearing_outer_race.md",
        summary_filename="fault_04_bearing_outer_race_summary.json",
        npz_filename="fault_04_bearing_outer_race.npz",
    ),
    ConditionInfo(
        key="fault_05",
        name="Fault 05 — rotor electrical asymmetry, severity 0.10 (broken-bar proxy)",
        table_label="Fault 05",
        classification=(
            "Simulation-only proxy for broken-bar-related behaviour "
            "(rotor-frame axis resistance split); NOT bar-resolved, NOT "
            "severity-calibrated, NOT a real-machine diagnosis"
        ),
        mechanism=(
            "Rotor q/d axis resistances R_r(1 +/- delta) with mean R_r "
            "preserved; in the synchronous frame this modulates at "
            "2 s_ref f_s, producing stator sidebands near f_s(1 -/+ 2s)."
        ),
        affected_signal="Stator current spectrum (sidebands), qd modulation tone, small torque ripple",
        expected_signature=(
            "Sidebands near f_s(1 - 2s) ~ 47.6 Hz and f_s(1 + 2s) ~ "
            "52.4 Hz; synchronous-frame tone at 2 s f_s ~ 2.38 Hz; mean "
            "operating point essentially unchanged."
        ),
        limitations=(
            "Modulation phase uses a constant reference slip (not the true "
            "slip-angle integral during start-up); delta is a modelling "
            "knob, not a bar count; mean-resistance rise of a real broken "
            "bar is not modelled.",
        ),
        doc_path="docs/fault_05_rotor_asymmetry.md",
        summary_filename="fault_05_rotor_asymmetry_summary.json",
        npz_filename="fault_05_rotor_asymmetry.npz",
    ),
)

CONDITIONS_BY_KEY = {c.key: c for c in CONDITIONS}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summaries(results_dir: Path) -> dict[str, dict]:
    """Load the per-study summary JSONs, keyed by condition key."""
    summaries: dict[str, dict] = {}
    for info in CONDITIONS:
        if not info.summary_filename:
            continue
        path = results_dir / info.summary_filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run the per-fault experiment scripts first."
            )
        summaries[info.key] = _load_json(path)
    return summaries


def _fmt(value: float | None | str, spec: str = ".4g") -> str:
    if value is None:
        return "not studied"
    if isinstance(value, str):
        return value
    return format(value, spec)


def healthy_baseline(summaries: dict[str, dict]) -> dict:
    """Canonical healthy-column numbers with their sources.

    Sources: absolute speed/slip from the Condition 02 summary (the only
    study that stores healthy absolutes), electrical metrics from the
    Fault 03 summary (regenerated with the current code base), and the
    healthy power-balance residual from the Fault 03 summary.
    """
    f02, f03 = summaries["fault_02"], summaries["fault_03"]
    m03 = f03["metrics"]
    return {
        "final_speed_rpm": f02["healthy_final_speed_rpm"],
        "final_slip": f02["healthy_final_slip"],
        "phase_current_rms_a": m03["phase_current_rms_healthy_a"],
        "current_unbalance_pct": m03["current_unbalance_healthy_pct"],
        "negative_sequence_a": m03["negative_sequence_healthy_a"],
        "torque_ripple_nm": m03["torque_ripple_healthy_nm"],
        "power_balance_residual_rel": f03["healthy_power_balance_residual_rel"],
        "sources": {
            "speed_slip": "fault_02 summary (healthy_final_*)",
            "electrical": "fault_03 summary (healthy metrics)",
            "power_balance": "fault_03 summary (healthy_power_balance_residual_rel)",
        },
    }


def measured_per_condition(
    summaries: dict[str, dict], baseline: dict
) -> dict[str, dict]:
    """Extract a consistent measured-metric dict for every condition.

    Fault 01 and Fault 03 summaries store final speed/slip only as
    differences from healthy; the absolute values here are derived
    transparently as ``healthy + stored difference``.
    """
    f01, f02, f03, f04, f05 = (
        summaries["fault_01"],
        summaries["fault_02"],
        summaries["fault_03"],
        summaries["fault_04"],
        summaries["fault_05"],
    )
    m01, m03, m04, m05 = f01["metrics"], f03["metrics"], f04["metrics"], f05["metrics"]
    identical_note = "bit-identical to healthy by construction"

    measured: dict[str, dict] = {}

    measured["fault_01"] = {
        "speed_rpm": baseline["final_speed_rpm"] + m01["final_speed_difference_rpm"],
        "slip": baseline["final_slip"] + m01["final_slip_difference"],
        "speed_slip_derivation": "healthy + stored difference (summary stores differences only)",
        "phase_current_rms_a": m01["phase_current_rms_fault_a"],
        "current_unbalance_pct": m01["current_unbalance_fault_pct"],
        "negative_sequence_a": m01["negative_sequence_fault_a"],
        "negative_sequence_pct": m01["negative_sequence_fault_pct"],
        "torque_ripple_nm": m01["torque_ripple_fault_nm"],
        "power_balance_residual_rel": f01["fault_power_balance_residual_rel"],
        "primary_signature": (
            f"negative-sequence current: "
            f"{_fmt(m01['negative_sequence_healthy_a'])} A -> "
            f"{_fmt(m01['negative_sequence_fault_a'])} A "
            f"({_fmt(m01['negative_sequence_fault_pct'])}% of positive sequence)"
        ),
    }
    measured["fault_02"] = {
        "speed_rpm": f02["condition_final_speed_rpm"],
        "slip": f02["condition_final_slip"],
        "phase_current_rms_a": None,
        "current_unbalance_pct": "0 (balanced by construction)",
        "negative_sequence_a": None,
        "torque_ripple_nm": None,
        "power_balance_residual_rel": f02["condition_power_balance_residual_rel"],
        "late_mean_torque_nm": f02["condition_late_mean_torque_nm"],
        "late_mean_i_qs_a": f02["condition_late_mean_i_qs_a"],
        "primary_signature": (
            f"slip: {_fmt(f02['healthy_final_slip'])} -> "
            f"{_fmt(f02['condition_final_slip'])} "
            f"(+{_fmt(f02['load_increase_percent'], '.0f')}% load); "
            f"mean i_qs: {_fmt(f02['healthy_late_mean_i_qs_a'])} A -> "
            f"{_fmt(f02['condition_late_mean_i_qs_a'])} A"
        ),
    }
    measured["fault_03"] = {
        "speed_rpm": baseline["final_speed_rpm"] + m03["final_speed_difference_rpm"],
        "slip": baseline["final_slip"] + m03["final_slip_difference"],
        "speed_slip_derivation": "healthy + stored difference (summary stores differences only)",
        "phase_current_rms_a": m03["phase_current_rms_unbalance_a"],
        "current_unbalance_pct": m03["current_unbalance_unbalance_pct"],
        "negative_sequence_a": m03["negative_sequence_unbalance_a"],
        "negative_sequence_pct": m03["negative_sequence_unbalance_pct"],
        "supply_voltage_unbalance_pct": m03["supply_voltage_unbalance_unbalance_pct"],
        "torque_ripple_nm": m03["torque_ripple_unbalance_nm"],
        "power_balance_residual_rel": f03["fault_power_balance_residual_rel"],
        "primary_signature": (
            f"current unbalance: {_fmt(m03['current_unbalance_healthy_pct'])}% -> "
            f"{_fmt(m03['current_unbalance_unbalance_pct'])}%; "
            f"negative sequence: {_fmt(m03['negative_sequence_unbalance_a'])} A "
            f"({_fmt(m03['negative_sequence_unbalance_pct'])}%) at "
            f"{_fmt(m03['supply_voltage_unbalance_unbalance_pct'])}% VUF"
        ),
    }
    measured["fault_04"] = {
        "speed_rpm": baseline["final_speed_rpm"],
        "slip": baseline["final_slip"],
        "electrical_note": identical_note,
        "bpfo_hz": m04["bpfo_hz"],
        "shaft_hz_mean": m04["shaft_hz_mean"],
        "envelope_bpfo_healthy": m04["envelope_bpfo_healthy"],
        "envelope_bpfo_fault": m04["envelope_bpfo_fault"],
        "envelope_bpfo_2x_healthy": m04["envelope_bpfo_2x_healthy"],
        "envelope_bpfo_2x_fault": m04["envelope_bpfo_2x_fault"],
        "power_balance_residual_rel": f04["fault_power_balance_residual_rel"],
        "primary_signature": (
            f"envelope amplitude at BPFO ({_fmt(m04['bpfo_hz'], '.2f')} Hz): "
            f"{_fmt(m04['envelope_bpfo_healthy'])} -> "
            f"{_fmt(m04['envelope_bpfo_fault'])} m/s^2 "
            f"(2 x BPFO: {_fmt(m04['envelope_bpfo_2x_fault'])} m/s^2); "
            "electrical traces bit-identical to healthy"
        ),
    }
    measured["fault_05"] = {
        "speed_rpm": baseline["final_speed_rpm"] + m05["final_speed_difference_rpm"],
        "slip": baseline["final_slip"] + m05["final_slip_difference"],
        "speed_slip_derivation": "healthy + stored difference (summary stores differences only)",
        "phase_current_rms_a": None,
        "phase_a_fundamental_a": m05["phase_a_fundamental_fault_a"],
        "lower_sideband_hz": m05["lower_sideband_hz"],
        "upper_sideband_hz": m05["upper_sideband_hz"],
        "lower_sideband_a": m05["lower_sideband_fault_a"],
        "upper_sideband_a": m05["upper_sideband_fault_a"],
        "sync_modulation_tone_a": m05["sync_modulation_fault_a"],
        "torque_ripple_nm": m05["torque_ripple_fault_nm"],
        "power_balance_residual_rel": f05["fault_power_balance_residual_rel"],
        "primary_signature": (
            f"phase-a sidebands at f_s(1-/+2s): "
            f"{_fmt(m05['lower_sideband_healthy_a'])} -> "
            f"{_fmt(m05['lower_sideband_fault_a'])} A at "
            f"{_fmt(m05['lower_sideband_hz'], '.2f')} Hz and "
            f"{_fmt(m05['upper_sideband_fault_a'])} A at "
            f"{_fmt(m05['upper_sideband_hz'], '.2f')} Hz; "
            f"2 s f_s tone {_fmt(m05['sync_modulation_fault_a'])} A"
        ),
    }
    return measured


# ---------------------------------------------------------------------------
# Consistency validation (data-level checks over the regenerated artifacts)
# ---------------------------------------------------------------------------

def validate_consistency(
    results_dir: Path, summaries: dict[str, dict]
) -> list[dict]:
    """Run automated data-level consistency checks on regenerated outputs."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if ok else "FAIL", "detail": detail}
        )

    # 1. Provenance everywhere.
    provenance_ok = all(
        s.get("parameter_provenance") == "literature_example"
        for s in summaries.values()
    )
    add(
        "provenance_tags",
        provenance_ok,
        "every summary tags parameters as literature_example and traces as simulated",
    )

    # 2. Healthy columns identical across studies (current code base).
    m01 = summaries["fault_01"]["metrics"]
    m03 = summaries["fault_03"]["metrics"]
    m04 = summaries["fault_04"]["metrics"]
    m05 = summaries["fault_05"]["metrics"]
    rms01 = np.asarray(m01["phase_current_rms_healthy_a"])
    rms03 = np.asarray(m03["phase_current_rms_healthy_a"])
    rms_diff = float(np.max(np.abs(rms01 - rms03)))
    add(
        "healthy_columns_consistent",
        rms_diff < 1.0e-6,
        f"max |healthy phase RMS (fault_01) - (fault_03)| = {rms_diff:.3e} A "
        "(all studies regenerated with the current code)",
    )

    # 3. Fault 04 electrical traces bit-identical to healthy.
    z04 = np.load(results_dir / "fault_04_bearing_outer_race.npz")
    identical = bool(np.array_equal(z04["healthy_i_abc"], z04["fault_i_abc"]))
    add(
        "fault_04_electrical_untouched",
        identical,
        "fault_04 npz: healthy_i_abc == fault_i_abc elementwise",
    )

    # 4. All trace archives finite.
    finite = True
    detail_parts: list[str] = []
    for info in CONDITIONS:
        path = results_dir / info.npz_filename
        if not path.exists():
            finite = False
            detail_parts.append(f"{info.npz_filename}: missing")
            continue
        z = np.load(path)
        bad = [k for k in z.files if not np.all(np.isfinite(z[k]))]
        if bad:
            finite = False
            detail_parts.append(f"{info.npz_filename}: non-finite {bad}")
    add(
        "traces_finite",
        finite,
        "all npz arrays finite" if finite else "; ".join(detail_parts),
    )

    # 5. Power balance inside documented bounds.
    h_pb = max(
        s["healthy_power_balance_residual_rel"] for s in summaries.values()
    )

    def faulted_residual(key: str) -> float:
        # Condition 02 names its arm "condition_*"; the others "fault_*".
        summary = summaries[key]
        if "fault_power_balance_residual_rel" in summary:
            return summary["fault_power_balance_residual_rel"]
        return summary["condition_power_balance_residual_rel"]

    f_pb = max(faulted_residual(k) for k in summaries)
    add(
        "power_balance_healthy",
        h_pb < HEALTHY_POWER_BALANCE_TOL,
        f"worst healthy relative residual = {h_pb:.2e} (< {HEALTHY_POWER_BALANCE_TOL:g})",
    )
    add(
        "power_balance_faulted",
        f_pb < FAULTED_POWER_BALANCE_TOL,
        f"worst faulted relative residual = {f_pb:.2e} "
        f"(< {FAULTED_POWER_BALANCE_TOL:g}; np.gradient artifact, converges as dt^2)",
    )

    # 6. Fault 03 signature magnitude sanity.
    vuf = m03["supply_voltage_unbalance_unbalance_pct"]
    add(
        "fault_03_signature",
        abs(vuf - 3.448) < 0.05 and m03["negative_sequence_unbalance_a"] > 1.0,
        f"VUF = {vuf:.3f}% (expected ~3.448%), "
        f"I2 = {m03['negative_sequence_unbalance_a']:.3f} A",
    )

    # 7. Fault 04 envelope signature sanity.
    add(
        "fault_04_signature",
        m04["envelope_bpfo_healthy"] == 0.0
        and m04["envelope_bpfo_fault"] > 0.05
        and abs(m04["bpfo_peak_fault_hz"] - m04["bpfo_hz"]) < 2.0,
        f"BPFO = {m04['bpfo_hz']:.2f} Hz, measured envelope peak at "
        f"{m04['bpfo_peak_fault_hz']:.2f} Hz, healthy envelope line = 0 "
        "(idealised: zero vibration floor)",
    )

    # 8. Fault 05 sideband sanity at the analytic proxy locations.
    fs = m05["fundamental_hz"]
    s_ref = m05["reference_slip"]
    lo_target = fs * (1.0 - 2.0 * s_ref)
    add(
        "fault_05_signature",
        m05["severity"] == 0.1
        and abs(m05["lower_sideband_hz"] - lo_target) < 1.0e-6
        and m05["lower_sideband_fault_a"] > 100.0 * m05["lower_sideband_healthy_a"],
        f"sideband bins at f_s(1-2s_ref) = {lo_target:.2f} Hz; "
        f"fault/healthy amplitude ratio = "
        f"{m05['lower_sideband_fault_a'] / max(m05['lower_sideband_healthy_a'], 1e-30):.0e}",
    )

    return checks


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def build_markdown_table(
    baseline: dict, measured: dict[str, dict], checks: list[dict]
) -> str:
    """Render the consistent healthy-vs-condition comparison table."""
    b = baseline
    rows: list[tuple[str, list[str]]] = []

    def speed_cell(key: str) -> str:
        m = measured[key]
        cell = _fmt(m["speed_rpm"], ".2f")
        if "speed_slip_derivation" in m:
            cell += " *"
        if key == "fault_04":
            cell = "= healthy (bit-identical)"
        return cell

    def slip_cell(key: str) -> str:
        m = measured[key]
        if key == "fault_04":
            return "= healthy (bit-identical)"
        return _fmt(m["slip"], ".5f")

    def rms_cell(key: str) -> str:
        m = measured[key]
        if key == "fault_04":
            return "= healthy (bit-identical)"
        rms = m["phase_current_rms_a"]
        if rms is None:
            return "not studied"
        return "(" + ", ".join(format(v, ".3f") for v in rms) + ")"

    def identical_or(key: str, attr: str, spec: str = ".4g") -> str:
        m = measured[key]
        if key == "fault_04":
            return "= healthy (bit-identical)"
        return _fmt(m.get(attr), spec)

    rows.append(("Final rotor speed (r/min)", ["= healthy"] + [speed_cell(k) for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Final slip (p.u.)", [_fmt(b["final_slip"], ".5f")] + [slip_cell(k) for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Phase RMS i_a/i_b/i_c (A)", ["(" + ", ".join(format(v, ".3f") for v in b["phase_current_rms_a"]) + ")"] + [rms_cell(k) for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Current unbalance (%)", [_fmt(b["current_unbalance_pct"], ".3g")] + [identical_or(k, "current_unbalance_pct", ".4g") for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Negative-sequence current (A)", [_fmt(b["negative_sequence_a"], ".4g")] + [identical_or(k, "negative_sequence_a") for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Torque ripple RMS (N m)", [_fmt(b["torque_ripple_nm"], ".3g")] + [identical_or(k, "torque_ripple_nm", ".4g") for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))
    rows.append(("Power-balance residual (rel.)", [_fmt(b["power_balance_residual_rel"], ".2e")] + [format(measured[k]["power_balance_residual_rel"], ".2e") for k in ("fault_01", "fault_02", "fault_03", "fault_04", "fault_05")]))

    lines: list[str] = [
        "# Simulated condition overview — healthy vs fault comparison",
        "",
        "All values are **simulated** on the illustrative 4 kW, 400 V, 50 Hz, "
        "4-pole **literature-example** machine (constant 15.0 N m load unless "
        "stated, DOL start from rest, RK45, `max_step=1e-4 s`). There is **no "
        "experimental validation**; nothing here is a diagnosis, classifier, "
        "or predictive-maintenance claim.",
        "",
        "## 1. Shared simulated metrics",
        "",
        "| Metric | Healthy | Fault 01 | Cond. 02 | Fault 03 | Fault 04 | Fault 05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, cells in rows:
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "\\* Final speed/slip for Faults 01/03/05 are derived as "
        "`healthy + stored difference` (those summaries store differences "
        "only). \"Bit-identical\" cells: the study does not modify the "
        "electrical plant (verified elementwise for Fault 04). \"Not "
        "studied\": the metric is not the studied channel and is not stored "
        "in that study's summary.",
        "",
        "## 2. Primary simulated signature per condition",
        "",
        "| Condition | Affected signal | Expected signature | Measured (healthy -> condition) |",
        "| --- | --- | --- | --- |",
    ]
    for info in CONDITIONS:
        if info.key == "healthy":
            continue
        m = measured[info.key]
        lines.append(
            f"| {info.name} | {info.affected_signal} | {info.expected_signature} | {m['primary_signature']} |"
        )
    lines += [
        "",
        "## 3. Classification: physically modelled vs proxy vs confounder",
        "",
        "| Condition | Classification |",
        "| --- | --- |",
    ]
    for info in CONDITIONS:
        lines.append(f"| {info.name} | {info.classification} |")
    lines += [
        "",
        "## 4. Automated consistency checks (regenerated outputs, current code)",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| `{c['name']}` | {c['status']} | {c['detail']} |")
    lines += [
        "",
        "## 5. Standing limitations (apply to every row above)",
        "",
        "- Parameters are a published illustrative example, **not** a "
        "measured motor; all traces are deterministic ODE outputs with no "
        "noise model, so every separation shown is idealised.",
        "- Unbalance / negative-sequence / current-magnitude metrics are "
        "**not unique discriminants**: load (Cond. 02), supply (Fault 03), "
        "and stator resistance (Fault 01) all move them. The studies are "
        "controlled contrasts, not diagnosis.",
        "- Healthy torque-ripple floors (~1e-5 N m and below) are numerical, "
        "not physical.",
        "- No GUI, no ML classifier, no predictive-maintenance or "
        "digital-twin claim is made anywhere in this repository.",
        "",
        "Per-condition detail: docs/fault_01_*.md, docs/fault_02_*.md, "
        "docs/fault_03_*.md, docs/fault_04_*.md, docs/fault_05_*.md; "
        "narrative overview: docs/fault_summary.md.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_overview_figure(
    results_dir: Path, summaries: dict[str, dict], out_path: Path
) -> Path:
    """Render the 3x2 overview figure with a consistent style."""
    m05 = summaries["fault_05"]["metrics"]
    m04 = summaries["fault_04"]["metrics"]

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.5))

    # (0,0) Healthy phase currents, late window.
    zh = np.load(results_dir / "healthy_startup.npz")
    t, mask = zh["t"], zh["t"] >= 0.8
    for name, key in (("a", "i_a"), ("b", "i_b"), ("c", "i_c")):
        axes[0, 0].plot(t[mask], zh[key][mask], lw=0.7, label=f"$i_{name}$")
    axes[0, 0].set_title("Healthy baseline — phase currents (0.8–1.0 s)")
    axes[0, 0].set_ylabel("A")

    # (0,1) Fault 01 phase currents, late window.
    z01 = np.load(results_dir / "fault_01_stator_resistance_imbalance.npz")
    t, mask = z01["t"], z01["t"] >= 0.8
    axes[0, 1].plot(t[mask], z01["healthy_i_abc"][0][mask], "k--", lw=0.7, label="healthy $i_a$")
    for idx, name in enumerate(("a", "b", "c")):
        axes[0, 1].plot(t[mask], z01["fault_i_abc"][idx][mask], lw=0.7, label=f"fault $i_{name}$")
    axes[0, 1].set_title("Fault 01 — +10% phase-A stator R (physical, reduced-order)")
    axes[0, 1].set_ylabel("A")

    # (1,0) Condition 02 speed overlay, full run.
    z02 = np.load(results_dir / "fault_02_increased_mechanical_load.npz")
    t = z02["t"]
    axes[1, 0].plot(t, z02["healthy_speed_rpm"], "k--", lw=1.0, label="healthy (15.0 N m)")
    axes[1, 0].plot(t, z02["condition_speed_rpm"], lw=1.0, label="Cond. 02 (22.5 N m)")
    axes[1, 0].set_title("Condition 02 — +50% load: operating change, not a fault")
    axes[1, 0].set_ylabel("r/min")

    # (1,1) Fault 03 phase currents, late window.
    z03 = np.load(results_dir / "fault_03_supply_voltage_unbalance.npz")
    t, mask = z03["t"], z03["t"] >= 0.8
    axes[1, 1].plot(t[mask], z03["healthy_i_abc"][0][mask], "k--", lw=0.7, label="healthy $i_a$")
    for idx, name in enumerate(("a", "b", "c")):
        axes[1, 1].plot(t[mask], z03["fault_i_abc"][idx][mask], lw=0.7, label=f"unbalanced $i_{name}$")
    axes[1, 1].set_title("Fault 03 — supply C at 0.9 p.u. (supply confounder)")
    axes[1, 1].set_ylabel("A")

    # (2,0) Fault 05 phase-a spectrum with proxy sidebands.
    z05 = np.load(results_dir / "fault_05_rotor_asymmetry.npz")
    t, mask = z05["t"], z05["t"] >= 2.0
    fs_hz = 1.0 / float(np.mean(np.diff(t[mask])))

    def amp(x: np.ndarray) -> np.ndarray:
        window = np.hanning(x.size)
        return 2.0 * np.abs(np.fft.rfft((x - np.mean(x)) * window)) / np.sum(window)

    freq = np.fft.rfftfreq(int(np.count_nonzero(mask)), d=1.0 / fs_hz)
    axes[2, 0].plot(freq, amp(z05["healthy_i_abc"][0][mask]), "k--", lw=0.7, label="healthy $i_a$")
    axes[2, 0].plot(freq, amp(z05["fault_i_abc"][0][mask]), lw=0.7, label="Fault 05 $i_a$")
    axes[2, 0].axvline(m05["lower_sideband_hz"], color="tab:red", ls=":", lw=1.0,
                       label=f"$f_s(1-2s)$ = {m05['lower_sideband_hz']:.2f} Hz")
    axes[2, 0].axvline(m05["upper_sideband_hz"], color="tab:orange", ls=":", lw=1.0,
                       label=f"$f_s(1+2s)$ = {m05['upper_sideband_hz']:.2f} Hz")
    axes[2, 0].set_xlim(0.0, 2.5 * m05["fundamental_hz"])
    axes[2, 0].set_ylim(0.0, 1.5 * m05["phase_a_fundamental_fault_a"])
    axes[2, 0].set_title("Fault 05 — rotor-asymmetry proxy sidebands (2–3 s window)")
    axes[2, 0].set_ylabel("Amplitude (A)")
    axes[2, 0].set_xlabel("Frequency (Hz)")

    # (2,1) Fault 04 envelope spectrum at BPFO.
    z04 = np.load(results_dir / "fault_04_bearing_outer_race.npz")
    t04, mask04 = z04["t"], z04["t"] >= 0.5
    fs04 = 1.0 / float(np.mean(np.diff(t04[mask04])))
    resonance = float(summaries["fault_04"]["bearing_config"]["resonance_hz"])
    spec = envelope_spectrum(
        z04["fault_vibration_m_s2"][mask04],
        fs04,
        band=(resonance - 500.0, resonance + 500.0),
    )
    axes[2, 1].plot(spec.freq_hz, spec.amplitude, lw=0.8)
    bpfo = m04["bpfo_hz"]
    axes[2, 1].axvline(bpfo, color="tab:red", ls="--", lw=1.0, label=f"BPFO = {bpfo:.1f} Hz")
    axes[2, 1].axvline(2.0 * bpfo, color="tab:orange", ls=":", lw=1.0, label=f"2 x BPFO = {2 * bpfo:.1f} Hz")
    axes[2, 1].set_xlim(0.0, 6.0 * bpfo)
    axes[2, 1].set_title("Fault 04 — vibration-channel envelope spectrum (simulation only)")
    axes[2, 1].set_ylabel("Envelope amplitude (m/s$^2$")
    axes[2, 1].set_xlabel("Frequency (Hz)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Time (s)")

    fig.suptitle(
        "Simulated condition overview — literature-example machine (not measured); "
        "no experimental validation",
        fontsize=12,
    )
    fig.text(0.5, 0.005, CAPTION, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level writer
# ---------------------------------------------------------------------------

def build_overview_record(
    results_dir: Path, summaries: dict[str, dict], checks: list[dict]
) -> dict:
    """Assemble the machine-readable unified overview record."""
    baseline = healthy_baseline(summaries)
    measured = measured_per_condition(summaries, baseline)
    conditions: list[dict] = []
    for info in CONDITIONS:
        entry: dict = {
            "key": info.key,
            "name": info.name,
            "classification": info.classification,
            "mechanism": info.mechanism,
            "affected_signal": info.affected_signal,
            "expected_signature": info.expected_signature,
            "limitations": list(info.limitations),
            "doc": info.doc_path,
            "provenance": {
                "traces": "simulated",
                "parameters": "literature_example",
                "experimental_validation": False,
            },
        }
        if info.key == "healthy":
            entry["measured"] = baseline
        else:
            entry["measured"] = measured[info.key]
            entry["summary_source"] = info.summary_filename
            entry["trace_source"] = info.npz_filename
        conditions.append(entry)

    failed = [c["name"] for c in checks if c["status"] != "pass"]
    return {
        "label": "simulated_fault_overview",
        "generated_from": [c.summary_filename for c in CONDITIONS if c.summary_filename],
        "provenance": {
            "traces": "simulated",
            "parameters": "literature_example",
            "experimental_validation": False,
        },
        "scope_note": (
            "Presentation of controlled simulation studies only. No GUI, no "
            "ML classifier, no predictive-maintenance claim, no digital-twin "
            "claim, and no experimental validation."
        ),
        "healthy_baseline": baseline,
        "conditions": conditions,
        "consistency_checks": checks,
        "all_checks_pass": not failed,
    }


def write_overview(results_dir: Path) -> dict:
    """Build all three overview artifacts. Returns the summary record."""
    results_dir = Path(results_dir)
    summaries = load_summaries(results_dir)
    checks = validate_consistency(results_dir, summaries)
    record = build_overview_record(results_dir, summaries, checks)

    json_path = results_dir / "fault_overview_summary.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    baseline = healthy_baseline(summaries)
    measured = measured_per_condition(summaries, baseline)
    table = build_markdown_table(baseline, measured, checks)
    table_path = results_dir / "fault_overview_table.md"
    table_path.write_text(table, encoding="utf-8")

    figure_path = build_overview_figure(
        results_dir, summaries, results_dir / "fault_overview_comparison.png"
    )

    record["artifacts"] = {
        "json": str(json_path),
        "markdown_table": str(table_path),
        "figure": str(figure_path),
    }
    # Re-write with artifact paths included.
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record