"""Diagnostic explanation and reasoning generator for simulated conditions.

This module is **reporting only**. It produces structured diagnostic explanations
for why each simulated condition exhibits its specific classification, based
strictly on existing stored summary metrics, physical mechanisms, and cross-condition
discrimination rules.

It enforces the repository's scientific honesty rules:
- No machine learning, probabilistic scores, anomaly scores, or confidence scores.
- Every metric cited is sourced directly from stored simulation records.
- Confounders (load, supply unbalance) and proxies (rotor asymmetry, bearing
  vibration) are explicitly noted with their physical and diagnostic limitations.
- Simulation-only scope: no experimental validation, no digital twin, no
  confirmed real-machine diagnosis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricEvidence:
    """A single piece of metric evidence supporting or qualifying a condition."""

    metric_name: str
    observed_value: float | str | None
    baseline_value: float | str | None
    unit: str | None
    diagnostic_role: str  # "primary_discriminant" | "confounder_check" | "normal_baseline" | "unaffected_channel"
    interpretation: str


@dataclass(frozen=True)
class ConditionExplanation:
    """Structured diagnostic explanation for a simulated condition."""

    condition_id: str
    classification_category: str
    classification_label: str
    physical_mechanism: str
    why_classified: str
    supporting_evidence: list[MetricEvidence]
    discrimination_vs_confounders: list[str]
    standing_limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt_val(val: float | None, fmt: str = ".4g") -> str:
    if val is None:
        return "not studied"
    return format(val, fmt)


def generate_explanation(
    condition_id: str, summary_data: dict[str, Any]
) -> ConditionExplanation:
    """Generate structured diagnostic explanation for a given condition and summary.

    Parameters
    ----------
    condition_id : str
        The condition identifier ("healthy", "fault-01", "condition-02",
        "fault-03", "fault-04", "fault-05").
    summary_data : dict[str, Any]
        The loaded JSON dictionary for the corresponding summary file.

    Returns
    -------
    ConditionExplanation
        The structured diagnostic explanation record.
    """
    metrics = summary_data.get("metrics", {})

    if condition_id == "healthy":
        return ConditionExplanation(
            condition_id="healthy",
            classification_category="baseline",
            classification_label="Reference baseline — no fault",
            physical_mechanism=(
                "Fifth-order two-axis (dq) squirrel-cage model under Krause convention "
                "with balanced 50 Hz sinusoidal voltage supply and constant 15.0 N m load torque."
            ),
            why_classified=(
                "All three phase currents are balanced 50 Hz sinusoids with negligible "
                "negative-sequence component (< 1e-7 A) and negligible torque ripple. "
                "The motor operates stably at nominal steady-state slip."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="current_unbalance_pct",
                    observed_value=1.433e-6,
                    baseline_value=1.433e-6,
                    unit="%",
                    diagnostic_role="normal_baseline",
                    interpretation="Phase currents are symmetric; unbalance is at numerical zero.",
                ),
                MetricEvidence(
                    metric_name="negative_sequence_a",
                    observed_value=4.651e-8,
                    baseline_value=4.651e-8,
                    unit="A",
                    diagnostic_role="normal_baseline",
                    interpretation="Absence of negative-sequence current confirms electrical symmetry.",
                ),
                MetricEvidence(
                    metric_name="torque_ripple_nm",
                    observed_value=6.265e-6,
                    baseline_value=6.265e-6,
                    unit="N m",
                    diagnostic_role="normal_baseline",
                    interpretation="Steady-state electromagnetic torque is smooth with near-zero ripple.",
                ),
            ],
            discrimination_vs_confounders=[
                "Serves as the unperturbed reference baseline against which all load steps, supply asymmetries, and internal fault proxies are evaluated.",
            ],
            standing_limitations=[
                "Deterministic ODE simulation with linear magnetics, sinusoidal MMF, and uniform air gap.",
                "Parameters are from a published literature example, not measured on a physical motor.",
                "Ideal zero-noise floor: baseline ripple and unbalance floors are numerical artifacts.",
                "No experimental validation.",
            ],
        )

    if condition_id in ("fault-01", "fault_01"):
        neg_seq_fault = metrics.get("negative_sequence_fault_a")
        neg_seq_h = metrics.get("negative_sequence_healthy_a")
        unbal_fault = metrics.get("current_unbalance_fault_pct")
        unbal_h = metrics.get("current_unbalance_healthy_pct")
        torque_fault = metrics.get("torque_ripple_fault_nm")
        torque_h = metrics.get("torque_ripple_healthy_nm")

        return ConditionExplanation(
            condition_id="fault-01",
            classification_category="internal-fault-proxy",
            classification_label=(
                "Physically modelled in the reduced-order plant (proxy for a "
                "high-resistance connection/joint; NOT a turn fault)"
            ),
            physical_mechanism=(
                "Per-phase stator resistances (1.5455, 1.405, 1.405) Ohm projected into the "
                "synchronous qd frame as a time-varying resistance matrix producing 100 Hz (2*f_s) ripple."
            ),
            why_classified=(
                "Internal stator resistance asymmetry creates nonzero fundamental negative-sequence "
                "current and current unbalance while the terminal supply voltage remains perfectly balanced (VUF = 0%)."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="negative_sequence_fault_a",
                    observed_value=neg_seq_fault,
                    baseline_value=neg_seq_h,
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=(
                        f"Negative-sequence current rises from baseline {_fmt_val(neg_seq_h)} A to "
                        f"{_fmt_val(neg_seq_fault)} A due to internal phase impedance asymmetry."
                    ),
                ),
                MetricEvidence(
                    metric_name="current_unbalance_fault_pct",
                    observed_value=unbal_fault,
                    baseline_value=unbal_h,
                    unit="%",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Current unbalance increases to {_fmt_val(unbal_fault)}% across the stator phases.",
                ),
                MetricEvidence(
                    metric_name="torque_ripple_fault_nm",
                    observed_value=torque_fault,
                    baseline_value=torque_h,
                    unit="N m",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Interaction between positive and negative sequence fields produces 100 Hz torque ripple of {_fmt_val(torque_fault)} N m.",
                ),
                MetricEvidence(
                    metric_name="supply_voltage_unbalance_pct",
                    observed_value=0.0,
                    baseline_value=0.0,
                    unit="%",
                    diagnostic_role="confounder_check",
                    interpretation="Supply voltage is balanced (VUF = 0%), distinguishing this internal stator asymmetry from an external supply unbalance.",
                ),
            ],
            discrimination_vs_confounders=[
                "Distinguished from Condition 02 (Load): Load increases current magnitude but maintains balanced phases (I2 remains near zero).",
                "Distinguished from Fault 03 (Supply unbalance): Fault 01 has balanced terminal voltage (VUF = 0%), whereas Fault 03 exhibits external supply voltage unbalance (VUF ~ 3.45%). Both produce I2, requiring terminal voltage measurement to separate.",
                "Distinguished from Fault 05 (Rotor asymmetry): Fault 01 produces fundamental 50 Hz negative sequence (100 Hz ripple); Fault 05 produces dynamic modulation sidebands at f_s(1 +/- 2s).",
            ],
            standing_limitations=[
                "Models an external high-resistance joint or terminal connection, NOT an inter-turn stator winding short.",
                "No thermal dynamics or localized hot-spot progression modelled.",
                "Star connection with isolated neutral; zero-sequence currents cannot circulate.",
                "No experimental validation.",
            ],
        )

    if condition_id in ("condition-02", "fault-02", "fault_02"):
        slip_c = summary_data.get("condition_final_slip")
        slip_h = summary_data.get("healthy_final_slip")
        iqs_c = summary_data.get("condition_late_mean_i_qs_a")
        iqs_h = summary_data.get("healthy_late_mean_i_qs_a")
        speed_diff = summary_data.get("final_speed_difference_rpm")

        return ConditionExplanation(
            condition_id="condition-02",
            classification_category="operating-condition",
            classification_label=(
                "Operating-condition change, NOT an internal motor fault "
                "(confounder for current-magnitude and slip features)"
            ),
            physical_mechanism=(
                "Constant mechanical load torque increased by +50% (15.0 N m to 22.5 N m). "
                "Motor parameters, differential equations, and supply voltage remain completely healthy."
            ),
            why_classified=(
                "Rotor slip and q-axis stator current increase proportionally to satisfy the higher load torque, "
                "while current balance is preserved and negative-sequence current remains near numerical zero."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="condition_final_slip",
                    observed_value=slip_c,
                    baseline_value=slip_h,
                    unit="p.u.",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Rotor slip increases from {_fmt_val(slip_h)} to {_fmt_val(slip_c)} to generate the required electromagnetic torque.",
                ),
                MetricEvidence(
                    metric_name="condition_late_mean_i_qs_a",
                    observed_value=iqs_c,
                    baseline_value=iqs_h,
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Torque-producing q-axis stator current rises from {_fmt_val(iqs_h)} A to {_fmt_val(iqs_c)} A.",
                ),
                MetricEvidence(
                    metric_name="final_speed_difference_rpm",
                    observed_value=speed_diff,
                    baseline_value=0.0,
                    unit="rpm",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Rotor speed decreases by {abs(speed_diff):.2f} rpm along the torque-speed curve.",
                ),
                MetricEvidence(
                    metric_name="current_unbalance_pct",
                    observed_value=0.0,
                    baseline_value=0.0,
                    unit="%",
                    diagnostic_role="confounder_check",
                    interpretation="Phase currents remain balanced by construction; no asymmetry or negative sequence is introduced.",
                ),
            ],
            discrimination_vs_confounders=[
                "Acts as a key diagnostic confounder: elevated current magnitude and slip alone cannot diagnose a winding or rotor fault, because benign mechanical load variations produce similar increases.",
                "Distinguished from Fault 01 and Fault 03: Condition 02 has zero current unbalance and zero negative-sequence current.",
                "Distinguished from Fault 05: Condition 02 produces no sideband peaks around the fundamental current spectrum.",
            ],
            standing_limitations=[
                "Represents a constant mechanical torque load step; speed-dependent (fan/pump) load characteristics are out of scope.",
                "Demonstrates that current magnitude alone is an ambiguous condition-monitoring metric.",
                "No experimental validation.",
            ],
        )

    if condition_id in ("fault-03", "fault_03"):
        vuf_fault = metrics.get("supply_voltage_unbalance_unbalance_pct")
        vuf_h = metrics.get("supply_voltage_unbalance_healthy_pct")
        neg_seq_fault = metrics.get("negative_sequence_unbalance_a")
        neg_seq_h = metrics.get("negative_sequence_healthy_a")
        unbal_fault = metrics.get("current_unbalance_unbalance_pct")
        torque_fault = metrics.get("torque_ripple_unbalance_nm")

        return ConditionExplanation(
            condition_id="fault-03",
            classification_category="supply-confounder",
            classification_label=(
                "Supply-side confounder, physically modelled supply; motor "
                "equations and parameters unchanged (NOT winding damage)"
            ),
            physical_mechanism=(
                "Phase C supply voltage reduced to 0.9 p.u., generating an external "
                "negative-sequence voltage (VUF ~ 3.45%) applied to a fully healthy motor."
            ),
            why_classified=(
                "Negative-sequence voltage drives severe negative-sequence currents and high current unbalance "
                "through the low negative-sequence impedance of the healthy induction motor."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="supply_voltage_unbalance_unbalance_pct",
                    observed_value=vuf_fault,
                    baseline_value=vuf_h,
                    unit="%",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"External voltage unbalance factor (VUF) is {_fmt_val(vuf_fault)}%, demonstrating supply-side asymmetry.",
                ),
                MetricEvidence(
                    metric_name="negative_sequence_unbalance_a",
                    observed_value=neg_seq_fault,
                    baseline_value=neg_seq_h,
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=(
                        f"Large negative-sequence current ({_fmt_val(neg_seq_fault)} A) is drawn by the healthy motor "
                        "due to its low negative-sequence impedance (~ 1 / (2 - s))."
                    ),
                ),
                MetricEvidence(
                    metric_name="current_unbalance_unbalance_pct",
                    observed_value=unbal_fault,
                    baseline_value=metrics.get("current_unbalance_healthy_pct"),
                    unit="%",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Current unbalance reaches {_fmt_val(unbal_fault)}%, disproportionately higher than the 3.45% voltage unbalance.",
                ),
                MetricEvidence(
                    metric_name="torque_ripple_unbalance_nm",
                    observed_value=torque_fault,
                    baseline_value=metrics.get("torque_ripple_healthy_nm"),
                    unit="N m",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Counter-rotating negative-sequence field creates 100 Hz torque ripple of {_fmt_val(torque_fault)} N m.",
                ),
            ],
            discrimination_vs_confounders=[
                "Mimics the current-unbalance and negative-sequence symptoms of stator resistance fault (Fault 01).",
                "Critical diagnostic distinction: Fault 03 is conclusively separated from Fault 01 only by measuring supply terminal voltages (VUF > 0 in Fault 03 vs VUF ~ 0 in Fault 01). Current-only monitoring cannot reliably separate them.",
                "Motor parameters are completely healthy; remediating the grid supply restores fully balanced operation without motor maintenance.",
            ],
            standing_limitations=[
                "Static phase-amplitude reduction only (no phase angle displacement, harmonic distortion, or voltage sag transients).",
                "Confirms that current-based fault indicators must be cross-checked against supply voltage quality.",
                "No experimental validation.",
            ],
        )

    if condition_id in ("fault-04", "fault_04"):
        bpfo = metrics.get("bpfo_hz")
        env_fault = metrics.get("envelope_bpfo_fault")
        env_h = metrics.get("envelope_bpfo_healthy")
        env_2x = metrics.get("envelope_bpfo_2x_fault")
        identical = summary_data.get("electrical_traces_identical_to_healthy", True)

        return ConditionExplanation(
            condition_id="fault-04",
            classification_category="vibration-signature",
            classification_label=(
                "Simulation-only vibration-channel signature (bearing "
                "kinematics + assumed resonance); electrical plant bit-for-bit "
                "healthy — NOT a motor-model fault, NOT an MCSA bearing claim"
            ),
            physical_mechanism=(
                "Synthetic impact impulse train generated at outer-race ball pass frequency "
                f"BPFO = (Nb/2) * f_r * (1 - (Bd/Pd)*cos(phi)) (~ {bpfo:.2f} Hz), exciting an assumed "
                "2.0 kHz structural resonance. Injected solely into an auxiliary vibration channel."
            ),
            why_classified=(
                "Envelope demodulation of the simulated vibration signal reveals sharp spectral peaks at BPFO "
                "and its second harmonic, while the electrical motor model remains bit-for-bit identical to healthy."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="envelope_bpfo_fault",
                    observed_value=env_fault,
                    baseline_value=env_h,
                    unit="m/s^2",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Vibration envelope spectrum exhibits a clear peak of {_fmt_val(env_fault)} m/s^2 at BPFO ({_fmt_val(bpfo)} Hz).",
                ),
                MetricEvidence(
                    metric_name="envelope_bpfo_2x_fault",
                    observed_value=env_2x,
                    baseline_value=metrics.get("envelope_bpfo_2x_healthy"),
                    unit="m/s^2",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Second harmonic peak of {_fmt_val(env_2x)} m/s^2 is observed at 2 x BPFO.",
                ),
                MetricEvidence(
                    metric_name="electrical_traces_identical_to_healthy",
                    observed_value="true" if identical else "false",
                    baseline_value="true",
                    unit=None,
                    diagnostic_role="unaffected_channel",
                    interpretation="Electrical traces (i_abc, v_abc, speed, torque) are bit-for-bit identical to healthy baseline.",
                ),
            ],
            discrimination_vs_confounders=[
                "Independent channel signature: alters only the auxiliary vibration channel, leaving electrical MCSA channels completely unaffected.",
                "Cannot be detected or confused with stator unbalance, load variations, or rotor asymmetry in electrical signals.",
            ],
            standing_limitations=[
                "Simulation-only kinematic signature; no bearing physical dynamics or load transmission are coupled into the motor ODEs.",
                "Healthy vibration baseline is identically zero by construction (no background machine vibration, white noise, or structural resonances).",
                "Does NOT model bearing fault signatures in stator currents (MCSA bearing detection is explicitly out of scope).",
                "Vibration amplitude is an arbitrary literature scaling; no severity calibration or experimental validation.",
            ],
        )

    if condition_id in ("fault-05", "fault_05"):
        lo_f = metrics.get("lower_sideband_fault_a")
        lo_h = metrics.get("lower_sideband_healthy_a")
        hi_f = metrics.get("upper_sideband_fault_a")
        hi_h = metrics.get("upper_sideband_healthy_a")
        lo_hz = metrics.get("lower_sideband_hz")
        hi_hz = metrics.get("upper_sideband_hz")
        mod_tone = metrics.get("sync_modulation_fault_a")

        return ConditionExplanation(
            condition_id="fault-05",
            classification_category="proxy-fault",
            classification_label=(
                "Simulation-only proxy for broken-bar-related behaviour "
                "(rotor-frame axis resistance split); NOT bar-resolved, NOT "
                "severity-calibrated, NOT a real-machine diagnosis"
            ),
            physical_mechanism=(
                "Rotor resistance split into orthogonal axes R_rq = R_r*(1 + delta) and R_rd = R_r*(1 - delta) "
                "with severity delta = 0.10. Projected into synchronous frame at modulation frequency 2*s_ref*f_s (~ 2.38 Hz)."
            ),
            why_classified=(
                "Modulation produces characteristic stator current MCSA sidebands at f_s*(1 - 2s) and f_s*(1 + 2s), "
                "and a synchronous-frame modulation tone at 2*s*f_s, while fundamental 50 Hz negative sequence remains negligible."
            ),
            supporting_evidence=[
                MetricEvidence(
                    metric_name="lower_sideband_fault_a",
                    observed_value=lo_f,
                    baseline_value=lo_h,
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=(
                        f"Lower sideband peak rises from baseline {_fmt_val(lo_h)} A to {_fmt_val(lo_f)} A "
                        f"at {_fmt_val(lo_hz)} Hz [f_s(1 - 2s)]."
                    ),
                ),
                MetricEvidence(
                    metric_name="upper_sideband_fault_a",
                    observed_value=hi_f,
                    baseline_value=hi_h,
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=(
                        f"Upper sideband peak rises from baseline {_fmt_val(hi_h)} A to {_fmt_val(hi_f)} A "
                        f"at {_fmt_val(hi_hz)} Hz [f_s(1 + 2s)]."
                    ),
                ),
                MetricEvidence(
                    metric_name="sync_modulation_fault_a",
                    observed_value=mod_tone,
                    baseline_value=metrics.get("sync_modulation_healthy_a"),
                    unit="A",
                    diagnostic_role="primary_discriminant",
                    interpretation=f"Synchronous-frame current modulation tone appears at 2*s*f_s with amplitude {_fmt_val(mod_tone)} A.",
                ),
                MetricEvidence(
                    metric_name="fundamental_negative_sequence_check",
                    observed_value="< 0.005 A",
                    baseline_value="< 1e-7 A",
                    unit="A",
                    diagnostic_role="confounder_check",
                    interpretation="50 Hz fundamental negative sequence remains near zero, distinguishing rotor modulation from stator resistance asymmetry.",
                ),
            ],
            discrimination_vs_confounders=[
                "Distinguished from Fault 01 (Stator R): Fault 05 produces dynamic slip-dependent sidebands at f_s(1 +/- 2s); Fault 01 produces fundamental 50 Hz negative-sequence current and 100 Hz torque ripple.",
                "Distinguished from Fault 03 (Supply unbalance): Supply unbalance causes massive fundamental negative-sequence current (1.91 A), whereas Fault 05 fundamental negative-sequence is negligible (< 0.005 A).",
                "Distinguished from Condition 02 (Load): Higher load increases current magnitude uniformly and shifts sideband locations by altering slip, but does not generate sideband peaks by itself.",
            ],
            standing_limitations=[
                "Two-axis resistance split is a macro-level behavioral proxy, NOT a bar-resolved discrete circuit model.",
                "Severity delta = 0.10 is an arbitrary simulation knob, not calibrated to an exact number of physically broken bars.",
                "Modulation uses constant steady-state reference slip, not instantaneous slip angle integration during DOL startup.",
                "Does not model the mean resistance increase or cage heating of real broken bars; no experimental validation.",
            ],
        )

    # Fallback for unknown condition id
    return ConditionExplanation(
        condition_id=condition_id,
        classification_category="unknown",
        classification_label="Unclassified condition",
        physical_mechanism="No physical mechanism documented.",
        why_classified="Condition has no documented diagnostic rules.",
        supporting_evidence=[],
        discrimination_vs_confounders=[],
        standing_limitations=["Undefined condition record."],
    )
