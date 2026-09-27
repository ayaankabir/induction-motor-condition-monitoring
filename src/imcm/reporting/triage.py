"""Deterministic, source-data-only rule-based triage.

This module evaluates stored summary records. It never runs a simulation,
calculates a signal feature, or assigns a probabilistic or aggregate score.
Missing or invalid required metrics produce a safe non-classified result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

TriageStatus = Literal["classified", "ambiguous", "indeterminate", "not_evaluable"]
RuleStatus = Literal["matched", "not_matched", "not_evaluable"]


@dataclass(frozen=True)
class TriagePredicate:
    """One transparent comparison made by a rule."""

    metric_name: str
    operator: str
    expected: float | int | bool | None
    observed: float | int | bool | None
    unit: str | None = None
    passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriageConfounderCheck:
    """A required check against a competing explanation."""

    metric_name: str
    operator: str
    expected: float | int | bool | None
    observed: float | int | bool | None
    unit: str | None = None
    passed: bool | None = None
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriageMetricEvidence:
    metric_name: str
    observed_value: float | int | bool | str | None
    baseline_value: float | int | bool | str | None
    unit: str | None
    diagnostic_role: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriageRuleTrace:
    rule_id: str
    condition_id: str
    status: RuleStatus
    why: str
    predicates: tuple[TriagePredicate, ...]
    confounder_checks: tuple[TriageConfounderCheck, ...] = ()
    supporting_evidence: tuple[TriageMetricEvidence, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "condition_id": self.condition_id,
            "status": self.status,
            "why": self.why,
            "predicates": [item.to_dict() for item in self.predicates],
            "confounder_checks": [item.to_dict() for item in self.confounder_checks],
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
        }


@dataclass(frozen=True)
class TriageResult:
    schema_version: str
    status: TriageStatus
    condition_id: str | None
    classification_category: str | None
    classification_label: str | None
    triggered_rule_ids: tuple[str, ...]
    matched_rule_count: int
    why: str
    rules: tuple[TriageRuleTrace, ...]
    supporting_evidence: tuple[TriageMetricEvidence, ...]
    missing_metrics: tuple[str, ...]
    ambiguities: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "condition_id": self.condition_id,
            "classification_category": self.classification_category,
            "classification_label": self.classification_label,
            "triggered_rule_ids": list(self.triggered_rule_ids),
            "matched_rule_count": self.matched_rule_count,
            "why": self.why,
            "rules": [item.to_dict() for item in self.rules],
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "missing_metrics": list(self.missing_metrics),
            "ambiguities": list(self.ambiguities),
            "limitations": list(self.limitations),
        }


# Canonical labels are intentionally descriptive and simulation-scoped.
RULE_LABELS: dict[str, tuple[str, str]] = {
    "healthy": ("baseline", "Healthy reference baseline — simulated"),
    "fault-01": ("internal-fault-proxy", "Internal stator-resistance proxy — simulated"),
    "condition-02": ("operating-condition", "Increased-load operating condition — simulated"),
    "fault-03": ("supply-confounder", "Supply-voltage unbalance confounder — simulated"),
    "fault-04": (
        "vibration-signature",
        "BPFO vibration-channel signature — simulation only",
    ),
    "fault-05": (
        "proxy-fault",
        "Rotor-asymmetry/broken-bar proxy — simulation only",
    ),
}

RULE_WHY: dict[str, str] = {
    "healthy": "The complete stored reference profile has near-zero sequence/unbalance metrics and positive slip.",
    "fault-01": "The stored internal-resistance comparison has elevated negative-sequence current while terminal voltage is balanced.",
    "condition-02": "The stored load comparison has higher slip and q-axis current with lower speed and balanced electrical channels.",
    "fault-03": "The stored supply comparison has external voltage unbalance and a large negative-sequence current.",
    "fault-04": "The stored bearing study has an isolated vibration-envelope signature while electrical traces are identical to healthy.",
    "fault-05": "The stored rotor-asymmetry study has slip-dependent sidebands and a modulation tone with negligible fundamental negative sequence.",
}

RULE_LIMITATIONS: dict[str, tuple[str, ...]] = {
    "healthy": (
        "Deterministic simulation with an idealised numerical floor; not a real-machine health claim.",
    ),
    "fault-01": (
        "Proxy for a high-resistance connection/joint, not an inter-turn fault or calibrated diagnosis.",
    ),
    "condition-02": (
        "Controlled load-step contrast; current magnitude and slip are not unique fault indicators.",
        "Load torque alone is not a fault label; the balance channels are required for this rule.",
    ),
    "fault-03": (
        "Supply-side confounder; motor equations are healthy and current-only monitoring cannot separate it from Fault 01.",
    ),
    "fault-04": (
        "Synthetic vibration-channel signature with arbitrary amplitude; it is not an MCSA result and does not quantify condition extent.",
    ),
    "fault-05": (
        "Two-axis rotor-resistance proxy, not a bar-resolved broken-bar model; the stored modulation knob is not a real-machine calibration.",
        "The healthy negative-sequence comparison is required to separate this signature from stator/supply asymmetry.",
    ),
}

_MISSING = object()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_value(value: Any, expected_type: type | tuple[type, ...]) -> bool:
    if value is _MISSING or value is None:
        return False
    if expected_type is float or expected_type is int:
        return _finite_number(value)
    if expected_type is bool:
        return isinstance(value, bool)
    return isinstance(value, expected_type)


def _read(mapping: dict[str, Any], key: str) -> Any:
    return mapping.get(key, _MISSING)


def _display(value: Any) -> float | int | bool | str | None:
    if value is _MISSING:
        return None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _compare(
    observed: Any, operator: str, expected: float | int | bool | None
) -> bool | None:
    if expected is None or not _valid_value(observed, type(expected)):
        return None
    try:
        if operator == ">":
            return observed > expected
        if operator == ">=":
            return observed >= expected
        if operator == "<":
            return observed < expected
        if operator == "<=":
            return observed <= expected
        if operator == "==":
            return observed == expected
        if operator == "!=":
            return observed != expected
    except (TypeError, ValueError):
        return None
    return None


def _predicate(
    values: dict[str, Any],
    key: str,
    operator: str,
    expected: float | int | bool | None,
    unit: str | None,
) -> TriagePredicate:
    observed = _read(values, key)
    return TriagePredicate(
        metric_name=key,
        operator=operator,
        expected=expected,
        observed=_display(observed),
        unit=unit,
        passed=_compare(observed, operator, expected),
    )


def _check(
    values: dict[str, Any],
    key: str,
    operator: str,
    expected: float | int | bool | None,
    unit: str | None,
    interpretation: str,
) -> TriageConfounderCheck:
    observed = _read(values, key)
    return TriageConfounderCheck(
        metric_name=key,
        operator=operator,
        expected=expected,
        observed=_display(observed),
        unit=unit,
        passed=_compare(observed, operator, expected),
        interpretation=interpretation,
    )


def _evidence(
    key: str,
    observed: Any,
    baseline: Any,
    unit: str | None,
    role: str,
    interpretation: str,
) -> TriageMetricEvidence:
    return TriageMetricEvidence(
        metric_name=key,
        observed_value=_display(observed),
        baseline_value=_display(baseline),
        unit=unit,
        diagnostic_role=role,
        interpretation=interpretation,
    )


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    raw = summary.get("metrics")
    return dict(raw) if isinstance(raw, dict) else {}


def _summary_values(summary: dict[str, Any]) -> dict[str, Any]:
    """Expose only already-stored values under canonical rule names."""

    m = _metrics(summary)
    values: dict[str, Any] = {}

    def add(canonical: str, *keys: str) -> None:
        for key in keys:
            if key in m:
                values[canonical] = m[key]
                return
        for key in keys:
            if key in summary:
                values[canonical] = summary[key]
                return
        values[canonical] = _MISSING

    add("current_unbalance_pct", "current_unbalance_pct", "current_unbalance_fault_pct", "current_unbalance_unbalance_pct")
    add("negative_sequence_a", "negative_sequence_a", "negative_sequence_fault_a", "negative_sequence_unbalance_a")
    add("supply_voltage_unbalance_pct", "supply_voltage_unbalance_pct", "supply_voltage_unbalance_unbalance_pct", "fault_supply_voltage_unbalance_pct")
    add("torque_ripple_nm", "torque_ripple_nm", "torque_ripple_fault_nm", "torque_ripple_unbalance_nm")
    add("final_slip", "final_slip", "condition_final_slip")
    add("late_mean_i_qs_a", "late_mean_i_qs_a", "condition_late_mean_i_qs_a")
    add("final_speed_difference_rpm", "final_speed_difference_rpm")
    add("electrical_traces_identical_to_healthy", "electrical_traces_identical_to_healthy")
    add("bpfo_hz", "bpfo_hz")
    add("bpfo_peak_fault_hz", "bpfo_peak_fault_hz")
    add("envelope_bpfo_healthy", "envelope_bpfo_healthy")
    add("envelope_bpfo_fault", "envelope_bpfo_fault")
    add("envelope_bpfo_2x_healthy", "envelope_bpfo_2x_healthy")
    add("envelope_bpfo_2x_fault", "envelope_bpfo_2x_fault")
    add("fundamental_hz", "fundamental_hz")
    add("reference_slip", "reference_slip")
    add("lower_sideband_hz", "lower_sideband_hz")
    add("upper_sideband_hz", "upper_sideband_hz")
    add("lower_sideband_healthy_a", "lower_sideband_healthy_a")
    add("lower_sideband_fault_a", "lower_sideband_fault_a")
    add("upper_sideband_healthy_a", "upper_sideband_healthy_a")
    add("upper_sideband_fault_a", "upper_sideband_fault_a")
    add("sync_modulation_healthy_a", "sync_modulation_healthy_a")
    add("sync_modulation_fault_a", "sync_modulation_fault_a")

    # These are comparison values already stored with the source summary.
    # They are not derived by this module.
    add("_healthy_final_slip", "healthy_final_slip")
    add("_healthy_late_mean_i_qs_a", "healthy_late_mean_i_qs_a")
    # A healthy baseline is a reference profile, not a catch-all for any
    # balanced-looking record.
    values["_reference_profile"] = summary.get("label") == "healthy"
    return values


def _provenance_valid(summary: dict[str, Any]) -> bool:
    if summary.get("parameter_provenance") != "literature_example":
        return False
    if summary.get("provenance") not in (None, "simulated"):
        return False
    if summary.get("experimental_validation") is True:
        return False
    return True


def _rule_trace(
    rule_id: str,
    condition_id: str,
    predicates: list[TriagePredicate],
    checks: list[TriageConfounderCheck],
    evidence: list[TriageMetricEvidence],
) -> TriageRuleTrace:
    statuses = [item.passed for item in predicates]
    if any(item is None for item in statuses):
        status: RuleStatus = "not_evaluable"
    elif all(item is True for item in statuses):
        status = "matched"
    else:
        status = "not_matched"
    if any(item.passed is None for item in checks):
        status = "not_evaluable"
    elif all(item.passed is True for item in checks):
        pass
    else:
        status = "not_matched"
    return TriageRuleTrace(
        rule_id=rule_id,
        condition_id=condition_id,
        status=status,
        why=RULE_WHY[condition_id],
        predicates=tuple(predicates),
        confounder_checks=tuple(checks),
        supporting_evidence=tuple(evidence),
    )


def _r00(values: dict[str, Any]) -> TriageRuleTrace:
    predicates = [
        _predicate(values, "current_unbalance_pct", "<", 0.01, "%"),
        _predicate(values, "negative_sequence_a", "<", 1e-5, "A"),
        _predicate(values, "final_slip", ">", 0.0, "p.u."),
    ]
    return _rule_trace("R00_HEALTHY_BASELINE", "healthy", predicates, [], [
        _evidence("current_unbalance_pct", values.get("current_unbalance_pct", _MISSING), None, "%", "normal_baseline", "Stored current unbalance is the healthy reference value."),
        _evidence("negative_sequence_a", values.get("negative_sequence_a", _MISSING), None, "A", "normal_baseline", "Stored fundamental negative-sequence current is the healthy reference value."),
    ])


def _r01(values: dict[str, Any]) -> TriageRuleTrace:
    predicates = [
        _predicate(values, "negative_sequence_a", ">", 0.01, "A"),
        _predicate(values, "supply_voltage_unbalance_pct", "<", 1e-6, "%"),
    ]
    checks = [
        _check(values, "supply_voltage_unbalance_pct", "<", 1e-6, "%", "Terminal voltage is balanced, separating internal resistance asymmetry from supply unbalance."),
    ]
    return _rule_trace("R01_STATOR_RESISTANCE_ASYMMETRY", "fault-01", predicates, checks, [
        _evidence("negative_sequence_a", values.get("negative_sequence_a", _MISSING), None, "A", "primary_discriminant", "Elevated negative-sequence current is the stored primary signature."),
        _evidence("current_unbalance_pct", values.get("current_unbalance_pct", _MISSING), None, "%", "primary_discriminant", "Current unbalance supports phase asymmetry."),
    ])


def _r02(values: dict[str, Any]) -> TriageRuleTrace:
    healthy_slip = values.get("_healthy_final_slip", _MISSING)
    healthy_iqs = values.get("_healthy_late_mean_i_qs_a", _MISSING)
    predicates = [
        _predicate(
            values,
            "final_slip",
            ">",
            healthy_slip if _valid_value(healthy_slip, float) else None,
            "p.u.",
        ),
        _predicate(
            values,
            "late_mean_i_qs_a",
            ">",
            healthy_iqs if _valid_value(healthy_iqs, float) else None,
            "A",
        ),
        _predicate(values, "final_speed_difference_rpm", "<", 0.0, "rpm"),
        _predicate(values, "current_unbalance_pct", "<", 0.01, "%"),
        _predicate(values, "negative_sequence_a", "<", 1e-5, "A"),
    ]
    checks = [
        _check(values, "current_unbalance_pct", "<", 0.01, "%", "Load change preserves phase balance."),
        _check(values, "negative_sequence_a", "<", 1e-5, "A", "Load change does not create fundamental negative sequence."),
    ]
    return _rule_trace("R02_LOAD_STEP", "condition-02", predicates, checks, [
        _evidence("final_slip", values.get("final_slip", _MISSING), values.get("_healthy_final_slip", _MISSING), "p.u.", "primary_discriminant", "Slip increases relative to the stored healthy reference."),
        _evidence("late_mean_i_qs_a", values.get("late_mean_i_qs_a", _MISSING), values.get("_healthy_late_mean_i_qs_a", _MISSING), "A", "primary_discriminant", "Torque-producing q-axis current increases relative to healthy."),
    ])


def _r03(values: dict[str, Any]) -> TriageRuleTrace:
    predicates = [
        _predicate(values, "supply_voltage_unbalance_pct", ">", 3.0, "%"),
        _predicate(values, "negative_sequence_a", ">", 0.5, "A"),
    ]
    checks = [
        _check(values, "supply_voltage_unbalance_pct", ">", 3.0, "%", "External terminal voltage unbalance distinguishes this condition from Fault 01."),
    ]
    return _rule_trace("R03_SUPPLY_UNBALANCE", "fault-03", predicates, checks, [
        _evidence("supply_voltage_unbalance_pct", values.get("supply_voltage_unbalance_pct", _MISSING), None, "%", "primary_discriminant", "Stored VUF is the external supply discriminator."),
        _evidence("negative_sequence_a", values.get("negative_sequence_a", _MISSING), None, "A", "primary_discriminant", "Large negative-sequence current is driven by the supply asymmetry."),
    ])


def _r04(values: dict[str, Any]) -> TriageRuleTrace:
    bpfo = values.get("bpfo_hz", _MISSING)
    peak = values.get("bpfo_peak_fault_hz", _MISSING)
    second_harmonic_healthy = values.get("envelope_bpfo_2x_healthy", _MISSING)
    predicates = [
        _predicate(values, "electrical_traces_identical_to_healthy", "==", True, None),
        _predicate(values, "envelope_bpfo_healthy", "==", 0.0, "m/s^2"),
        _predicate(values, "envelope_bpfo_fault", ">", 0.05, "m/s^2"),
        _predicate(
            values,
            "envelope_bpfo_2x_fault",
            ">",
            second_harmonic_healthy
            if _valid_value(second_harmonic_healthy, float)
            else None,
            "m/s^2",
        ),
    ]
    location_ok = _finite_number(bpfo) and _finite_number(peak) and abs(peak - bpfo) < 1.0
    predicates.append(TriagePredicate("bpfo_peak_error_hz", "<", 1.0, abs(peak - bpfo) if location_ok else None, "Hz", location_ok))
    checks = [_check(values, "electrical_traces_identical_to_healthy", "==", True, None, "Electrical plant is unchanged; the signature is isolated to vibration.")]
    return _rule_trace("R04_BEARING_VIBRATION_SIGNATURE", "fault-04", predicates, checks, [
        _evidence("envelope_bpfo_fault", values.get("envelope_bpfo_fault", _MISSING), values.get("envelope_bpfo_healthy", _MISSING), "m/s^2", "primary_discriminant", "BPFO envelope line is elevated above the zero healthy channel."),
        _evidence("envelope_bpfo_2x_fault", values.get("envelope_bpfo_2x_fault", _MISSING), values.get("envelope_bpfo_2x_healthy", _MISSING), "m/s^2", "primary_discriminant", "Second BPFO harmonic is elevated."),
    ])


def _r05(values: dict[str, Any]) -> TriageRuleTrace:
    fs = values.get("fundamental_hz", _MISSING)
    ref = values.get("reference_slip", _MISSING)
    lower = values.get("lower_sideband_hz", _MISSING)
    target = fs * (1.0 - 2.0 * ref) if _finite_number(fs) and _finite_number(ref) else _MISSING
    location_ok = _finite_number(target) and _finite_number(lower) and abs(lower - target) < 1e-6
    predicates = [
        _predicate(values, "negative_sequence_a", "<", 0.005, "A"),
        TriagePredicate("lower_sideband_location_error_hz", "<", 1e-6, abs(lower - target) if location_ok else None, "Hz", location_ok),
        TriagePredicate("lower_sideband_fault_to_healthy_ratio", ">", 100.0, (values.get("lower_sideband_fault_a", _MISSING) / values.get("lower_sideband_healthy_a", _MISSING)) if _finite_number(values.get("lower_sideband_fault_a", _MISSING)) and _finite_number(values.get("lower_sideband_healthy_a", _MISSING)) and values.get("lower_sideband_healthy_a", _MISSING) != 0 else None, "ratio", None if not (_finite_number(values.get("lower_sideband_fault_a", _MISSING)) and _finite_number(values.get("lower_sideband_healthy_a", _MISSING))) else (values.get("lower_sideband_fault_a", _MISSING) > 100.0 * values.get("lower_sideband_healthy_a", _MISSING))),
    ]
    upper_healthy = values.get("upper_sideband_healthy_a", _MISSING)
    modulation_healthy = values.get("sync_modulation_healthy_a", _MISSING)
    predicates.extend(
        [
            _predicate(
                values,
                "upper_sideband_fault_a",
                ">",
                upper_healthy if _valid_value(upper_healthy, float) else None,
                "A",
            ),
            _predicate(
                values,
                "sync_modulation_fault_a",
                ">",
                modulation_healthy
                if _valid_value(modulation_healthy, float)
                else None,
                "A",
            ),
        ]
    )
    checks = [_check(values, "negative_sequence_a", "<", 0.005, "A", "Fundamental negative sequence remains negligible, unlike stator/supply asymmetry.")]
    return _rule_trace("R05_ROTOR_ASYMMETRY_PROXY", "fault-05", predicates, checks, [
        _evidence("lower_sideband_fault_a", values.get("lower_sideband_fault_a", _MISSING), values.get("lower_sideband_healthy_a", _MISSING), "A", "primary_discriminant", "Lower slip-dependent sideband is elevated."),
        _evidence("upper_sideband_fault_a", values.get("upper_sideband_fault_a", _MISSING), values.get("upper_sideband_healthy_a", _MISSING), "A", "primary_discriminant", "Upper slip-dependent sideband is elevated."),
        _evidence("sync_modulation_fault_a", values.get("sync_modulation_fault_a", _MISSING), values.get("sync_modulation_healthy_a", _MISSING), "A", "primary_discriminant", "Synchronous-frame modulation tone is present."),
    ])


RULE_BUILDERS = (_r04, _r03, _r01, _r05, _r02, _r00)


def _missing_from_trace(trace: TriageRuleTrace) -> tuple[str, ...]:
    missing: list[str] = []
    for item in (*trace.predicates, *trace.confounder_checks):
        if item.passed is None and item.metric_name not in missing:
            missing.append(item.metric_name)
    return tuple(missing)


def _candidate_missing(trace: TriageRuleTrace) -> tuple[str, ...]:
    # Only report missing fields for a rule that has positive stored evidence;
    # otherwise every condition would report every unrelated channel as missing.
    if any(item.passed is True for item in trace.predicates):
        return _missing_from_trace(trace)
    return ()


def evaluate_triage(
    summary: dict[str, Any], healthy_baseline: dict[str, Any] | None = None
) -> TriageResult:
    """Evaluate every rule against one stored summary without deriving metrics.

    ``healthy_baseline`` is an already-loaded stored overview record. It is
    used only to fill the healthy reference profile; it never supplies a
    condition-specific missing discriminator.
    """

    if not isinstance(summary, dict):
        summary = {}
    values = _summary_values(summary)

    # The healthy summary stores operating quantities at the top level, while
    # the canonical sequence/unbalance values live in the existing overview
    # artifact. Only the explicitly healthy record receives this baseline.
    if healthy_baseline and summary.get("label") == "healthy":
        baseline = healthy_baseline.get("healthy_baseline")
        if isinstance(baseline, dict):
            for canonical, key in (
                ("current_unbalance_pct", "current_unbalance_pct"),
                ("negative_sequence_a", "negative_sequence_a"),
                ("final_slip", "final_slip"),
            ):
                if values.get(canonical, _MISSING) is _MISSING and key in baseline:
                    values[canonical] = baseline[key]
    traces = tuple(builder(values) for builder in RULE_BUILDERS)
    if not values.get("_reference_profile", False):
        traces = tuple(
            TriageRuleTrace(
                rule_id=trace.rule_id,
                condition_id=trace.condition_id,
                status="not_matched" if trace.status == "matched" else trace.status,
                why=trace.why,
                predicates=trace.predicates,
                confounder_checks=trace.confounder_checks,
                supporting_evidence=trace.supporting_evidence,
            )
            if trace.rule_id == "R00_HEALTHY_BASELINE"
            else trace
            for trace in traces
        )
    matched = tuple(trace for trace in traces if trace.status == "matched")
    candidate_missing = tuple(
        dict.fromkeys(
            item
            for trace in traces
            for item in _candidate_missing(trace)
        )
    )
    evidence = tuple(item for trace in traces for item in trace.supporting_evidence if item.observed_value is not None)
    ambiguities: list[str] = []

    if not _provenance_valid(summary):
        status: TriageStatus = "not_evaluable"
        matched = ()
        why = "Stored provenance is not a simulated literature-example record; no triage rule was applied."
        condition_id = None
        category = None
        label = None
    elif len(matched) > 1:
        status = "ambiguous"
        why = "Multiple complete deterministic rules matched; no condition was selected."
        condition_id = None
        category = None
        label = None
        ambiguities = [f"{trace.rule_id} ({trace.condition_id})" for trace in matched]
    elif len(matched) == 1:
        status = "classified"
        condition_id = matched[0].condition_id
        category, label = RULE_LABELS[condition_id]
        why = matched[0].why
    else:
        status = "indeterminate"
        why = "No unique complete deterministic rule matched; no condition was assigned."
        condition_id = None
        category = None
        label = None

    limitations = tuple(
        dict.fromkeys(
            item
            for trace in traces
            for item in RULE_LIMITATIONS[trace.condition_id]
        )
    )
    if status in {"indeterminate", "not_evaluable", "ambiguous"}:
        limitations = limitations + (
            "A non-classified result is intentionally not a diagnosis or a forecast.",
        )

    return TriageResult(
        schema_version="deterministic_rules_v1",
        status=status,
        condition_id=condition_id,
        classification_category=category,
        classification_label=label,
        triggered_rule_ids=tuple(trace.rule_id for trace in matched),
        matched_rule_count=len(matched),
        why=why,
        rules=traces,
        supporting_evidence=evidence,
        missing_metrics=candidate_missing,
        ambiguities=tuple(ambiguities),
        limitations=limitations,
    )