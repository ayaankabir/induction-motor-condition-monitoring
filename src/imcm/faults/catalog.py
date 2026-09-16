"""Named fault cases for the controlled fault studies.

Implemented entries carry a simulated, documented representation; the honesty
note states what each representation can and cannot claim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannedFault:
    id: str
    name: str
    mechanism: str
    honesty_note: str
    phase: int


# First milestone: healthy plant only. These entries are a later-phase catalog.

PLANNED_FAULTS: tuple[PlannedFault, ...] = (
    PlannedFault(
        id="voltage_unbalance",
        name="Supply voltage unbalance",
        mechanism="Unequal phase voltages or added negative-sequence voltage; motor equations unchanged.",
        honesty_note="This is a supply confounder, not a winding fault.",
        phase=3,
    ),
    PlannedFault(
        id="stator_resistance_unbalance",
        name="Stator resistance unbalance",
        mechanism="Unequal R_sa, R_sb, R_sc in an abc-stator extension of the plant.",
        honesty_note="Models a high-resistance connection, not an inter-turn short.",
        phase=4,
    ),
    PlannedFault(
        id="rotor_asymmetry_proxy",
        name="Rotor electrical asymmetry (broken-bar proxy) — Fault 05 implemented",
        mechanism=(
            "Rotor-frame axis resistance split R_r(1 +/- severity) on the "
            "two-axis cage, projected into the synchronous frame where it "
            "modulates at 2*s*f_s, producing stator-current components near "
            "f_s(1 -/+ 2s). Mean rotor resistance is preserved."
        ),
        honesty_note=(
            "Simulation-only proxy for broken-bar-related behaviour; may "
            "produce sidebands near f_s(1±2s). Not a physically complete or "
            "bar-resolved electromagnetic model, not severity-calibrated, not "
            "experimentally validated, and not a real-machine diagnosis."
        ),
        phase=5,
    ),
    PlannedFault(
        id="bearing_outer_race_bpfo",
        name="Bearing outer-race fault (BPFO vibration signature)",
        mechanism=(
            "Simulated vibration channel with an impulse train at "
            "BPFO = (Nb/2) f_r (1 - (Bd/Pd) cos(phi)); motor ODEs unchanged."
        ),
        honesty_note=(
            "Simulated condition-monitoring signature on a literature-example "
            "bearing geometry. Not a measured bearing, not a stator-current "
            "signature, and not a confirmed real-machine diagnosis."
        ),
        phase=6,
    ),
)
