"""Named fault cases planned for later phases.

These identifiers are documentation, not simulated defects.
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
        name="Rotor electrical asymmetry (broken-bar proxy)",
        mechanism="Unequal R_qr, R_dr or 2*omega_r resistance modulation on the two-axis cage.",
        honesty_note="May produce sidebands near f_s(1±2s). It is not a bar-resolved cage model.",
        phase=5,
    ),
)
