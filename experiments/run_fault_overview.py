"""Build the unified simulated condition overview (JSON + table + figure).

Reads the per-study summaries and trace archives under ``results/`` (run
the healthy and per-fault experiment scripts first) and writes:

- ``results/fault_overview_summary.json``
- ``results/fault_overview_table.md``
- ``results/fault_overview_comparison.png``

All artifacts are simulated, literature-example parameters only, and no
diagnostic, classifier, GUI, or predictive-maintenance claim is implied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imcm.reporting.fault_summary import write_overview


def main() -> None:
    record = write_overview(ROOT / "results")
    checks = record["consistency_checks"]
    failed = [c for c in checks if c["status"] != "pass"]
    print(f"Consistency checks: {len(checks) - len(failed)}/{len(checks)} passed")
    for check in failed:
        print(f"  FAIL {check['name']}: {check['detail']}")
    for artifact in record["artifacts"].values():
        print(f"Wrote {artifact}")
    print(f"all_checks_pass: {record['all_checks_pass']}")


if __name__ == "__main__":
    main()