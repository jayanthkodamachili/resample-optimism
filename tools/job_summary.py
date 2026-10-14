"""Print the table a reviewer should actually see, for the CI job summary.

The summary leads with E4, the experiment whose answer is least flattering to the fix this
repository teaches. Putting the headline optimism number here instead would make the job
summary an advertisement.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E4 = ROOT / "docs" / "experiments" / "e4_was_it_worth_it.json"
METRICS = ROOT / "docs" / "metrics.json"


def main() -> int:
    if not E4.exists() or not METRICS.exists():
        print("### receipts\n\nexperiment output missing, nothing to summarise.")
        return 0

    e4 = json.loads(E4.read_text())
    metrics = json.loads(METRICS.read_text())["metrics"]

    print("### Was the resampling worth doing at all?\n")
    print(
        "Correct, leak-free resampling measured against no resampling, on a holdout "
        "neither has seen.\n"
    )
    print("| metric | cells improved | runs that got worse | grand mean |")
    print("| --- | --- | --- | --- |")
    for name, sweep in e4["sweeps"].items():
        print(
            f"| {name} | {sweep['cells_positive']}/{sweep['cells_total']} "
            f"| {sweep['runs_negative']}/{sweep['runs_total']} "
            f"| {sweep['grand_mean_benefit']:+.4f} |"
        )
    print()
    print(
        f"The two metrics disagree on the sign. The average precision loss is "
        f"{metrics['e4_loss_gain_ratio']} times the size of the ROC-AUC gain.\n"
    )
    print(
        f"Leaked protocol optimism: {metrics['leaked_optimism_mean']:+.4f} "
        f"ROC-AUC over {metrics['identical_model_seeds']} seeds, "
        f"from a model identical to the one the corrected protocol ships."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
