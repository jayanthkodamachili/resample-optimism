"""E4: having fixed the protocol, was the resampling worth doing at all?

This is the experiment the repository exists to be able to run. Once the leak is removed,
the honest question is whether correctly applied resampling beats not resampling, measured
on a holdout neither has touched. The literature is not kind to resampling under
threshold-free metrics, and the point here is to measure it rather than cite it.

The result decides whether the rest of the repository is a fix worth applying or a fix to a
step that should be deleted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from resample_optimism.config import ScenarioConfig  # noqa: E402
from resample_optimism.optimism import evaluate_all, resampling_benefit  # noqa: E402

SEEDS = (20260214, 11, 2027, 404, 90210)
MINORITY_RATES = (0.02, 0.05, 0.10, 0.20)
ESTIMATORS = ("forest", "logistic", "tree")
HEADLINE_METRIC = "average_precision"
METRICS = ("average_precision", "roc_auc")


def sweep(metric: str) -> dict:
    cells = []
    for estimator in ESTIMATORS:
        for rate in MINORITY_RATES:
            benefits = []
            for seed in SEEDS:
                config = ScenarioConfig(
                    minority_rate=rate, estimator=estimator, metric=metric, seed=seed
                )
                benefits.append(resampling_benefit(evaluate_all(config)))
            cells.append(
                {
                    "estimator": estimator,
                    "minority_rate": rate,
                    "benefit_mean": float(np.mean(benefits)),
                    "benefit_min": float(np.min(benefits)),
                    "benefit_max": float(np.max(benefits)),
                    "seeds_positive": int(sum(1 for b in benefits if b > 0)),
                    "seeds_total": len(SEEDS),
                }
            )

    runs_total = len(cells) * len(SEEDS)
    runs_positive = sum(c["seeds_positive"] for c in cells)
    return {
        "metric": metric,
        "cells": cells,
        "cells_total": len(cells),
        "cells_positive": sum(1 for c in cells if c["benefit_mean"] > 0),
        "runs_total": runs_total,
        "runs_positive": runs_positive,
        "runs_negative": runs_total - runs_positive,
        "best_cell": max(cells, key=lambda c: c["benefit_mean"]),
        "worst_cell": min(cells, key=lambda c: c["benefit_mean"]),
        "grand_mean_benefit": float(np.mean([c["benefit_mean"] for c in cells])),
    }


def run() -> dict:
    """Sweep both metrics, because they disagree and reporting one of them would be a choice.

    The first version of this experiment reported average precision alone and concluded that
    correct resampling never helps. Running ROC-AUC over the identical cells reverses the
    sign. Whichever metric a reader prefers, the honest headline is that the answer depends
    on the metric, so both sweeps are published side by side.
    """
    sweeps = {metric: sweep(metric) for metric in METRICS}
    headline = sweeps[HEADLINE_METRIC]
    return {
        "experiment": "e4_was_it_worth_it",
        "question": "Does correctly applied SMOTE beat no resampling on an untouched holdout?",
        "seeds": list(SEEDS),
        "headline_metric": HEADLINE_METRIC,
        "sweeps": sweeps,
        "metrics_disagree": (
            sweeps["average_precision"]["grand_mean_benefit"] > 0
        ) != (sweeps["roc_auc"]["grand_mean_benefit"] > 0),
        **{k: v for k, v in headline.items() if k != "cells"},
        "cells": headline["cells"],
    }


def main() -> None:
    payload = run()
    print("E4 honest gain from correct SMOTE against no resampling\n")
    for metric, result in payload["sweeps"].items():
        rows = [
            [
                f"{c['estimator']} @ {c['minority_rate']:.0%}",
                f"{c['benefit_mean']:+.4f}",
                f"{c['benefit_min']:+.4f}",
                f"{c['benefit_max']:+.4f}",
                f"{c['seeds_positive']}/{c['seeds_total']}",
            ]
            for c in result["cells"]
        ]
        print(f"metric: {metric}")
        print(table(["cell", "mean", "worst", "best", "seeds up"], rows, [20, 11, 11, 11, 11]))
        print(
            f"{result['cells_positive']}/{result['cells_total']} cells improved on average; "
            f"{result['runs_negative']}/{result['runs_total']} runs got worse; "
            f"grand mean {result['grand_mean_benefit']:+.4f}\n"
        )
    print(f"the two metrics disagree on sign: {payload['metrics_disagree']}")
    path = write("e4_was_it_worth_it", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
