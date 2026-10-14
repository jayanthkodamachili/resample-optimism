"""E3: does the leak depend on the model's capacity to memorise?

A duplicated row only inflates a score if the model can recognise it. A one-nearest-style
learner finds its own training point sitting in the validation fold; a linear model with
twenty coefficients has nowhere to store it. If this holds, the same protocol mistake is
nearly harmless for one team and catastrophic for another, which is a more useful thing to
know than the rule on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from resample_optimism.config import ScenarioConfig  # noqa: E402
from resample_optimism.optimism import by_name, evaluate_all  # noqa: E402

SEEDS = (20260214, 11, 2027, 404, 90210)
ESTIMATORS = ("knn", "tree", "forest", "logistic")


def run() -> dict:
    rows = {}
    for estimator in ESTIMATORS:
        optimisms, cvs, holdouts = [], [], []
        for seed in SEEDS:
            config = ScenarioConfig(estimator=estimator, resampler="random_over", seed=seed)
            results = evaluate_all(config)
            leaked = by_name(results, "leaked_resample_then_cv")
            optimisms.append(leaked.optimism)
            cvs.append(leaked.cv_estimate)
            holdouts.append(leaked.holdout_score)
        rows[estimator] = {
            "leaked_cv_mean": float(np.mean(cvs)),
            "holdout_mean": float(np.mean(holdouts)),
            "optimism_mean": float(np.mean(optimisms)),
            "optimism_max": float(np.max(optimisms)),
        }

    ranked = sorted(rows.items(), key=lambda kv: kv[1]["optimism_mean"], reverse=True)
    return {
        "experiment": "e3_by_estimator",
        "question": "Does leak severity track the model's capacity to memorise a duplicate?",
        "seeds": list(SEEDS),
        "resampler": "random_over",
        "by_estimator": rows,
        "worst_estimator": ranked[0][0],
        "worst_optimism": ranked[0][1]["optimism_mean"],
        "mildest_estimator": ranked[-1][0],
        "mildest_optimism": ranked[-1][1]["optimism_mean"],
    }


def main() -> None:
    payload = run()
    rows = [
        [
            name,
            f"{stats['leaked_cv_mean']:.4f}",
            f"{stats['holdout_mean']:.4f}",
            f"{stats['optimism_mean']:+.4f}",
        ]
        for name, stats in payload["by_estimator"].items()
    ]
    print(f"E3 leak severity by estimator, {len(SEEDS)} seeds, random oversampling\n")
    print(table(["estimator", "leaked cv", "holdout", "optimism"], rows, [14, 12, 11, 11]))
    print()
    print(
        f"worst {payload['worst_estimator']} at {payload['worst_optimism']:+.4f}, "
        f"mildest {payload['mildest_estimator']} at {payload['mildest_optimism']:+.4f}"
    )
    path = write("e3_by_estimator", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
