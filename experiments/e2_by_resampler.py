"""E2: does the leak depend on which resampler is used?

The hypothesis worth testing is that the leak is not about resampling, it is about
dependence between rows. Random oversampling emits exact duplicates, SMOTE emits
interpolations of near neighbours, and random undersampling emits nothing new at all. If
the hypothesis holds, undersampling should barely leak, which would make "resampling before
the split is wrong" too coarse a rule to be useful.
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
RESAMPLERS = ("random_over", "smote", "adasyn", "random_under")


def run() -> dict:
    rows = {}
    for resampler in RESAMPLERS:
        optimisms, cvs, holdouts = [], [], []
        for seed in SEEDS:
            config = ScenarioConfig(resampler=resampler, seed=seed)
            results = evaluate_all(config)
            leaked = by_name(results, "leaked_resample_then_cv")
            optimisms.append(leaked.optimism)
            cvs.append(leaked.cv_estimate)
            holdouts.append(leaked.holdout_score)
        rows[resampler] = {
            "leaked_cv_mean": float(np.mean(cvs)),
            "holdout_mean": float(np.mean(holdouts)),
            "optimism_mean": float(np.mean(optimisms)),
            "optimism_min": float(np.min(optimisms)),
            "optimism_max": float(np.max(optimisms)),
        }

    ranked = sorted(rows.items(), key=lambda kv: kv[1]["optimism_mean"], reverse=True)
    return {
        "experiment": "e2_by_resampler",
        "question": "Does leak severity track row dependence rather than resampling itself?",
        "seeds": list(SEEDS),
        "estimator": "forest",
        "by_resampler": rows,
        "worst_resampler": ranked[0][0],
        "worst_optimism": ranked[0][1]["optimism_mean"],
        "mildest_resampler": ranked[-1][0],
        "mildest_optimism": ranked[-1][1]["optimism_mean"],
        "severity_ratio": ranked[0][1]["optimism_mean"] / ranked[-1][1]["optimism_mean"],
    }


def main() -> None:
    payload = run()
    rows = [
        [
            name,
            f"{stats['leaked_cv_mean']:.4f}",
            f"{stats['holdout_mean']:.4f}",
            f"{stats['optimism_mean']:+.4f}",
            f"{stats['optimism_max']:+.4f}",
        ]
        for name, stats in payload["by_resampler"].items()
    ]
    print(f"E2 leak severity by resampler, {len(SEEDS)} seeds, random forest\n")
    headers = ["resampler", "leaked cv", "holdout", "optimism", "worst"]
    print(table(headers, rows, [16, 12, 11, 11, 10]))
    print()
    print(
        f"worst {payload['worst_resampler']} at {payload['worst_optimism']:+.4f}, "
        f"mildest {payload['mildest_resampler']} at {payload['mildest_optimism']:+.4f}"
    )
    path = write("e2_by_resampler", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
