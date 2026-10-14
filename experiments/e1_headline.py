"""E1: the headline. One scenario, three protocols, averaged over seeds.

Answers the question the README opens with: how many points of ROC-AUC separate what the
leaked protocol reports from what its own model actually achieves, and do the leaked and
corrected protocols really ship the same model every time.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from resample_optimism.config import ScenarioConfig  # noqa: E402
from resample_optimism.optimism import (  # noqa: E402
    evaluate_all,
    resampling_benefit,
    same_model_shipped,
)

SEEDS = (20260214, 11, 2027, 404, 90210, 13, 777, 31337, 5150, 1024)


def run() -> dict:
    per_seed = []
    for seed in SEEDS:
        config = ScenarioConfig(seed=seed)
        results = evaluate_all(config)
        per_seed.append(
            {
                "seed": seed,
                "identical_model": same_model_shipped(results),
                "resampling_benefit": resampling_benefit(results),
                "protocols": {r.name: r.as_dict() for r in results},
            }
        )

    def mean_of(protocol: str, field: str) -> float:
        return float(np.mean([s["protocols"][protocol][field] for s in per_seed]))

    def spread_of(protocol: str, field: str) -> float:
        values = [s["protocols"][protocol][field] for s in per_seed]
        return float(max(values) - min(values))

    names = ["leaked_resample_then_cv", "correct_resample_in_fold", "no_resampling"]
    summary = {
        name: {
            "cv_estimate_mean": mean_of(name, "cv_estimate"),
            "holdout_score_mean": mean_of(name, "holdout_score"),
            "optimism_mean": mean_of(name, "optimism"),
            "optimism_spread": spread_of(name, "optimism"),
        }
        for name in names
    }

    leaked = summary["leaked_resample_then_cv"]
    corrected = summary["correct_resample_in_fold"]
    payload = {
        "experiment": "e1_headline",
        "question": "How far is the leaked estimate from the honest score of its own model?",
        "seeds": list(SEEDS),
        "scenario": ScenarioConfig().as_dict(),
        "summary": summary,
        "per_seed": per_seed,
        "headline": {
            "leaked_cv_mean": leaked["cv_estimate_mean"],
            "leaked_holdout_mean": leaked["holdout_score_mean"],
            "leaked_optimism_mean": leaked["optimism_mean"],
            "corrected_optimism_mean": corrected["optimism_mean"],
            "optimism_ratio": leaked["optimism_mean"] / abs(corrected["optimism_mean"]),
            "seeds_with_identical_model": sum(1 for s in per_seed if s["identical_model"]),
            "seeds_total": len(SEEDS),
        },
    }
    return payload


def main() -> None:
    payload = run()
    rows = [
        [
            name,
            f"{payload['summary'][name]['cv_estimate_mean']:.4f}",
            f"{payload['summary'][name]['holdout_score_mean']:.4f}",
            f"{payload['summary'][name]['optimism_mean']:+.4f}",
            f"{payload['summary'][name]['optimism_spread']:.4f}",
        ]
        for name in payload["summary"]
    ]
    print(f"E1 headline, {len(SEEDS)} seeds, SMOTE + random forest, 5 percent minority\n")
    headers = ["protocol", "cv est", "holdout", "optimism", "spread"]
    print(table(headers, rows, [30, 10, 11, 11, 10]))
    head = payload["headline"]
    print()
    print(
        f"identical model shipped in {head['seeds_with_identical_model']}"
        f"/{head['seeds_total']} seeds"
    )
    path = write("e1_headline", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
