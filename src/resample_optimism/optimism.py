"""Price each protocol's reported score against an honest reference it never touched.

Optimism here has a narrow definition and it is worth stating precisely, because the word is
used loosely elsewhere: optimism is the protocol's cross-validated estimate minus the score
the model that protocol actually ships achieves on a holdout no resampler has seen. It is a
property of the measuring procedure, not of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score

from resample_optimism.config import ScenarioConfig
from resample_optimism.generate import make_scenario, split_pool_and_holdout
from resample_optimism.protocols import PROTOCOLS, model_fingerprint

_SCORERS = {
    "roc_auc": roc_auc_score,
    "average_precision": average_precision_score,
}


@dataclass(frozen=True)
class ProtocolResult:
    """One protocol's reported number, its honest number, and the gap between them."""

    name: str
    cv_estimate: float
    holdout_score: float
    optimism: float
    cv_fold_scores: tuple[float, ...] = field(repr=False, default=())
    train_rows: int = 0
    fingerprint: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "cv_estimate": round(self.cv_estimate, 6),
            "holdout_score": round(self.holdout_score, 6),
            "optimism": round(self.optimism, 6),
            "train_rows": self.train_rows,
            "fingerprint": self.fingerprint,
        }


def score_on_holdout(
    model: BaseEstimator, x_holdout: np.ndarray, y_holdout: np.ndarray, metric: str
) -> float:
    scorer = _SCORERS[metric]
    proba = model.predict_proba(x_holdout)[:, 1]
    return float(scorer(y_holdout, proba))


def evaluate_all(config: ScenarioConfig) -> list[ProtocolResult]:
    """Run every protocol against one shared scenario and return their results in order."""
    x, y = make_scenario(config)
    x_pool, y_pool, x_holdout, y_holdout = split_pool_and_holdout(x, y, config)

    results: list[ProtocolResult] = []
    for protocol in PROTOCOLS:
        fitted = protocol(x_pool, y_pool, config)
        holdout = score_on_holdout(fitted.model, x_holdout, y_holdout, config.metric)
        results.append(
            ProtocolResult(
                name=fitted.name,
                cv_estimate=fitted.cv_estimate,
                holdout_score=holdout,
                optimism=fitted.cv_estimate - holdout,
                cv_fold_scores=fitted.cv_fold_scores,
                train_rows=fitted.train_rows,
                fingerprint=model_fingerprint(fitted.model, x_holdout),
            )
        )
    return results


def by_name(results: list[ProtocolResult], name: str) -> ProtocolResult:
    for result in results:
        if result.name == name:
            return result
    raise KeyError(f"no protocol named {name!r} in results")


def same_model_shipped(results: list[ProtocolResult]) -> bool:
    """True when the leaked and the corrected protocol ship a byte-identical model.

    This is the repository's central claim reduced to a boolean. It holds whenever the
    resampler is deterministic given its seed, which covers every resampler offered here,
    and it is checked rather than asserted in prose.
    """
    leaked = by_name(results, "leaked_resample_then_cv")
    correct = by_name(results, "correct_resample_in_fold")
    return leaked.fingerprint == correct.fingerprint


def resampling_benefit(results: list[ProtocolResult]) -> float:
    """Honest holdout gain from resampling correctly, against not resampling at all.

    Positive means the resampler earned its place. The sign of this number across the sweep
    in experiment four is the uncomfortable part of this repository.
    """
    correct = by_name(results, "correct_resample_in_fold")
    baseline = by_name(results, "no_resampling")
    return correct.holdout_score - baseline.holdout_score
