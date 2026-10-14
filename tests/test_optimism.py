from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from resample_optimism.config import ScenarioConfig
from resample_optimism.optimism import (
    ProtocolResult,
    by_name,
    evaluate_all,
    resampling_benefit,
    same_model_shipped,
    score_on_holdout,
)
from resample_optimism.protocols import no_resampling


@pytest.fixture(scope="module")
def results():
    config = ScenarioConfig(
        n_samples=600, n_features=8, n_informative=4, minority_rate=0.15,
        n_splits=3, estimator="tree", seed=7,
    )
    return evaluate_all(config)


def test_every_protocol_appears_once(results):
    names = [r.name for r in results]
    assert sorted(names) == sorted(set(names))
    assert len(names) == 3


def test_optimism_is_the_estimate_minus_the_honest_score(results):
    for result in results:
        assert result.optimism == pytest.approx(result.cv_estimate - result.holdout_score)


def test_the_leaked_protocol_is_the_most_optimistic(results):
    leaked = by_name(results, "leaked_resample_then_cv")
    others = [r.optimism for r in results if r.name != "leaked_resample_then_cv"]
    assert leaked.optimism > max(others)


def test_leaked_and_corrected_report_the_same_honest_score(results):
    """They ship the same model, so the honest column cannot distinguish them.

    This is the asymmetry the repository is about. If this ever fails while the fingerprint
    test passes, the holdout scoring path has become non-deterministic.
    """
    leaked = by_name(results, "leaked_resample_then_cv")
    correct = by_name(results, "correct_resample_in_fold")
    assert leaked.holdout_score == pytest.approx(correct.holdout_score)


def test_same_model_shipped_is_true_for_a_seeded_resampler(results):
    assert same_model_shipped(results) is True


def test_resampling_benefit_compares_corrected_against_no_resampling(results):
    correct = by_name(results, "correct_resample_in_fold")
    baseline = by_name(results, "no_resampling")
    assert resampling_benefit(results) == pytest.approx(
        correct.holdout_score - baseline.holdout_score
    )


def test_by_name_raises_for_an_absent_protocol(results):
    with pytest.raises(KeyError, match="no protocol named"):
        by_name(results, "wishful_thinking")


def test_score_on_holdout_handles_both_metrics(small_pool, small_config):
    x_pool, y_pool, x_holdout, y_holdout = small_pool
    fitted = no_resampling(x_pool, y_pool, small_config)
    auc = score_on_holdout(fitted.model, x_holdout, y_holdout, "roc_auc")
    ap = score_on_holdout(fitted.model, x_holdout, y_holdout, "average_precision")
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ap <= 1.0


def test_a_perfect_separator_scores_one_on_the_holdout():
    """A sanity anchor, so a broken scorer cannot hide behind plausible middling numbers."""

    class Oracle:
        def __init__(self, labels):
            self.labels = labels

        def predict_proba(self, x):
            return np.column_stack([1 - self.labels, self.labels]).astype(float)

    y = np.array([0, 0, 1, 1, 0, 1])
    assert score_on_holdout(Oracle(y), np.zeros((6, 2)), y, "roc_auc") == pytest.approx(1.0)


def test_result_as_dict_rounds_and_keeps_every_published_field(results):
    payload = by_name(results, "no_resampling").as_dict()
    assert set(payload) == {
        "name", "cv_estimate", "holdout_score", "optimism", "train_rows", "fingerprint"
    }
    assert len(payload["fingerprint"]) == 16


def test_results_are_frozen_so_a_report_cannot_edit_a_measurement(results):
    with pytest.raises(FrozenInstanceError):
        results[0].optimism = 0.0  # type: ignore[misc]


def test_protocol_result_defaults_are_safe_to_construct():
    result = ProtocolResult(name="x", cv_estimate=0.5, holdout_score=0.4, optimism=0.1)
    assert result.train_rows == 0
    assert result.cv_fold_scores == ()
