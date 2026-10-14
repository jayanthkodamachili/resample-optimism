from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from resample_optimism.config import ScenarioConfig
from resample_optimism.protocols import (
    PROTOCOLS,
    build_estimator,
    build_resampler,
    correct_resample_in_fold,
    leaked_resample_then_cv,
    model_fingerprint,
    no_resampling,
)


def test_build_resampler_returns_none_when_resampling_is_off():
    assert build_resampler(ScenarioConfig(resampler="none")) is None


@pytest.mark.parametrize("name", ["smote", "random_over", "adasyn", "random_under"])
def test_every_resampler_can_be_built_and_balances_its_output(name, small_pool, small_config):
    config = ScenarioConfig(**{**small_config.as_dict(), "resampler": name})
    x_pool, y_pool, _, _ = small_pool
    resampler = build_resampler(config)
    _, y_res = resampler.fit_resample(x_pool, y_pool)
    minority = int(np.sum(y_res == 1))
    majority = int(np.sum(y_res == 0))
    assert abs(minority - majority) <= max(2, 0.12 * majority)


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("forest", RandomForestClassifier),
        ("tree", DecisionTreeClassifier),
        ("logistic", LogisticRegression),
        ("knn", KNeighborsClassifier),
    ],
)
def test_every_estimator_builds_to_its_class(name, cls):
    assert isinstance(build_estimator(ScenarioConfig(estimator=name)), cls)


def test_build_estimator_pins_single_threaded_execution_where_the_class_allows_it():
    """A published fingerprint must not depend on how many cores the runner had."""
    forest = build_estimator(ScenarioConfig(estimator="forest"))
    assert forest.n_jobs == 1


def test_leaked_protocol_trains_on_more_rows_than_the_raw_pool(small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    leaked = leaked_resample_then_cv(x_pool, y_pool, small_config)
    assert leaked.train_rows > x_pool.shape[0]


def test_no_resampling_protocol_trains_on_exactly_the_raw_pool(small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    baseline = no_resampling(x_pool, y_pool, small_config)
    assert baseline.train_rows == x_pool.shape[0]


def test_leaked_and_corrected_protocols_ship_an_identical_model(small_pool, small_config):
    """The central claim of the repository, asserted rather than described.

    Both protocols resample the whole pool and fit the same estimator on the result when
    they produce a final model. The resampler is seeded, so the two fits are identical. If
    this test ever fails, the README's headline is wrong and the build should stop.
    """
    x_pool, y_pool, x_holdout, _ = small_pool
    leaked = leaked_resample_then_cv(x_pool, y_pool, small_config)
    correct = correct_resample_in_fold(x_pool, y_pool, small_config)
    assert model_fingerprint(leaked.model, x_holdout) == model_fingerprint(
        correct.model, x_holdout
    )


def test_leaked_protocol_reports_a_higher_score_than_the_corrected_one(small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    leaked = leaked_resample_then_cv(x_pool, y_pool, small_config)
    correct = correct_resample_in_fold(x_pool, y_pool, small_config)
    assert leaked.cv_estimate > correct.cv_estimate


def test_corrected_protocol_falls_back_to_a_bare_estimator_when_resampling_is_off(
    small_pool, small_config
):
    config = ScenarioConfig(**{**small_config.as_dict(), "resampler": "none"})
    x_pool, y_pool, _, _ = small_pool
    correct = correct_resample_in_fold(x_pool, y_pool, config)
    assert correct.train_rows == x_pool.shape[0]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_protocol_reports_one_score_per_fold(protocol, small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    fitted = protocol(x_pool, y_pool, small_config)
    assert len(fitted.cv_fold_scores) == small_config.n_splits


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_protocol_mean_matches_its_fold_scores(protocol, small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    fitted = protocol(x_pool, y_pool, small_config)
    assert fitted.cv_estimate == pytest.approx(float(np.mean(fitted.cv_fold_scores)))


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_protocol_is_reproducible(protocol, small_pool, small_config):
    x_pool, y_pool, _, _ = small_pool
    first = protocol(x_pool, y_pool, small_config)
    second = protocol(x_pool, y_pool, small_config)
    assert first.cv_fold_scores == second.cv_fold_scores


def test_fingerprint_is_stable_across_calls(small_pool, small_config):
    x_pool, y_pool, x_holdout, _ = small_pool
    fitted = no_resampling(x_pool, y_pool, small_config)
    assert model_fingerprint(fitted.model, x_holdout) == model_fingerprint(
        fitted.model, x_holdout
    )


def test_fingerprint_separates_two_genuinely_different_models(small_pool, small_config):
    x_pool, y_pool, x_holdout, _ = small_pool
    leaked = leaked_resample_then_cv(x_pool, y_pool, small_config)
    baseline = no_resampling(x_pool, y_pool, small_config)
    assert model_fingerprint(leaked.model, x_holdout) != model_fingerprint(
        baseline.model, x_holdout
    )


def test_fingerprint_survives_a_perturbation_below_the_rounding_floor(small_pool, small_config):
    """Rounding to ten places is deliberate, so prove the tolerance behaves as intended."""

    class Jittered:
        def __init__(self, inner, delta):
            self.inner = inner
            self.delta = delta

        def predict_proba(self, x):
            return self.inner.predict_proba(x) + self.delta

    x_pool, y_pool, x_holdout, _ = small_pool
    fitted = no_resampling(x_pool, y_pool, small_config)
    base = model_fingerprint(fitted.model, x_holdout)
    assert model_fingerprint(Jittered(fitted.model, 1e-15), x_holdout) == base
    assert model_fingerprint(Jittered(fitted.model, 1e-6), x_holdout) != base
