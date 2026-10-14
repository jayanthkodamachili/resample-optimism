"""Every test name states the property the configuration object must hold."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from resample_optimism.config import ESTIMATORS, METRICS, RESAMPLERS, ConfigError, ScenarioConfig


def test_default_config_is_constructible():
    assert ScenarioConfig().resampler == "smote"


def test_config_is_frozen_so_a_protocol_cannot_mutate_the_shared_scenario():
    config = ScenarioConfig()
    with pytest.raises(FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


def test_unknown_resampler_is_rejected():
    with pytest.raises(ConfigError, match="unknown resampler"):
        ScenarioConfig(resampler="smoothy")


def test_unknown_estimator_is_rejected():
    with pytest.raises(ConfigError, match="unknown estimator"):
        ScenarioConfig(estimator="transformer")


def test_unknown_metric_is_rejected():
    with pytest.raises(ConfigError, match="unknown metric"):
        ScenarioConfig(metric="accuracy")


@pytest.mark.parametrize("rate", [0.0, -0.1, 0.5, 0.9])
def test_minority_rate_outside_the_open_unit_half_is_rejected(rate):
    with pytest.raises(ConfigError, match="minority_rate"):
        ScenarioConfig(minority_rate=rate)


@pytest.mark.parametrize("frac", [0.0, 0.95, 1.0, -0.2])
def test_holdout_fraction_outside_its_range_is_rejected(frac):
    with pytest.raises(ConfigError, match="holdout_frac"):
        ScenarioConfig(holdout_frac=frac)


def test_more_informative_features_than_features_is_rejected():
    with pytest.raises(ConfigError, match="n_informative"):
        ScenarioConfig(n_features=5, n_informative=9)


def test_single_fold_is_rejected():
    with pytest.raises(ConfigError, match="n_splits"):
        ScenarioConfig(n_splits=1)


def test_scenario_with_under_two_positives_per_fold_is_rejected():
    """This guard exists because of a real failure during the build.

    An early minority-rate sweep reached 0.5 percent at four thousand rows. Three of the
    five folds held one positive, ROC-AUC came back nan for those folds, numpy's mean
    propagated the nan, and the experiment wrote nan into its JSON without raising. The
    sweep looked complete. The guard turns that into a refusal at construction time.
    """
    with pytest.raises(ConfigError, match="two positives per fold"):
        ScenarioConfig(n_samples=800, minority_rate=0.005, n_splits=5)


def test_as_dict_round_trips_into_a_new_config():
    config = ScenarioConfig(resampler="adasyn", seed=11)
    assert ScenarioConfig(**config.as_dict()) == config


@pytest.mark.parametrize("resampler", RESAMPLERS)
def test_every_declared_resampler_is_accepted(resampler):
    assert ScenarioConfig(resampler=resampler).resampler == resampler


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_declared_estimator_is_accepted(estimator):
    assert ScenarioConfig(estimator=estimator).estimator == estimator


@pytest.mark.parametrize("metric", METRICS)
def test_every_declared_metric_is_accepted(metric):
    assert ScenarioConfig(metric=metric).metric == metric
