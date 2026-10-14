from __future__ import annotations

import numpy as np
import pytest

from resample_optimism.config import ScenarioConfig
from resample_optimism.generate import (
    imbalance_ratio,
    make_scenario,
    split_pool_and_holdout,
)


def test_generated_scenario_has_the_requested_shape(small_config):
    x, y = make_scenario(small_config)
    assert x.shape == (small_config.n_samples, small_config.n_features)
    assert y.shape == (small_config.n_samples,)


def test_generation_is_reproducible_from_the_seed(small_config):
    first_x, first_y = make_scenario(small_config)
    second_x, second_y = make_scenario(small_config)
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


def test_a_different_seed_produces_different_data(small_config):
    other = ScenarioConfig(**{**small_config.as_dict(), "seed": small_config.seed + 1})
    first_x, _ = make_scenario(small_config)
    second_x, _ = make_scenario(other)
    assert not np.allclose(first_x, second_x)


def test_minority_share_lands_near_the_requested_rate(small_config):
    _, y = make_scenario(small_config)
    observed = float(np.mean(y == 1))
    assert abs(observed - small_config.minority_rate) < 0.05


def test_holdout_receives_the_requested_share(small_config):
    x, y = make_scenario(small_config)
    x_pool, _, x_holdout, _ = split_pool_and_holdout(x, y, small_config)
    expected = round(small_config.n_samples * small_config.holdout_frac)
    assert x_holdout.shape[0] == expected
    assert x_pool.shape[0] + x_holdout.shape[0] == small_config.n_samples


def test_split_is_stratified_so_both_sides_carry_the_minority(small_config):
    x, y = make_scenario(small_config)
    _, y_pool, _, y_holdout = split_pool_and_holdout(x, y, small_config)
    pool_rate = float(np.mean(y_pool == 1))
    holdout_rate = float(np.mean(y_holdout == 1))
    assert abs(pool_rate - holdout_rate) < 0.02


def test_pool_and_holdout_share_no_rows(small_config):
    """The honest reference is only honest if it is disjoint from the modelling pool."""
    x, y = make_scenario(small_config)
    x_pool, _, x_holdout, _ = split_pool_and_holdout(x, y, small_config)
    pool_rows = {row.tobytes() for row in x_pool}
    holdout_rows = {row.tobytes() for row in x_holdout}
    assert pool_rows.isdisjoint(holdout_rows)


def test_splitting_is_reproducible(small_config):
    x, y = make_scenario(small_config)
    first = split_pool_and_holdout(x, y, small_config)
    second = split_pool_and_holdout(x, y, small_config)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_imbalance_ratio_counts_majority_per_minority():
    y = np.array([0] * 90 + [1] * 10)
    assert imbalance_ratio(y) == pytest.approx(9.0)


def test_imbalance_ratio_refuses_a_single_class():
    with pytest.raises(ValueError, match="no positive class"):
        imbalance_ratio(np.zeros(10, dtype=int))
