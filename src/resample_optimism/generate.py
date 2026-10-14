"""A seeded generator rather than a committed real dataset.

A public imbalanced dataset has an unknown true decision boundary, so the honest holdout
score it produces is itself an estimate and the optimism measured against it mixes two
errors. A generator gives an arbitrarily large untouched reference set drawn from the same
distribution, which is the only way to price the leak against something that is not also
guessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from resample_optimism.config import ScenarioConfig


def make_scenario(config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a single imbalanced binary dataset for the given configuration."""
    x, y = make_classification(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[1.0 - config.minority_rate, config.minority_rate],
        class_sep=config.class_sep,
        flip_y=0.01,
        shuffle=True,
        random_state=config.seed,
    )
    return x, y


def split_pool_and_holdout(
    x: np.ndarray, y: np.ndarray, config: ScenarioConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carve the untouched honest reference off the modelling pool.

    The holdout is produced once, here, and no resampler is ever shown it. Every protocol
    receives exactly the same pool and is scored against exactly the same holdout, which is
    what makes the differences between protocols attributable to the protocol.
    """
    x_pool, x_holdout, y_pool, y_holdout = train_test_split(
        x,
        y,
        test_size=config.holdout_frac,
        stratify=y,
        random_state=config.seed,
    )
    return x_pool, y_pool, x_holdout, y_holdout


def imbalance_ratio(y: np.ndarray) -> float:
    """Majority count divided by minority count, the number people quote in standups."""
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        raise ValueError("no positive class present, imbalance ratio is undefined")
    return negatives / positives
