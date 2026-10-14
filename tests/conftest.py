"""Shared fixtures.

Scenarios here are deliberately small. The repository's published numbers come from the
experiments, not from the test suite, so a test only has to exercise a code path and assert
a property. A test suite that re-runs the headline scenario buys nothing and costs several
minutes on every push.
"""

from __future__ import annotations

import pytest

from resample_optimism.config import ScenarioConfig
from resample_optimism.generate import make_scenario, split_pool_and_holdout


@pytest.fixture
def small_config() -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=600,
        n_features=8,
        n_informative=4,
        minority_rate=0.15,
        n_splits=3,
        estimator="tree",
        seed=7,
    )


@pytest.fixture
def small_pool(small_config):
    x, y = make_scenario(small_config)
    return split_pool_and_holdout(x, y, small_config)
