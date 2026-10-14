"""Measure the inflation a cross-validation score picks up when resampling escapes the fold.

The package exists because of one asymmetry. Resampling an imbalanced training set is correct
at fit time and wrong at estimate time, and the two look identical in code. A pipeline that
resamples the whole dataset and then cross-validates produces the same fitted model as a
pipeline that resamples inside each fold, and reports a different score for it. The score is
the only thing that moved, which is precisely why no amount of staring at the score reveals
the problem.
"""

from resample_optimism.config import ScenarioConfig
from resample_optimism.optimism import ProtocolResult, evaluate_all

__all__ = ["ProtocolConfig", "ProtocolResult", "ScenarioConfig", "evaluate_all"]
__version__ = "0.1.0"
