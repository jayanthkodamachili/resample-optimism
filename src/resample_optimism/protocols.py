"""The three evaluation protocols, written so their only difference is where the resampler sits.

Read the three functions side by side. `leaked_resample_then_cv` and `correct_resample_in_fold`
fit the identical final model from the identical data with the identical seed. They disagree
only about which rows the scorer is allowed to see. That is the entire subject of this
repository, and the code is laid out to make the single line of difference visible rather
than to minimise duplication.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from resample_optimism.config import ScenarioConfig


@dataclass(frozen=True)
class FittedProtocol:
    """What a protocol produces: a number it reports, and the model it would actually ship."""

    name: str
    cv_estimate: float
    cv_fold_scores: tuple[float, ...]
    model: BaseEstimator
    train_rows: int


def build_resampler(config: ScenarioConfig):
    """Return the configured resampler, or None when the scenario asks for no resampling."""
    if config.resampler == "none":
        return None
    if config.resampler == "smote":
        return SMOTE(random_state=config.seed, k_neighbors=5)
    if config.resampler == "random_over":
        return RandomOverSampler(random_state=config.seed)
    if config.resampler == "adasyn":
        return ADASYN(random_state=config.seed, n_neighbors=5)
    if config.resampler == "random_under":
        return RandomUnderSampler(random_state=config.seed)
    raise ValueError(f"unreachable: config validation should have rejected {config.resampler!r}")


def build_estimator(config: ScenarioConfig) -> BaseEstimator:
    """Return a fresh, deterministic estimator.

    n_jobs is pinned to one everywhere. Parallel tree building in scikit-learn is
    deterministic given a seed, but the repository publishes a model fingerprint and a
    fingerprint that depends on worker scheduling is not a fingerprint.
    """
    if config.estimator == "forest":
        return RandomForestClassifier(
            n_estimators=60, random_state=config.seed, n_jobs=1, min_samples_leaf=1
        )
    if config.estimator == "tree":
        return DecisionTreeClassifier(random_state=config.seed)
    if config.estimator == "logistic":
        return LogisticRegression(max_iter=2000, random_state=config.seed)
    if config.estimator == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    raise ValueError(f"unreachable: config validation should have rejected {config.estimator!r}")


def _folds(config: ScenarioConfig) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)


def _resample(x: np.ndarray, y: np.ndarray, config: ScenarioConfig):
    resampler = build_resampler(config)
    if resampler is None:
        return x, y
    return resampler.fit_resample(x, y)


def leaked_resample_then_cv(
    x_pool: np.ndarray, y_pool: np.ndarray, config: ScenarioConfig
) -> FittedProtocol:
    """Resample the whole pool, then cross-validate the resampled pool.

    This is the common mistake and it does not look like one. The resampler runs once at the
    top of the notebook, the cross-validation runs below it, and the two are separated by
    enough cells that nobody connects them. Synthetic minority points interpolated from a row
    that lands in the validation fold are now sitting in the training fold, so the model is
    scored partly on neighbourhoods it was fitted on.
    """
    x_res, y_res = _resample(x_pool, y_pool, config)
    estimator = build_estimator(config)
    scores = cross_val_score(
        clone(estimator), x_res, y_res, cv=_folds(config), scoring=config.metric, n_jobs=1
    )
    shipped = clone(estimator).fit(x_res, y_res)
    return FittedProtocol(
        name="leaked_resample_then_cv",
        cv_estimate=float(np.mean(scores)),
        cv_fold_scores=tuple(float(s) for s in scores),
        model=shipped,
        train_rows=int(x_res.shape[0]),
    )


def correct_resample_in_fold(
    x_pool: np.ndarray, y_pool: np.ndarray, config: ScenarioConfig
) -> FittedProtocol:
    """Resample inside each fold, on the training portion only.

    The resampler becomes a pipeline step, so cross_val_score refits it per fold and the
    validation portion of every fold is raw. Note what has not changed: fitting this pipeline
    on the full pool resamples the full pool and fits the same estimator on the result, which
    is exactly what the leaked protocol shipped. The fix corrects the estimate, not the model.
    """
    estimator = build_estimator(config)
    resampler = build_resampler(config)
    steps = ([("resampler", resampler)] if resampler is not None else []) + [("clf", estimator)]
    pipeline = ImbPipeline(steps)
    scores = cross_val_score(
        clone(pipeline), x_pool, y_pool, cv=_folds(config), scoring=config.metric, n_jobs=1
    )
    shipped = clone(pipeline).fit(x_pool, y_pool)
    x_res, _ = _resample(x_pool, y_pool, config)
    return FittedProtocol(
        name="correct_resample_in_fold",
        cv_estimate=float(np.mean(scores)),
        cv_fold_scores=tuple(float(s) for s in scores),
        model=shipped,
        train_rows=int(x_res.shape[0]),
    )


def no_resampling(
    x_pool: np.ndarray, y_pool: np.ndarray, config: ScenarioConfig
) -> FittedProtocol:
    """Cross-validate the raw imbalanced pool with no resampling anywhere.

    Present as the control nobody runs. The interesting comparison in this repository is not
    leaked against correct, which is a foregone conclusion, but correct against this.
    """
    estimator = build_estimator(config)
    scores = cross_val_score(
        clone(estimator), x_pool, y_pool, cv=_folds(config), scoring=config.metric, n_jobs=1
    )
    shipped = clone(estimator).fit(x_pool, y_pool)
    return FittedProtocol(
        name="no_resampling",
        cv_estimate=float(np.mean(scores)),
        cv_fold_scores=tuple(float(s) for s in scores),
        model=shipped,
        train_rows=int(x_pool.shape[0]),
    )


PROTOCOLS = (leaked_resample_then_cv, correct_resample_in_fold, no_resampling)


def model_fingerprint(model: BaseEstimator, x: np.ndarray) -> str:
    """A stable digest of a model's behaviour on fixed inputs.

    Two protocols claiming to ship the same model is a claim, and a claim in this repository
    has to be checkable. Probabilities are rounded to ten places before hashing because the
    last bits of a float sum move with summation order on some BLAS builds, and a fingerprint
    that changes with the linear algebra backend would fail a build for no reason.
    """
    proba = model.predict_proba(x)[:, 1]
    rounded = np.round(proba.astype(np.float64), 10)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
