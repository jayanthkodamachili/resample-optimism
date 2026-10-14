"""The single frozen description of a scenario, so every protocol sees identical data.

Kept as one immutable dataclass rather than scattered keyword arguments because the whole
claim of this repository is that two protocols differ in nothing except where the resampler
sits. If a protocol could quietly see a different seed or a different split, the measured
optimism would be partly an artefact of configuration drift and the result would mean
nothing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

RESAMPLERS = ("smote", "random_over", "adasyn", "random_under", "none")
ESTIMATORS = ("forest", "tree", "logistic", "knn")
METRICS = ("roc_auc", "average_precision")


class ConfigError(ValueError):
    """Raised when a scenario is impossible to construct rather than merely unusual."""


@dataclass(frozen=True)
class ScenarioConfig:
    """One fully specified experiment.

    Attributes:
        n_samples: rows generated before the holdout is carved off.
        n_features: total columns, informative plus noise.
        n_informative: columns that carry real signal.
        minority_rate: target share of the positive class, before flip noise.
        class_sep: separability passed to make_classification. Lower is harder.
        holdout_frac: share reserved as the untouched honest reference.
        n_splits: stratified folds used by every protocol.
        resampler: which resampling strategy the protocols apply.
        estimator: which classifier sits at the end of the pipeline.
        metric: the score both the CV estimate and the holdout reference report.
        seed: controls generation, splitting, resampling and the estimator.
    """

    n_samples: int = 4000
    n_features: int = 20
    n_informative: int = 6
    minority_rate: float = 0.05
    class_sep: float = 0.8
    holdout_frac: float = 0.3
    n_splits: int = 5
    resampler: str = "smote"
    estimator: str = "forest"
    metric: str = "roc_auc"
    seed: int = 20260214

    def __post_init__(self) -> None:
        if self.resampler not in RESAMPLERS:
            raise ConfigError(f"unknown resampler {self.resampler!r}, expected one of {RESAMPLERS}")
        if self.estimator not in ESTIMATORS:
            raise ConfigError(f"unknown estimator {self.estimator!r}, expected one of {ESTIMATORS}")
        if self.metric not in METRICS:
            raise ConfigError(f"unknown metric {self.metric!r}, expected one of {METRICS}")
        if not 0.0 < self.minority_rate < 0.5:
            raise ConfigError("minority_rate must sit strictly between 0 and 0.5")
        if not 0.0 < self.holdout_frac < 0.9:
            raise ConfigError("holdout_frac must sit strictly between 0 and 0.9")
        if self.n_informative > self.n_features:
            raise ConfigError("n_informative cannot exceed n_features")
        if self.n_splits < 2:
            raise ConfigError("n_splits must be at least 2")
        self._check_minority_survives_folding()

    def _check_minority_survives_folding(self) -> None:
        """Reject a scenario whose rarest fold would hold fewer than two positives.

        A stratified fold with one positive makes ROC-AUC undefined on that fold, and
        scikit-learn fills the gap with a warning and a nan rather than an error. The nan
        then propagates into a mean and the protocol reports nothing while looking like it
        reported something, so the check belongs here rather than downstream.
        """
        pool = int(self.n_samples * (1.0 - self.holdout_frac))
        positives_per_fold = pool * self.minority_rate / self.n_splits
        if positives_per_fold < 2:
            raise ConfigError(
                "scenario yields fewer than two positives per fold "
                f"({positives_per_fold:.2f}); raise n_samples or minority_rate"
            )

    def as_dict(self) -> dict:
        return asdict(self)
