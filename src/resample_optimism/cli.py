"""Command line entry point.

The exit codes carry the result rather than merely reporting success. A tool that returns
zero whenever it managed to run is useless in CI, because the interesting outcome is not
"the audit completed" but "the audit found a leak". Code 1 therefore means the tool worked
perfectly and the answer was bad news.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from resample_optimism.config import ESTIMATORS, METRICS, RESAMPLERS, ConfigError, ScenarioConfig
from resample_optimism.optimism import evaluate_all, resampling_benefit, same_model_shipped

EXIT_WITHIN_TOLERANCE = 0
EXIT_LEAK_DETECTED = 1
EXIT_BAD_CONFIG = 2
EXIT_INTERNAL_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resample-optimism",
        description="Measure how much a CV score is inflated when resampling escapes the fold.",
    )
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-informative", type=int, default=6)
    parser.add_argument("--minority-rate", type=float, default=0.05)
    parser.add_argument("--class-sep", type=float, default=0.8)
    parser.add_argument("--holdout-frac", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--resampler", choices=RESAMPLERS, default="smote")
    parser.add_argument("--estimator", choices=ESTIMATORS, default="forest")
    parser.add_argument("--metric", choices=METRICS, default="roc_auc")
    parser.add_argument("--seed", type=int, default=20260214)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="optimism above this, in metric points, exits 1",
    )
    parser.add_argument("--json", action="store_true", help="emit machine readable output only")
    return parser


def config_from_args(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        minority_rate=args.minority_rate,
        class_sep=args.class_sep,
        holdout_frac=args.holdout_frac,
        n_splits=args.n_splits,
        resampler=args.resampler,
        estimator=args.estimator,
        metric=args.metric,
        seed=args.seed,
    )


def render_table(results, config, benefit, identical) -> str:
    lines = [
        f"scenario: {config.resampler} + {config.estimator}, "
        f"minority {config.minority_rate:.1%}, metric {config.metric}",
        "",
        f"{'protocol':<28}{'cv estimate':>13}{'honest holdout':>16}{'optimism':>11}",
        "-" * 68,
    ]
    for result in results:
        lines.append(
            f"{result.name:<28}{result.cv_estimate:>13.4f}"
            f"{result.holdout_score:>16.4f}{result.optimism:>+11.4f}"
        )
    lines.append("-" * 68)
    lines.append(
        "leaked and corrected protocols ship an identical model: "
        + ("yes" if identical else "no")
    )
    lines.append(f"honest gain from resampling correctly: {benefit:+.4f} {config.metric} points")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except ConfigError as exc:
        print(f"invalid scenario: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        results = evaluate_all(config)
        benefit = resampling_benefit(results)
        identical = same_model_shipped(results)
    except Exception as exc:  # noqa: BLE001
        print(f"audit failed to complete: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    leaked = next(r for r in results if r.name == "leaked_resample_then_cv")

    if args.json:
        payload = {
            "config": config.as_dict(),
            "results": [r.as_dict() for r in results],
            "same_model_shipped": identical,
            "resampling_benefit": round(benefit, 6),
            "tolerance": args.tolerance,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_table(results, config, benefit, identical))

    return EXIT_LEAK_DETECTED if leaked.optimism > args.tolerance else EXIT_WITHIN_TOLERANCE


if __name__ == "__main__":
    raise SystemExit(main())
