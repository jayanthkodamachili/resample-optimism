"""Re-measure every published figure and write docs/metrics.json.

Nothing in this file types a number. It runs the suite, runs the four experiments, reads
their machine readable output and derives the handful of values the documents quote. The
test count and coverage come from the junit XML and the coverage JSON rather than from
parsing pytest's progress line, because a parsed progress line silently becomes whatever
the last invocation happened to print.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

EXPERIMENTS = (
    "e1_headline",
    "e2_by_resampler",
    "e3_by_estimator",
    "e4_was_it_worth_it",
)

CHECKED_DOCUMENTS = (
    "README.md",
    "docs/defense-guide.md",
    "docs/adr/ADR-005-what-this-cannot-measure.md",
)

# Each metric names the exact wording a document must use, with the formatted value
# substituted for the brace. Alternatives, because the same figure reads differently in a
# table cell and in a sentence. Anchors stay short: a long phrase wraps across a newline in
# rendered markdown and then matches nothing while the prose it guards is perfectly correct.
ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests"],
    "coverage_line_pct": ["{} percent line coverage", "{}% line coverage"],
    "leaked_cv_mean": ["reports {}", "{} mean ROC-AUC"],
    "leaked_holdout_mean": ["scores {}", "{} on the holdout"],
    "leaked_optimism_mean": ["{} points", "optimism of {}"],
    "corrected_optimism_mean": ["{} points", "corrected sits at {}"],
    "identical_model_seeds": ["{} of 10 seeds", "{}/10 seeds"],
    "random_over_leaked_cv": ["a perfect {}", "{} ROC-AUC"],
    "random_under_optimism": ["{} for undersampling", "undersampling at {}"],
    "worst_estimator_optimism": ["{} points", "costs {}"],
    "mildest_estimator_optimism": ["{} points", "to {} for logistic"],
    "estimator_severity_ratio": ["{} times", "a factor of {}"],
    "e4_cells_total": ["of {} cells", "{} cells swept"],
    "e4_runs_total": ["of {} runs", "{} individual runs"],
    "e4_ap_cells_positive": ["{} of 12 cells", "{}/12 cells"],
    "e4_ap_runs_negative": ["{} of 60", "{}/60 runs"],
    "e4_ap_grand_mean": ["{} average precision", "mean of {}"],
    "e4_ap_worst_cell": ["worst cell loses {}", "down {}"],
    "e4_auc_cells_positive": ["{} of the same 12", "{} cells improve"],
    "e4_auc_grand_mean": ["{} ROC-AUC", "gains {}"],
    "e4_loss_gain_ratio": ["{} times larger", "{} times the size"],
}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=False, **kwargs)


def measure_suite(skip_tests: bool) -> dict:
    """Read the test count and coverage from machine readable artefacts.

    With skip_tests, the artefacts are expected to already exist because CI ran pytest as
    its own step. That separation matters: a failing test surfacing here as a traceback
    from a metrics script is the kind of CI signature that costs an hour to attribute.
    """
    junit = ROOT / "junit.xml"
    cov = ROOT / "coverage.json"

    if skip_tests:
        missing = [p.name for p in (junit, cov) if not p.exists()]
        if missing:
            raise SystemExit(f"--skip-tests given but {', '.join(missing)} not found")
    else:
        result = run(
            [
                sys.executable, "-m", "pytest",
                f"--junitxml={junit}",
                "--cov=resample_optimism",
                "--cov-report=json",
                "-q",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-4000:], file=sys.stderr)
            raise SystemExit("test suite failed, refusing to publish metrics")

    suite = ET.parse(junit).getroot()
    node = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(node.get("tests", 0))
    failures = int(node.get("failures", 0)) + int(node.get("errors", 0))
    skipped = int(node.get("skipped", 0))

    coverage = json.loads(cov.read_text())["totals"]["percent_covered"]
    return {
        "tests_total": total - skipped,
        "tests_failed": failures,
        "coverage_line_pct": round(coverage, 1),
    }


def ensure_experiments(rerun: bool) -> dict[str, dict]:
    payloads = {}
    for name in EXPERIMENTS:
        path = EXPERIMENT_DIR / f"{name}.json"
        if rerun or not path.exists():
            print(f"running experiment {name}")
            result = run([sys.executable, str(ROOT / "experiments" / f"{name}.py")])
            if result.returncode != 0:
                raise SystemExit(f"experiment {name} failed")
        payloads[name] = json.loads(path.read_text())
    return payloads


def derive(suite: dict, exp: dict[str, dict]) -> dict[str, float | int]:
    e1 = exp["e1_headline"]["headline"]
    e2 = exp["e2_by_resampler"]
    e3 = exp["e3_by_estimator"]
    e4 = exp["e4_was_it_worth_it"]
    ap = e4["sweeps"]["average_precision"]
    auc = e4["sweeps"]["roc_auc"]

    worst_est = e3["worst_optimism"]
    mildest_est = e3["mildest_optimism"]

    return {
        "tests_total": suite["tests_total"],
        "coverage_line_pct": suite["coverage_line_pct"],
        "leaked_cv_mean": round(e1["leaked_cv_mean"], 4),
        "leaked_holdout_mean": round(e1["leaked_holdout_mean"], 4),
        "leaked_optimism_mean": round(e1["leaked_optimism_mean"], 4),
        "corrected_optimism_mean": round(e1["corrected_optimism_mean"], 4),
        "identical_model_seeds": e1["seeds_with_identical_model"],
        "random_over_leaked_cv": round(e2["by_resampler"]["random_over"]["leaked_cv_mean"], 4),
        "random_under_optimism": round(e2["by_resampler"]["random_under"]["optimism_mean"], 4),
        "worst_estimator_optimism": round(worst_est, 4),
        "mildest_estimator_optimism": round(mildest_est, 4),
        "estimator_severity_ratio": round(worst_est / mildest_est, 1),
        "e4_cells_total": ap["cells_total"],
        "e4_runs_total": ap["runs_total"],
        "e4_ap_cells_positive": ap["cells_positive"],
        "e4_ap_runs_negative": ap["runs_negative"],
        "e4_ap_grand_mean": round(ap["grand_mean_benefit"], 4),
        "e4_ap_worst_cell": round(ap["worst_cell"]["benefit_mean"], 4),
        "e4_auc_cells_positive": auc["cells_positive"],
        "e4_auc_grand_mean": round(auc["grand_mean_benefit"], 4),
        "e4_loss_gain_ratio": round(
            abs(ap["grand_mean_benefit"]) / abs(auc["grand_mean_benefit"]), 1
        ),
    }


# Formatting is declared per metric rather than inferred from the name. The first version
# inferred it from substrings and silently printed a ratio as 6.7000 and a signed mean
# without its plus, because neither name contained the words the heuristic looked for.
SIGNED = frozenset(
    {
        "leaked_optimism_mean",
        "corrected_optimism_mean",
        "random_under_optimism",
        "worst_estimator_optimism",
        "mildest_estimator_optimism",
        "e4_ap_grand_mean",
        "e4_ap_worst_cell",
        "e4_auc_grand_mean",
    }
)
ONE_PLACE = frozenset({"estimator_severity_ratio", "e4_loss_gain_ratio"})


def display_strings(metrics: dict) -> dict[str, str]:
    """How each value must appear in prose, so the checker and the author agree on format."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, int):
            out[key] = str(value)
        elif key in ONE_PLACE:
            out[key] = f"{value:.1f}"
        elif key == "coverage_line_pct":
            out[key] = f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
        elif key in SIGNED:
            out[key] = f"{value:+.4f}"
        else:
            out[key] = f"{value:.4f}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-measure every published figure.")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="read existing experiment JSON instead of re-running, for local iteration only",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="read junit.xml and coverage.json produced by a separate pytest step",
    )
    args = parser.parse_args()

    suite = measure_suite(skip_tests=args.skip_tests)
    experiments = ensure_experiments(rerun=not args.reuse)
    metrics = derive(suite, experiments)

    payload = {
        "metrics": metrics,
        "display": display_strings(metrics),
        "anchors": ANCHORS,
        "checked_documents": list(CHECKED_DOCUMENTS),
        "note": "every value here is produced by running the suite and the four experiments",
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {METRICS_PATH} with {len(metrics)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
