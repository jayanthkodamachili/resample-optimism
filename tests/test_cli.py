from __future__ import annotations

import json

import pytest

from resample_optimism.cli import (
    EXIT_BAD_CONFIG,
    EXIT_LEAK_DETECTED,
    EXIT_WITHIN_TOLERANCE,
    build_parser,
    main,
)

FAST = [
    "--n-samples", "600", "--n-features", "8", "--n-informative", "4",
    "--minority-rate", "0.15", "--n-splits", "3", "--estimator", "tree", "--seed", "7",
]


def test_parser_defaults_match_the_documented_scenario():
    args = build_parser().parse_args([])
    assert args.resampler == "smote"
    assert args.estimator == "forest"
    assert args.tolerance == 0.02


def test_a_leaking_protocol_exits_one(capsys):
    code = main([*FAST, "--tolerance", "0.01"])
    capsys.readouterr()
    assert code == EXIT_LEAK_DETECTED


def test_a_tolerance_above_the_measured_optimism_exits_zero(capsys):
    """Exit 0 and exit 1 both mean the tool worked. Only 2 and 3 mean it did not."""
    code = main([*FAST, "--tolerance", "0.99"])
    capsys.readouterr()
    assert code == EXIT_WITHIN_TOLERANCE


def test_an_impossible_scenario_exits_two_without_a_traceback(capsys):
    code = main(["--n-samples", "800", "--minority-rate", "0.005", "--n-splits", "5"])
    captured = capsys.readouterr()
    assert code == EXIT_BAD_CONFIG
    assert "invalid scenario" in captured.err
    assert "Traceback" not in captured.err


def test_json_mode_emits_parseable_output_with_every_protocol(capsys):
    main([*FAST, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in payload["results"]} == {
        "leaked_resample_then_cv", "correct_resample_in_fold", "no_resampling"
    }
    assert payload["same_model_shipped"] is True
    assert payload["config"]["seed"] == 7


def test_table_mode_names_every_protocol_and_both_columns(capsys):
    main(FAST)
    out = capsys.readouterr().out
    assert "cv estimate" in out
    assert "honest holdout" in out
    for name in ("leaked_resample_then_cv", "correct_resample_in_fold", "no_resampling"):
        assert name in out


def test_table_mode_states_whether_the_shipped_models_match(capsys):
    main(FAST)
    assert "ship an identical model: yes" in capsys.readouterr().out


def test_an_unknown_resampler_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--resampler", "smoothy"])
