"""The paths that only run when something has already gone wrong.

These are the lines a coverage report is most likely to leave uncovered and a reviewer is
most likely to ask about, because an error path that has never executed is an error path
that does not work.
"""

from __future__ import annotations

import pytest

from resample_optimism import cli
from resample_optimism.config import ScenarioConfig
from resample_optimism.protocols import build_estimator, build_resampler


def _smuggle(config: ScenarioConfig, field: str, value: str) -> ScenarioConfig:
    """Bypass validation to reach a defensive branch.

    The dataclass is frozen and validated, so these branches are unreachable through the
    public API. They still have to work, because the alternative to an explicit raise is a
    silent fallthrough returning None into a pipeline.
    """
    object.__setattr__(config, field, value)
    return config


def test_build_resampler_raises_on_a_value_validation_should_have_caught():
    config = _smuggle(ScenarioConfig(), "resampler", "bogus")
    with pytest.raises(ValueError, match="unreachable"):
        build_resampler(config)


def test_build_estimator_raises_on_a_value_validation_should_have_caught():
    config = _smuggle(ScenarioConfig(), "estimator", "bogus")
    with pytest.raises(ValueError, match="unreachable"):
        build_estimator(config)


def test_cli_returns_three_when_the_audit_itself_fails(monkeypatch, capsys):
    def explode(_config):
        raise RuntimeError("BLAS went home")

    monkeypatch.setattr(cli, "evaluate_all", explode)
    code = cli.main(["--n-samples", "600", "--minority-rate", "0.15", "--n-splits", "3"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_INTERNAL_ERROR
    assert "audit failed to complete" in captured.err
    assert "BLAS went home" in captured.err


def test_internal_error_exit_code_is_distinct_from_the_leak_exit_code():
    """Three codes, three meanings. Collapsing any two of them makes the tool unusable in CI."""
    codes = {
        cli.EXIT_WITHIN_TOLERANCE,
        cli.EXIT_LEAK_DETECTED,
        cli.EXIT_BAD_CONFIG,
        cli.EXIT_INTERNAL_ERROR,
    }
    assert len(codes) == 4


def test_module_entry_point_is_wired(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv=None: 0)
    assert cli.main() == 0
