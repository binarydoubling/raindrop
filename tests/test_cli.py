"""Tests for top-level CLI behavior."""

import pytest
from click.testing import CliRunner

from raindrop.cache import reset_cache
from raindrop.cli import cli


def test_favorites_alias_is_registered() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["favorites", "--help"])

    assert result.exit_code == 0
    assert "Manage favorite locations" in result.output


def test_window_aliases_are_registered() -> None:
    runner = CliRunner()

    window_result = runner.invoke(cli, ["window", "--help"])
    outside_result = runner.invoke(cli, ["outside", "--help"])

    assert window_result.exit_code == 0
    assert outside_result.exit_code == 0
    assert "Peer through a weather window" in window_result.output
    assert "Peer through a weather window" in outside_result.output


def test_no_cache_global_option_disables_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAINDROP_NO_CACHE", raising=False)
    reset_cache(None)
    runner = CliRunner()

    result = runner.invoke(cli, ["--no-cache", "config", "cache"])

    assert result.exit_code == 0
    assert "No" in result.output
    reset_cache(None)
