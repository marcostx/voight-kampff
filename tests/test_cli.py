"""CLI scaffold tests: the `vk` command's banner, version, and help behavior.

These drive the Typer app directly through CliRunner, so they need no dataset
and run fast — the empathy test for the blade runner's own paperwork.
"""
# pylint: disable=missing-function-docstring  # test names are self-describing
import pytest
from typer.testing import CliRunner

from voight_kampff.cli.main import app, resolve_version

runner = CliRunner()


def test_help_opens_with_the_voight_kampff_banner():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # The figlet "VK", the project name, and the tagline all sit up top.
    assert "\\ V /" in result.output
    assert "Voight-Kampff" in result.output
    assert "empathy test" in result.output


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_reports_the_incept_date(flag):
    result = runner.invoke(app, [flag])
    assert result.exit_code == 0
    assert "voight-kampff" in result.output
    assert resolve_version() in result.output


def test_bare_invocation_shows_help():
    result = runner.invoke(app, [])
    # Typer's no_args_is_help exits 2 (no command given) and prints the banner.
    assert result.exit_code == 2
    assert "Usage:" in result.output
    assert "Voight-Kampff" in result.output
