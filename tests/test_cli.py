"""CLI tests: the `vk` command's banner, version, help, and interrogate behavior.

These drive the Typer app directly through CliRunner, so they need no dataset
and run fast — the empathy test for the blade runner's own paperwork.
"""
# pylint: disable=missing-function-docstring  # test names are self-describing
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from voight_kampff.cli.main import app, resolve_version
from voight_kampff.models.artifact import ModelArtifact
from tests.conftest import RATED_MOVIE_IDS

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


def test_help_lists_the_serve_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.output


def test_serve_launches_the_api_server():
    with patch("voight_kampff.cli.serve._launch_server") as launch_server:
        result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    launch_server.assert_called_once_with("data/raw")


def test_serve_loads_a_custom_data_dir(synthetic_data_dir):
    with (
        patch("voight_kampff.api.server.app.state.service", create=True),
        patch("voight_kampff.api.server.uvicorn.run") as run_server,
    ):
        result = runner.invoke(
            app,
            ["serve", "--data-dir", str(synthetic_data_dir)],
        )
        launched_app = run_server.call_args.args[0]
        selected_title = launched_app.state.service.title_for(1)

    assert result.exit_code == 0
    run_server.assert_called_once()
    assert selected_title == "Blade Runner (1982)"


def test_serve_missing_dataset_exits_nonzero(tmp_path):
    result = runner.invoke(
        app,
        ["serve", "--data-dir", str(tmp_path / "nowhere")],
    )
    assert result.exit_code == 1
    assert "No catalog" in result.output


def test_interrogate_by_id_returns_similar_movies(synthetic_data_dir):
    result = runner.invoke(
        app,
        ["interrogate", "1", "-n", "1", "--data-dir", str(synthetic_data_dir)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    # Movie 2 ("Blade Runner Clone") is the identical-ratings twin of movie 1.
    assert "Clone" in result.output


def test_interrogate_by_title_resolves_and_reports(synthetic_data_dir):
    result = runner.invoke(
        app,
        ["interrogate", "Blade Runner (1982)", "--data-dir", str(synthetic_data_dir)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "Clone" in result.output


def test_interrogate_unknown_movie_exits_nonzero(synthetic_data_dir):
    result = runner.invoke(
        app,
        ["interrogate", "999", "--data-dir", str(synthetic_data_dir)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "No response" in result.output


def test_interrogate_rejects_non_positive_n(synthetic_data_dir):
    result = runner.invoke(
        app,
        ["interrogate", "1", "-n", "0", "--data-dir", str(synthetic_data_dir)],
        env={"COLUMNS": "200"},
    )
    # Typer enforces the min=1 bound before we ever touch the dataset.
    assert result.exit_code == 2


def test_train_writes_an_artifact_stamped_with_its_incept_date(
    synthetic_data_dir, tmp_path
):
    output = tmp_path / "model.vk"
    result = runner.invoke(
        app,
        ["train", "--data-dir", str(synthetic_data_dir), "-o", str(output)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert output.exists()
    assert "incept" in result.output.lower()
    # The artifact is a real, reloadable model, not just a file on disk.
    assert ModelArtifact.load(output).movie_ids == RATED_MOVIE_IDS


def test_train_refuses_to_overwrite_without_force(synthetic_data_dir, tmp_path):
    output = tmp_path / "model.vk"
    output.write_text("a prior replicant")
    result = runner.invoke(
        app,
        ["train", "--data-dir", str(synthetic_data_dir), "-o", str(output)],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "already" in result.output.lower()
    # The existing file is left untouched.
    assert output.read_text() == "a prior replicant"


def test_train_force_overwrites_existing_artifact(synthetic_data_dir, tmp_path):
    output = tmp_path / "model.vk"
    output.write_text("a prior replicant")
    result = runner.invoke(
        app,
        ["train", "--data-dir", str(synthetic_data_dir), "-o", str(output), "--force"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert ModelArtifact.load(output).movie_ids == RATED_MOVIE_IDS


def test_train_missing_dataset_exits_nonzero(tmp_path):
    result = runner.invoke(
        app,
        ["train", "--data-dir", str(tmp_path / "nowhere"), "-o", str(tmp_path / "m.vk")],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 1
    assert "No catalog" in result.output
