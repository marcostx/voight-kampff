"""Tests for CLI configuration discovery and TOML validation."""
# pylint: disable=missing-function-docstring  # test names are self-describing
from pathlib import Path

import pytest

from voight_kampff.cli.config import (
    CliConfig,
    ConfigError,
    default_config_path,
    load_config,
)


def test_missing_default_config_uses_built_in_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert load_config() == CliConfig()
    assert default_config_path() == tmp_path / "voight-kampff" / "config.toml"


def test_load_config_reads_supported_values(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'data_dir = "~/movies"\n'
        'artifact_path = "~/models/model.vk"\n'
        "number = 7\n"
    )

    config = load_config(config_path)

    assert config.data_dir == str(Path("~/movies").expanduser())
    assert config.artifact_path == Path("~/models/model.vk").expanduser()
    assert config.number == 7


def test_explicit_missing_config_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="No configuration file"):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("number = 0\n", "integer of at least 1"),
        ("number = true\n", "integer of at least 1"),
        ("data_dir = 42\n", "non-empty string"),
        ("unexpected = 1\n", "Unknown configuration option"),
        ("not valid toml", "Could not read"),
    ],
)
def test_invalid_config_is_rejected(tmp_path, contents, message):
    config_path = tmp_path / "config.toml"
    config_path.write_text(contents)

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)
