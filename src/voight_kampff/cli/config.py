"""Configuration loading for the `vk` command-line interface."""
import os
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Optional

import typer

from voight_kampff.models.artifact import DEFAULT_ARTIFACT_PATH

DEFAULT_DATA_DIR = "data/raw"
DEFAULT_NUMBER = 5
_CONFIG_KEYS = {"data_dir", "artifact_path", "number"}


class ConfigError(ValueError):
    """Raised when a CLI configuration file cannot be used."""


@dataclass(frozen=True)
class CliConfig:
    """Resolved defaults shared by `vk` subcommands."""

    data_dir: str = DEFAULT_DATA_DIR
    artifact_path: Path = DEFAULT_ARTIFACT_PATH
    number: int = DEFAULT_NUMBER


def default_config_path() -> Path:
    """Return the XDG-compatible user configuration path."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "voight-kampff" / "config.toml"


def load_config(path: Optional[Path] = None) -> CliConfig:
    """Load CLI defaults from TOML, falling back when the default file is absent."""
    explicit_path = path is not None
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    if not config_path.exists():
        if explicit_path:
            raise ConfigError(f"No configuration file at {config_path}.")
        return CliConfig()

    try:
        with config_path.open("rb") as handle:
            values = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Could not read {config_path}: {exc}") from exc

    unknown = sorted(set(values) - _CONFIG_KEYS)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"Unknown configuration option(s) in {config_path}: {names}.")

    data_dir = _string_value(values, "data_dir", DEFAULT_DATA_DIR, config_path)
    artifact = _string_value(
        values,
        "artifact_path",
        str(DEFAULT_ARTIFACT_PATH),
        config_path,
    )
    number = values.get("number", DEFAULT_NUMBER)
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ConfigError(f"'number' in {config_path} must be an integer of at least 1.")

    return CliConfig(
        data_dir=str(Path(data_dir).expanduser()),
        artifact_path=Path(artifact).expanduser(),
        number=number,
    )


def config_from_context(ctx: typer.Context) -> CliConfig:
    """Return the root configuration attached to a Typer context."""
    if isinstance(ctx.obj, CliConfig):
        return ctx.obj
    return CliConfig()


def _string_value(
    values: dict,
    key: str,
    default: str,
    config_path: Path,
) -> str:
    """Read a non-empty string option from parsed TOML."""
    value = values.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' in {config_path} must be a non-empty string.")
    return value
