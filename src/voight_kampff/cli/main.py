"""The root `vk` Typer application — the scaffold for the CLI blade runner.

Every `vk` subcommand (interrogate, retire, search, train, serve) will hang
off the `app` defined here, which also carries the banner, `--help`, and
`--version`. Commands live in sibling modules and are registered below.
"""
from pathlib import Path
from typing import Optional

import typer

from voight_kampff.cli.config import ConfigError, load_config
from voight_kampff.cli.interrogate import interrogate
from voight_kampff.cli.output import ExitCode, fail
from voight_kampff.cli.serve import serve
from voight_kampff.cli.train import train
from voight_kampff.utils.version import resolve_version

# The "VK" figlet, the project name, and the tagline shown at the top of
# `vk --help`. Kept within 78 columns so it survives narrow terminals.
BANNER = r"""
 __     __ _  __
 \ \   / /| |/ /
  \ \ / / | ' /
   \ V /  | . \
    \_/   |_|\_\

 Voight-Kampff — an empathy test for your movie catalog.
"""

# Group help for `vk --help`: the banner (kept literal via Click's `\b`
# no-rewrap marker) followed by a short, reflowable orientation paragraph.
ROOT_HELP = (
    "\b"
    f"{BANNER}\n"
    "Interrogate the catalog, retire the films you're done with, and let the "
    "blade runner surface what you should watch next.\n"
    "\n"
    "Interrogation, training, and serving are live — retire and search arrive next."
)

app = typer.Typer(
    name="vk",
    help=ROOT_HELP,
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

# Subcommands live in sibling modules; register each on the app as it lands.
app.command()(interrogate)
app.command()(train)
app.command()(serve)


def _version_callback(show: bool) -> None:
    """Print the incept date (version) and exit when --version is passed."""
    if show:
        typer.echo(f"vk (voight-kampff) {resolve_version()}")
        raise typer.Exit()


@app.callback()
def root(
    ctx: typer.Context,
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        envvar="VK_CONFIG",
        help="TOML configuration file (default: ~/.config/voight-kampff/config.toml).",
    ),
    _version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the incept date (version) and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Load configuration shared by every vk subcommand."""
    try:
        ctx.obj = load_config(config_path)
    except ConfigError as exc:
        fail("Invalid configuration.", str(exc), ExitCode.CONFIG)


def main() -> None:
    """Console entry point for the `vk` command."""
    app()


if __name__ == "__main__":
    main()
