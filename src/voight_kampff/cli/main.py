"""The root `vk` Typer application — the scaffold for the CLI blade runner.

Every `vk` subcommand (interrogate, retire, search, train, serve) will hang
off the `app` defined here. For now this module is just the scaffold the
Nexus-2 milestone calls for: a themed banner, `--help`, and `--version`.
"""
from importlib.metadata import PackageNotFoundError, version as read_version
from typing import Optional

import typer

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
    "This is the scaffold — the subcommands arrive with the Nexus-2 milestones."
)

app = typer.Typer(
    name="vk",
    help=ROOT_HELP,
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)


def resolve_version() -> str:
    """Return the installed package version, or a placeholder when unknown."""
    try:
        return read_version("voight-kampff")
    except PackageNotFoundError:  # running from a source tree without an install
        return "0.0.0+unknown"


def _version_callback(show: bool) -> None:
    """Print the incept date (version) and exit when --version is passed."""
    if show:
        typer.echo(f"vk (voight-kampff) {resolve_version()}")
        raise typer.Exit()


@app.callback()
def root(
    _version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the incept date (version) and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Root options shared by every vk subcommand."""


def main() -> None:
    """Console entry point for the `vk` command."""
    app()


if __name__ == "__main__":
    main()
