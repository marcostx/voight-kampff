"""The `vk` command-line interface — Voight-Kampff on the street.

Deckard's toolkit: a Typer application whose subcommands will let you
interrogate, retire, and search the movie catalog. This package currently
holds the scaffold from the Nexus-2 milestone; the interrogation itself
arrives with the commands that follow.
"""
from voight_kampff.cli.main import app, main

__all__ = ["app", "main"]
