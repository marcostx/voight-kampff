"""Shared human- and machine-readable output helpers for `vk`."""
from contextlib import contextmanager
from enum import IntEnum
import json
from typing import Iterator, NoReturn

import typer
from rich.console import Console
from rich.markup import escape

console = Console()
error_console = Console(stderr=True)


class ExitCode(IntEnum):
    """Stable process exit codes for scripts invoking `vk`."""

    SUCCESS = 0
    ERROR = 1
    USAGE = 2
    NOT_FOUND = 3
    CONFLICT = 4
    CONFIG = 5


def emit_json(payload: object) -> None:
    """Write one compact JSON value to standard output."""
    typer.echo(json.dumps(payload, ensure_ascii=False))


@contextmanager
def working(message: str, json_output: bool = False) -> Iterator[None]:
    """Show a Rich activity indicator unless stdout must remain JSON-only."""
    if json_output:
        yield
        return
    with console.status(message):
        yield


def fail(
    summary: str,
    detail: str,
    code: ExitCode,
    json_output: bool = False,
) -> NoReturn:
    """Write a structured error to stderr and terminate with a stable code."""
    message = f"{summary} {detail}".strip()
    if json_output:
        emit_json_error(code, message)
    else:
        error_console.print(f"[bold red]{escape(summary)}[/] {escape(detail)}")
    raise typer.Exit(code=int(code))


def emit_json_error(code: ExitCode, message: str) -> None:
    """Write a machine-readable error object to standard error."""
    typer.echo(
        json.dumps(
            {
                "error": {
                    "code": code.name.lower(),
                    "message": message,
                }
            },
            ensure_ascii=False,
        ),
        err=True,
    )
