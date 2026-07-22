"""`vk serve` — launch the FastAPI recommendation server."""
from typing import Optional

import typer

from voight_kampff.cli.config import config_from_context
from voight_kampff.cli.output import ExitCode, fail


def _launch_server(data_dir: str) -> None:
    """Load the API only when serving, then start it with the selected catalog."""
    # Keep the FastAPI and uvicorn imports out of unrelated commands such as
    # `vk --help` and `vk train`.
    # pylint: disable-next=import-outside-toplevel
    from voight_kampff.api.server import main as run_server

    run_server(data_dir)


def serve(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="MovieLens data directory (default: config or data/raw).",
    ),
) -> None:
    """Launch the FastAPI server on 0.0.0.0:8000."""
    config = config_from_context(ctx)
    resolved_data_dir = data_dir if data_dir is not None else config.data_dir
    try:
        _launch_server(resolved_data_dir)
    except FileNotFoundError as exc:
        fail("No catalog to serve.", str(exc), ExitCode.NOT_FOUND)
