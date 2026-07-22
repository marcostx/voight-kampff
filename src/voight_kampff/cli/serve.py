"""`vk serve` — launch the FastAPI recommendation server."""
import typer
from rich.console import Console


console = Console()


def _launch_server(data_dir: str) -> None:
    """Load the API only when serving, then start it with the selected catalog."""
    # Keep the FastAPI and uvicorn imports out of unrelated commands such as
    # `vk --help` and `vk train`.
    # pylint: disable-next=import-outside-toplevel
    from voight_kampff.api.server import main as run_server

    run_server(data_dir)


def serve(
    data_dir: str = typer.Option(
        "data/raw",
        "--data-dir",
        help="Directory holding the MovieLens dataset (ml-latest-small/).",
    ),
) -> None:
    """Launch the FastAPI server on 0.0.0.0:8000."""
    try:
        _launch_server(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]No catalog to serve.[/] {exc}")
        raise typer.Exit(code=1) from exc
