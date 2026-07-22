"""`vk serve` — launch the FastAPI recommendation server."""


def _launch_server() -> None:
    """Load the API only when serving, then start it with its current defaults."""
    # Importing the API loads and trains the recommender, so keep that work out
    # of unrelated commands such as `vk --help` and `vk train`.
    # pylint: disable-next=import-outside-toplevel
    from voight_kampff.api.server import main as run_server

    run_server()


def serve() -> None:
    """Launch the FastAPI server on 0.0.0.0:8000 using data/raw."""
    _launch_server()
