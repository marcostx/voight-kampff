"""`vk interrogate` — the core command: find the movies most like one you name.

    "You sit the movie down, you ask it questions, you watch which memories
    respond."
"""
from dataclasses import asdict
from typing import Optional

import typer
from rich.table import Table

from voight_kampff.cli.config import config_from_context
from voight_kampff.cli.output import ExitCode, console, emit_json, fail, working
from voight_kampff.service import RecommenderService


def interrogate(
    ctx: typer.Context,
    movie: str = typer.Argument(
        ...,
        metavar="MOVIE",
        help="The movie to interrogate: a numeric movieId or a title.",
    ),
    number: Optional[int] = typer.Option(
        None,
        "--number",
        "-n",
        min=1,
        help="How many similar movies to return (default: config or 5).",
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="MovieLens data directory (default: config or data/raw).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Emit machine-readable JSON instead of a Rich table.",
    ),
) -> None:
    """Interrogate a movie and return the top-N most similar titles.

    MOVIE may be a numeric movieId (e.g. 1) or a title (e.g. "Toy Story
    (1995)"); titles are matched case-insensitively.
    """
    config = config_from_context(ctx)
    number = number if number is not None else config.number
    data_dir = data_dir if data_dir is not None else config.data_dir

    try:
        with working("[bold]Interrogating the catalog…", json_output):
            service = RecommenderService.from_data_dir(data_dir)
    except FileNotFoundError as exc:
        fail(
            "No catalog to interrogate.",
            str(exc),
            ExitCode.NOT_FOUND,
            json_output,
        )

    try:
        movie_id = service.resolve_movie(movie)
        recommendations = service.recommend(movie_id, number)
    except ValueError as exc:
        fail("No response.", str(exc), ExitCode.NOT_FOUND, json_output)

    subject = service.title_for(movie_id)
    if json_output:
        emit_json(
            {
                "subject": {
                    "movie_id": movie_id,
                    "title": subject,
                },
                "recommendations": [asdict(rec) for rec in recommendations],
            }
        )
        return

    table = Table(
        title=f"Movies that respond to “{subject}” (id {movie_id})",
        title_style="bold",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("movieId", justify="right")
    table.add_column("Title")
    table.add_column("Similarity", justify="right")
    for rank, rec in enumerate(recommendations, start=1):
        table.add_row(
            str(rank),
            str(rec.movie_id),
            rec.title,
            f"{rec.similarity_score:.4f}",
        )
    console.print(table)
