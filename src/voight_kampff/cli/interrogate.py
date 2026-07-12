"""`vk interrogate` — the core command: find the movies most like one you name.

    "You sit the movie down, you ask it questions, you watch which memories
    respond."
"""
import typer
from rich.console import Console
from rich.table import Table

from voight_kampff.service import RecommenderService

console = Console()


def interrogate(
    movie: str = typer.Argument(
        ...,
        metavar="MOVIE",
        help="The movie to interrogate: a numeric movieId or a title.",
    ),
    number: int = typer.Option(
        5,
        "--number",
        "-n",
        min=1,
        help="How many similar movies to bring in for questioning.",
    ),
    data_dir: str = typer.Option(
        "data/raw",
        "--data-dir",
        help="Directory holding the MovieLens dataset (ml-latest-small/).",
    ),
) -> None:
    """Interrogate a movie and return the top-N most similar titles.

    MOVIE may be a numeric movieId (e.g. 1) or a title (e.g. "Toy Story
    (1995)"); titles are matched case-insensitively.
    """
    try:
        with console.status("[bold]Interrogating the catalog…"):
            service = RecommenderService.from_data_dir(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]No catalog to interrogate.[/] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        movie_id = service.resolve_movie(movie)
        recommendations = service.recommend(movie_id, number)
    except ValueError as exc:
        console.print(f"[bold red]No response.[/] {exc}")
        raise typer.Exit(code=1) from exc

    subject = service.title_for(movie_id)
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
