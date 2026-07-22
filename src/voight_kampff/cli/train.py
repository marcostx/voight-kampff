"""`vk train` — build the item-item model and persist it with its incept date.

    "Time to die" is for the models we don't keep. This one we stamp with an
    incept date and file away, so the next blade runner doesn't have to
    rebuild it from memory.
"""
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voight_kampff.models.artifact import DEFAULT_ARTIFACT_PATH, ModelArtifact

console = Console()


def train(
    data_dir: str = typer.Option(
        "data/raw",
        "--data-dir",
        help="Directory holding the MovieLens dataset (ml-latest-small/).",
    ),
    output: Path = typer.Option(
        DEFAULT_ARTIFACT_PATH,
        "--output",
        "-o",
        help="Where to write the trained model artifact.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing artifact at the output path.",
    ),
) -> None:
    """Build the similarity matrix and persist it, stamped with its incept date.

    Reads the catalog under --data-dir, fits the item-item model, and writes a
    single artifact carrying the matrix, its movie IDs, and an incept stamp
    (dataset hash, version, and training time) for later loading.
    """
    output = Path(output)
    if output.exists() and not force:
        console.print(
            f"[bold red]An artifact already lives at[/] {output}. "
            "Pass [bold]--force[/] to give it a new incept date."
        )
        raise typer.Exit(code=1)

    try:
        with console.status("[bold]Training the model…"):
            artifact = ModelArtifact.train(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]No catalog to train on.[/] {exc}")
        raise typer.Exit(code=1) from exc

    saved_path = artifact.save(output)

    stamp = artifact.incept
    table = Table(title="A new model has been incepted", title_style="bold")
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("Artifact", str(saved_path))
    table.add_row("Incept date", stamp.trained_at)
    table.add_row("Dataset hash", stamp.short_hash)
    table.add_row("Version", stamp.package_version)
    table.add_row("Movies", f"{stamp.n_movies:,}")
    table.add_row("Users", f"{stamp.n_users:,}")
    table.add_row("Ratings", f"{stamp.n_ratings:,}")
    console.print(table)
