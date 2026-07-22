"""`vk train` — build the item-item model and persist it with its incept date.

    "Time to die" is for the models we don't keep. This one we stamp with an
    incept date and file away, so the next blade runner doesn't have to
    rebuild it from memory.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from voight_kampff.cli.config import config_from_context
from voight_kampff.cli.output import ExitCode, console, emit_json, fail, working
from voight_kampff.models.artifact import ModelArtifact


def train(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="MovieLens data directory (default: config or data/raw).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Artifact destination (default: config or data/processed/model.vk).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing artifact at the output path.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Emit machine-readable JSON instead of a Rich table.",
    ),
) -> None:
    """Build the similarity matrix and persist it, stamped with its incept date.

    Reads the catalog under --data-dir, fits the item-item model, and writes a
    single artifact carrying the matrix, its movie IDs, and an incept stamp
    (dataset hash, version, and training time) for later loading.
    """
    config = config_from_context(ctx)
    resolved_data_dir = data_dir if data_dir is not None else config.data_dir
    resolved_output = Path(output if output is not None else config.artifact_path)

    if resolved_output.exists() and not force:
        fail(
            "An artifact already lives at",
            f"{resolved_output}. Pass --force to give it a new incept date.",
            ExitCode.CONFLICT,
            json_output,
        )

    try:
        with working("[bold]Training the model…", json_output):
            artifact = ModelArtifact.train(resolved_data_dir)
    except FileNotFoundError as exc:
        fail("No catalog to train on.", str(exc), ExitCode.NOT_FOUND, json_output)

    saved_path = artifact.save(resolved_output)

    stamp = artifact.incept
    if json_output:
        emit_json(
            {
                "artifact": str(saved_path),
                "incept": asdict(stamp),
            }
        )
        return

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
