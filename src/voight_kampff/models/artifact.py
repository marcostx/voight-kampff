"""Persisting a trained model as an artifact stamped with its incept date.

A replicant has an incept date — the moment it was made and the identity it
carries out of the Tyrell Corporation. So does a trained model here: the
``InceptStamp`` records the dataset it was born from (a content hash), the
package version that built it, and the moment of training.

``vk train`` writes a ``ModelArtifact`` — the item-item similarity matrix, its
aligned movie IDs, and that stamp — to a single file, so the CLI and API can
later load a model instead of retraining from scratch on every startup.

The on-disk format is a compressed ``.npz`` (no pickling): two arrays plus a
small JSON metadata blob. ``ARTIFACT_FORMAT_VERSION`` guards against reading a
layout this build doesn't understand.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np

from voight_kampff.data.loader import MovieDataLoader
from voight_kampff.models.collaborative_filtering import CollaborativeFilter
from voight_kampff.utils.version import resolve_version

# Bump when the on-disk payload layout changes in a backward-incompatible way.
ARTIFACT_FORMAT_VERSION = 1

# Hash files in fixed-size chunks so we never load a whole CSV into memory.
_HASH_CHUNK_BYTES = 65536

# The conventional artifact location, alongside other processed data.
DEFAULT_ARTIFACT_PATH = Path("data/processed/model.vk")

PathLike = Union[str, Path]


def hash_dataset(paths: Iterable[Path]) -> str:
    """Return a SHA-256 over the given files' names and contents.

    The digest is order- and name-sensitive, so swapping or renaming a dataset
    file yields a different incept date even when the bytes are shared.

    Args:
        paths: Files to fold into the digest, in a stable order.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda h=handle: h.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InceptStamp:
    """The identity of a trained model: when, and from what, it was born."""

    dataset_hash: str
    package_version: str
    trained_at: str
    n_users: int
    n_movies: int
    n_ratings: int

    @property
    def short_hash(self) -> str:
        """The first 12 characters of the dataset hash, for display."""
        return self.dataset_hash[:12]


@dataclass
class ModelArtifact:
    """A trained item-item model plus the incept stamp identifying it."""

    item_similarity_matrix: np.ndarray
    movie_ids: List[int]
    incept: InceptStamp

    @classmethod
    def train(cls, data_dir: PathLike) -> "ModelArtifact":
        """Build the item-item similarity matrix from the dataset under ``data_dir``.

        Args:
            data_dir: Directory containing an ``ml-latest-small/`` folder of CSVs.

        Returns:
            A ready-to-persist artifact stamped with its incept date.

        Raises:
            FileNotFoundError: If the dataset CSVs are missing.
        """
        loader = MovieDataLoader(str(data_dir))
        loader.load_data()
        user_item_matrix, movie_ids = loader.create_user_item_matrix()

        model = CollaborativeFilter()
        model.fit(user_item_matrix, movie_ids)

        stamp = InceptStamp(
            dataset_hash=hash_dataset(loader.dataset_files()),
            package_version=resolve_version(),
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_users=int(user_item_matrix.shape[0]),
            n_movies=len(movie_ids),
            n_ratings=int(len(loader.ratings_df)),
        )
        return cls(model.item_similarity_matrix, list(movie_ids), stamp)

    def save(self, path: PathLike) -> Path:
        """Persist the artifact to ``path``, creating parent directories.

        Args:
            path: Destination file for the compressed artifact.

        Returns:
            The path written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"format_version": ARTIFACT_FORMAT_VERSION, **asdict(self.incept)}
        # Write through a handle so the exact filename is honored (np.savez
        # would otherwise append a .npz suffix).
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle,
                item_similarity_matrix=self.item_similarity_matrix,
                movie_ids=np.asarray(self.movie_ids, dtype=np.int64),
                meta=np.asarray(json.dumps(meta)),
            )
        return path

    @classmethod
    def load(cls, path: PathLike) -> "ModelArtifact":
        """Load an artifact previously written by :meth:`save`.

        Args:
            path: The artifact file to read.

        Returns:
            The reconstructed artifact.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the artifact's format version is not understood.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No model artifact at {path}.")

        with np.load(path, allow_pickle=False) as data:
            matrix = np.asarray(data["item_similarity_matrix"])
            movie_ids = np.asarray(data["movie_ids"], dtype=int).tolist()
            meta = json.loads(str(data["meta"]))

        version = meta.pop("format_version", None)
        if version != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported artifact format {version!r}; "
                f"this build reads version {ARTIFACT_FORMAT_VERSION}. Retrain with vk train."
            )
        return cls(matrix, movie_ids, InceptStamp(**meta))
