"""Tests for model artifacts: dataset hashing, the incept stamp, and persistence.

These use the synthetic dataset, so they need no MovieLens download and run
fast — the empathy test for a replicant's own paperwork.
"""
# pylint: disable=missing-function-docstring, redefined-outer-name  # pytest idiom
import json

import numpy as np
import pytest

from voight_kampff.data.loader import MovieDataLoader
from voight_kampff.models.artifact import (
    ARTIFACT_FORMAT_VERSION,
    InceptStamp,
    ModelArtifact,
    hash_dataset,
)
from voight_kampff.utils.version import resolve_version
from tests.conftest import N_USERS, RATED_MOVIE_IDS

# The synthetic ratings.csv fixture carries exactly this many ratings.
N_RATINGS = 8


@pytest.fixture
def artifact(synthetic_data_dir) -> ModelArtifact:
    return ModelArtifact.train(str(synthetic_data_dir))


def test_hash_dataset_is_deterministic(synthetic_data_dir):
    files = MovieDataLoader(str(synthetic_data_dir)).dataset_files()
    assert hash_dataset(files) == hash_dataset(files)
    assert len(hash_dataset(files)) == 64  # hex-encoded SHA-256


def test_hash_dataset_changes_when_the_dataset_changes(synthetic_data_dir):
    loader = MovieDataLoader(str(synthetic_data_dir))
    before = hash_dataset(loader.dataset_files())
    ratings = synthetic_data_dir / "ml-latest-small" / "ratings.csv"
    ratings.write_text(ratings.read_text() + "3,1,4.0,964980000\n")
    assert hash_dataset(loader.dataset_files()) != before


def test_train_stamps_incept_date_and_dataset_stats(artifact):
    stamp = artifact.incept
    assert stamp.n_users == N_USERS
    assert stamp.n_movies == len(RATED_MOVIE_IDS)
    assert stamp.n_ratings == N_RATINGS
    assert stamp.package_version == resolve_version()
    assert len(stamp.dataset_hash) == 64
    assert stamp.short_hash == stamp.dataset_hash[:12]
    # trained_at is an ISO-8601 timestamp in UTC.
    assert stamp.trained_at.endswith("+00:00")


def test_train_builds_matrix_aligned_with_movie_ids(artifact):
    assert artifact.movie_ids == RATED_MOVIE_IDS
    n = len(RATED_MOVIE_IDS)
    assert artifact.item_similarity_matrix.shape == (n, n)
    # Movies 1 and 2 have identical ratings → cosine similarity 1.0.
    i, j = artifact.movie_ids.index(1), artifact.movie_ids.index(2)
    assert artifact.item_similarity_matrix[i, j] == pytest.approx(1.0)


def test_save_and_load_roundtrip(artifact, tmp_path):
    path = tmp_path / "model.vk"
    saved = artifact.save(path)
    assert saved == path
    assert path.exists()

    loaded = ModelArtifact.load(path)
    assert loaded.movie_ids == artifact.movie_ids
    assert loaded.incept == artifact.incept
    np.testing.assert_allclose(
        loaded.item_similarity_matrix, artifact.item_similarity_matrix
    )


def test_save_creates_missing_parent_directories(artifact, tmp_path):
    path = tmp_path / "nested" / "processed" / "model.vk"
    artifact.save(path)
    assert path.exists()


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model artifact"):
        ModelArtifact.load(tmp_path / "absent.vk")


def test_load_rejects_unknown_format_version(tmp_path):
    """An artifact from a newer build must be refused, not silently misread."""
    path = tmp_path / "from_the_future.vk"
    meta = {
        "format_version": ARTIFACT_FORMAT_VERSION + 1,
        "dataset_hash": "deadbeef",
        "package_version": "9.9.9",
        "trained_at": "2099-01-01T00:00:00+00:00",
        "n_users": 1,
        "n_movies": 1,
        "n_ratings": 1,
    }
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            item_similarity_matrix=np.eye(1),
            movie_ids=np.asarray([1], dtype=np.int64),
            meta=np.asarray(json.dumps(meta)),
        )
    with pytest.raises(ValueError, match="Unsupported artifact format"):
        ModelArtifact.load(path)


def test_incept_stamp_is_immutable():
    stamp = InceptStamp("abc123", "0.1.0", "2026-01-01T00:00:00+00:00", 1, 1, 1)
    with pytest.raises(Exception):  # frozen dataclass forbids reassignment
        stamp.dataset_hash = "tampered"  # type: ignore[misc]
