"""Shared fixtures for the Voight-Kampff test suite."""
# pylint: disable=redefined-outer-name  # fixtures consuming fixtures is the pytest idiom
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from voight_kampff.data.loader import MovieDataLoader

# Synthetic catalog: movie 4 ("Phantom Replicant") deliberately has no
# ratings, so it must be absent from the user-item matrix columns. This is
# the exact scenario behind the ID misalignment bug fixed in PR #35.
MOVIES_CSV = """movieId,title,genres
1,Blade Runner (1982),Sci-Fi|Thriller
2,Blade Runner Clone (1982),Sci-Fi|Thriller
3,Singin' in the Rain (1952),Comedy|Musical
4,Phantom Replicant (2049),Sci-Fi
5,Brazil (1985),Fantasy|Sci-Fi
"""

# Ratings designed for known similarities: movies 1 and 2 have identical
# rating vectors (cosine = 1.0) and movie 3 is orthogonal to both (cosine = 0).
RATINGS_CSV = """userId,movieId,rating,timestamp
1,1,5.0,964982703
1,2,5.0,964982931
1,5,4.0,964982224
2,1,3.0,964983815
2,2,3.0,964982931
2,5,2.0,964980868
3,3,4.0,964982176
3,5,1.0,964984041
"""

RATED_MOVIE_IDS = [1, 2, 3, 5]
UNRATED_MOVIE_ID = 4
N_USERS = 3


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Write the synthetic MovieLens-shaped dataset to a temp directory."""
    dataset_dir = tmp_path / "ml-latest-small"
    dataset_dir.mkdir()
    (dataset_dir / "movies.csv").write_text(MOVIES_CSV)
    (dataset_dir / "ratings.csv").write_text(RATINGS_CSV)
    return tmp_path


@pytest.fixture
def synthetic_loader(synthetic_data_dir: Path) -> MovieDataLoader:
    """A loader with the synthetic dataset already loaded."""
    loader = MovieDataLoader(str(synthetic_data_dir))
    loader.load_data()
    return loader


@pytest.fixture(scope="session")
def real_loader() -> MovieDataLoader:
    """A loader over a local MovieLens dataset, if available."""
    dataset_dir = Path("data/raw") / "ml-latest-small"
    if not (dataset_dir / "movies.csv").exists() or not (dataset_dir / "ratings.csv").exists():
        pytest.skip(
            "Real MovieLens dataset not found at data/raw/ml-latest-small; "
            "skipping integration tests"
        )
    loader = MovieDataLoader("data/raw")
    loader.load_data()
    return loader


@pytest.fixture(scope="session")
def real_matrix_and_ids(
    real_loader: MovieDataLoader,
) -> Tuple[np.ndarray, List[int]]:
    """The real user-item matrix and its aligned movie IDs."""
    return real_loader.create_user_item_matrix()
