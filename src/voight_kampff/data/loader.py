"""Data loading and processing module for the movie recommendation system."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# The MovieLens layout under ``data_dir``. Kept as constants so the CSV paths
# have a single source of truth — the loader reads them and the incept stamp
# hashes them.
DATASET_SUBDIR = "ml-latest-small"
MOVIES_FILENAME = "movies.csv"
RATINGS_FILENAME = "ratings.csv"


class MovieDataLoader:
    """Handles loading and processing of movie dataset."""

    def __init__(self, data_dir: str) -> None:
        """Initialize the data loader.

        Args:
            data_dir: Directory containing the movie dataset files
        """
        self.data_dir = Path(data_dir)
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None

    def dataset_files(self) -> List[Path]:
        """Return the raw dataset files that define the catalog.

        The order is fixed (movies then ratings) so a content hash taken over
        them is reproducible — the basis for a trained model's incept date.

        Returns:
            The movies.csv and ratings.csv paths, in that order.
        """
        base = self.data_dir / DATASET_SUBDIR
        return [base / MOVIES_FILENAME, base / RATINGS_FILENAME]

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load movies and ratings data from CSV files.

        Returns:
            Tuple containing movies and ratings DataFrames
        """
        movies_path, ratings_path = self.dataset_files()
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        return self.movies_df, self.ratings_df

    def create_user_item_matrix(self) -> Tuple[np.ndarray, List[int]]:
        """Create a user-item matrix for collaborative filtering.

        Only movies that have at least one rating appear as columns, so the
        movie IDs are returned alongside the matrix to keep them aligned.
        Fitting a model with IDs from movies.csv instead would misattribute
        similarity scores to the wrong movies.

        Returns:
            Tuple of (user-item ratings matrix, movie IDs aligned with its columns)
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")

        rating_pivot = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        return rating_pivot.values, rating_pivot.columns.tolist()
