"""Data loading and processing module for the movie recommendation system."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load movies and ratings data from CSV files.

        Returns:
            Tuple containing movies and ratings DataFrames
        """
        self.movies_df = pd.read_csv(
            self.data_dir / "ml-latest-small" / "movies.csv"
        )
        self.ratings_df = pd.read_csv(
            self.data_dir / "ml-latest-small" / "ratings.csv"
        )
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
