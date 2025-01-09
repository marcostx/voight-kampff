"""Data loading and processing module for the movie recommendation system."""
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

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
    
    def create_user_item_matrix(self) -> np.ndarray:
        """Create a user-item matrix for collaborative filtering.
        
        Returns:
            numpy array containing user-item ratings matrix
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
            
        user_item_matrix = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0).values
        
        return user_item_matrix
