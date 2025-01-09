"""Collaborative filtering recommendation model implementation."""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilter:
    """Item-based collaborative filtering recommendation model."""
    
    def __init__(self) -> None:
        """Initialize the collaborative filtering model."""
        self.item_similarity_matrix: np.ndarray = np.array([])
        self.movie_ids: List[int] = []
        
    def fit(self, user_item_matrix: np.ndarray, movie_ids: List[int]) -> None:
        """Train the model by computing item-item similarity matrix.
        
        Args:
            user_item_matrix: Matrix of user-item ratings
            movie_ids: List of movie IDs corresponding to matrix columns
        """
        self.item_similarity_matrix = cosine_similarity(user_item_matrix.T)
        self.movie_ids = movie_ids
        
    def get_recommendations(
        self,
        movie_id: int,
        n_recommendations: int = 5
    ) -> List[Tuple[int, float]]:
        """Get movie recommendations based on a movie ID.
        
        Args:
            movie_id: ID of the movie to base recommendations on
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples containing (movie_id, similarity_score)
        """
        if movie_id not in self.movie_ids:
            raise ValueError(f"Movie ID {movie_id} not found in training data")
            
        movie_idx = self.movie_ids.index(movie_id)
        movie_similarities = self.item_similarity_matrix[movie_idx]
        
        # Get indices of most similar movies (excluding the input movie)
        similar_indices = np.argsort(movie_similarities)[::-1][1:n_recommendations+1]
        
        # Return movie IDs and similarity scores
        recommendations = [
            (self.movie_ids[idx], float(movie_similarities[idx]))
            for idx in similar_indices
        ]
        
        return recommendations
