"""Test script for the movie recommendation system."""
import pandas as pd
from src.data.loader import MovieDataLoader
from src.models.collaborative_filtering import CollaborativeFilter

def main() -> None:
    """Run tests for the recommendation system components."""
    # Test data loading
    loader = MovieDataLoader('data/raw')
    movies_df, ratings_df = loader.load_data()
    print('\nSample of movies data:')
    print(movies_df.head())

    # Test user-item matrix creation
    user_item_matrix = loader.create_user_item_matrix()
    print('\nUser-item matrix shape:', user_item_matrix.shape)

    # Test model
    model = CollaborativeFilter()
    model.fit(user_item_matrix, movies_df['movieId'].tolist())

    # Get recommendations for a sample movie
    test_movie_id = movies_df['movieId'].iloc[0]
    recommendations = model.get_recommendations(test_movie_id, n_recommendations=5)
    print('\nRecommendations for movie:', movies_df[movies_df['movieId'] == test_movie_id]['title'].iloc[0])
    for movie_id, score in recommendations:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f'- {title} (similarity: {score:.3f})')

if __name__ == "__main__":
    main()
