"""FastAPI server implementation for the movie recommendation system."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.data.loader import MovieDataLoader
from src.models.collaborative_filtering import CollaborativeFilter

app = FastAPI(title="Movie Recommender API")

# Initialize data and model
data_loader = MovieDataLoader("data/raw")
movies_df, ratings_df = data_loader.load_data()
user_item_matrix = data_loader.create_user_item_matrix()

model = CollaborativeFilter()
model.fit(user_item_matrix, movies_df['movieId'].tolist())

class MovieRecommendation(BaseModel):
    """Pydantic model for movie recommendations."""
    movie_id: int
    title: str
    similarity_score: float

@app.get(
    "/recommendations/{movie_id}",
    response_model=List[MovieRecommendation]
)
async def get_recommendations(
    movie_id: int,
    n_recommendations: int = 5
) -> List[Dict]:
    """Get movie recommendations based on a movie ID.
    
    Args:
        movie_id: ID of the movie to base recommendations on
        n_recommendations: Number of recommendations to return
        
    Returns:
        List of recommended movies with their details
    """
    try:
        recommendations = model.get_recommendations(
            movie_id,
            n_recommendations
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
    # Get movie titles for recommendations
    result = []
    for rec_id, score in recommendations:
        movie_title = movies_df[
            movies_df['movieId'] == rec_id
        ]['title'].iloc[0]
        result.append({
            "movie_id": rec_id,
            "title": movie_title,
            "similarity_score": score
        })
        
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
