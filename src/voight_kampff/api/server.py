"""FastAPI server implementation for the movie recommendation system."""
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

from voight_kampff.service import RecommenderService

DEFAULT_DATA_DIR = "data/raw"
app = FastAPI(title="Movie Recommender API")


class MovieRecommendation(BaseModel):
    """Pydantic model for movie recommendations."""
    movie_id: int
    title: str
    similarity_score: float


def _get_service() -> RecommenderService:
    """Return the configured service, loading the default for direct ASGI use."""
    service = getattr(app.state, "service", None)
    if service is None:
        service = RecommenderService.from_data_dir(DEFAULT_DATA_DIR)
        app.state.service = service
    return service


@app.get(
    "/recommendations/{movie_id}",
    response_model=List[MovieRecommendation]
)
async def get_recommendations(
    movie_id: int,
    n_recommendations: int = Query(5, ge=1)
) -> List[Dict]:
    """Get movie recommendations based on a movie ID.

    Args:
        movie_id: ID of the movie to base recommendations on
        n_recommendations: Number of recommendations to return

    Returns:
        List of recommended movies with their details
    """
    try:
        recommendations = _get_service().recommend(movie_id, n_recommendations)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return [
        {
            "movie_id": rec.movie_id,
            "title": rec.title,
            "similarity_score": rec.similarity_score,
        }
        for rec in recommendations
    ]


def main(data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Run the API server (console entry point)."""
    # Fit before accepting requests. Direct imports of `app` retain the same
    # default behavior, but the CLI can select a different catalog.
    app.state.service = RecommenderService.from_data_dir(data_dir)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
