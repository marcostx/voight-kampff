"""API tests via FastAPI TestClient.

Requests load the real MovieLens dataset and train the model, so these are
integration tests; run pytest from the repo root.
"""
# pylint: disable=missing-function-docstring  # test names are self-describing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

dataset_dir = Path("data/raw") / "ml-latest-small"
if not (dataset_dir / "movies.csv").exists() or not (dataset_dir / "ratings.csv").exists():
    pytest.skip(
        "Real MovieLens dataset not found at data/raw/ml-latest-small; "
        "skipping API integration tests",
        allow_module_level=True,
    )

from voight_kampff.api.server import app  # pylint: disable=wrong-import-position

client = TestClient(app)


def test_recommendations_returns_expected_schema():
    response = client.get("/recommendations/1")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
    for item in body:
        assert set(item) == {"movie_id", "title", "similarity_score"}
        assert isinstance(item["movie_id"], int)
        assert isinstance(item["title"], str)
        assert 0.0 <= item["similarity_score"] <= 1.0


def test_n_recommendations_parameter():
    response = client.get("/recommendations/1", params={"n_recommendations": 3})
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_non_positive_n_recommendations_returns_422():
    response = client.get("/recommendations/1", params={"n_recommendations": 0})
    assert response.status_code == 422


def test_unknown_movie_returns_404():
    response = client.get("/recommendations/999999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_alignment_regression_end_to_end():
    """Empathy test for issue #3: recommendations past the old divergence
    point must belong to the right movie.

    Before the fix, Monty Python and the Holy Grail (ID 1136) was matched
    with unrelated obscure titles at similarity 1.0 — including itself.
    """
    response = client.get("/recommendations/1136", params={"n_recommendations": 5})
    assert response.status_code == 200
    body = response.json()
    titles = [item["title"] for item in body]
    assert "Monty Python's Life of Brian (1979)" in titles
    assert all(item["movie_id"] != 1136 for item in body)
    assert all(item["similarity_score"] < 1.0 for item in body)
