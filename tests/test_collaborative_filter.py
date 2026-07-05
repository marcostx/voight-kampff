"""Tests for CollaborativeFilter with known-similarity fixtures."""
import numpy as np
import pytest

from src.models.collaborative_filtering import CollaborativeFilter
from tests.conftest import UNRATED_MOVIE_ID


@pytest.fixture
def fitted_model(synthetic_loader):
    matrix, movie_ids = synthetic_loader.create_user_item_matrix()
    model = CollaborativeFilter()
    model.fit(matrix, movie_ids)
    return model


def test_similarity_matrix_is_square_and_symmetric(fitted_model):
    sim = fitted_model.item_similarity_matrix
    n = len(fitted_model.movie_ids)
    assert sim.shape == (n, n)
    assert np.allclose(sim, sim.T)
    assert np.allclose(np.diag(sim), 1.0)


def test_identical_rating_vectors_have_similarity_one(fitted_model):
    """Movies 1 and 2 were rated identically by the same users."""
    recommendations = fitted_model.get_recommendations(1, n_recommendations=1)
    top_id, top_score = recommendations[0]
    assert top_id == 2
    assert top_score == pytest.approx(1.0)


def test_orthogonal_rating_vectors_have_similarity_zero(fitted_model):
    """Movie 3 shares no raters with movie 1: cosine similarity must be 0."""
    recommendations = dict(fitted_model.get_recommendations(1, n_recommendations=3))
    assert recommendations[3] == pytest.approx(0.0)


def test_recommendations_exclude_query_movie(fitted_model):
    recommendations = fitted_model.get_recommendations(1, n_recommendations=3)
    assert all(movie_id != 1 for movie_id, _ in recommendations)


def test_n_recommendations_is_respected(fitted_model):
    assert len(fitted_model.get_recommendations(1, n_recommendations=2)) == 2


def test_unknown_movie_fails_the_test(fitted_model):
    """A movie the model has never seen cannot be interrogated."""
    with pytest.raises(ValueError, match="not found"):
        fitted_model.get_recommendations(UNRATED_MOVIE_ID)
