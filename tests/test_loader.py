"""Tests for MovieDataLoader: shape checks and matrix/ID alignment."""
import numpy as np
import pandas as pd
import pytest

from src.data.loader import MovieDataLoader
from tests.conftest import N_USERS, RATED_MOVIE_IDS, UNRATED_MOVIE_ID


def test_load_data_returns_movies_and_ratings(synthetic_loader):
    movies_df, ratings_df = synthetic_loader.movies_df, synthetic_loader.ratings_df
    assert list(movies_df.columns) == ["movieId", "title", "genres"]
    assert {"userId", "movieId", "rating"} <= set(ratings_df.columns)
    assert len(movies_df) == 5
    assert len(ratings_df) == 8


def test_matrix_requires_loaded_data():
    loader = MovieDataLoader("data/raw")
    with pytest.raises(ValueError, match="load_data"):
        loader.create_user_item_matrix()


def test_matrix_shape_matches_returned_ids(synthetic_loader):
    matrix, movie_ids = synthetic_loader.create_user_item_matrix()
    assert matrix.shape == (N_USERS, len(movie_ids))


def test_matrix_ids_exclude_unrated_movies(synthetic_loader):
    """Regression test for the ID misalignment bug (issue #3 / PR #35).

    Movies without ratings do not appear as pivot columns, so the IDs the
    loader returns must be the pivot's own columns — not movies.csv. If a
    movie without ratings sneaks into the ID list, every movie after it is
    shifted and similarity scores get attributed to the wrong movies.
    """
    _, movie_ids = synthetic_loader.create_user_item_matrix()
    assert movie_ids == RATED_MOVIE_IDS
    assert UNRATED_MOVIE_ID not in movie_ids


def test_matrix_values_match_ratings(synthetic_loader):
    matrix, movie_ids = synthetic_loader.create_user_item_matrix()
    # User 1 rated movie 1 with 5.0; user 3 never rated it (filled with 0).
    col = movie_ids.index(1)
    assert matrix[0, col] == 5.0
    assert matrix[2, col] == 0.0


def test_real_dataset_alignment(real_matrix_and_ids, real_loader):
    """On the real dataset the matrix has one column per *rated* movie."""
    matrix, movie_ids = real_matrix_and_ids
    rated_in_csv = real_loader.ratings_df["movieId"].nunique()
    assert matrix.shape[1] == len(movie_ids) == rated_in_csv
    # Returned IDs must be a subset of the catalog, in pivot (sorted) order.
    catalog = set(real_loader.movies_df["movieId"])
    assert set(movie_ids) <= catalog
    assert movie_ids == sorted(movie_ids)
