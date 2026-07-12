"""Tests for RecommenderService: movie resolution and enriched recommendations."""
# pylint: disable=missing-function-docstring, redefined-outer-name  # pytest idiom
import pytest

from voight_kampff.service import RecommenderService
from tests.conftest import UNRATED_MOVIE_ID


@pytest.fixture
def service(synthetic_data_dir):
    return RecommenderService.from_data_dir(str(synthetic_data_dir))


def test_resolve_numeric_id_as_int_or_string(service):
    assert service.resolve_movie(1) == 1
    assert service.resolve_movie("1") == 1


def test_resolve_exact_title_is_case_insensitive(service):
    assert service.resolve_movie("blade runner (1982)") == 1


def test_resolve_unique_substring(service):
    # Only "Singin' in the Rain (1952)" contains "singin".
    assert service.resolve_movie("singin") == 3


def test_resolve_ambiguous_title_lists_candidates(service):
    # "Blade Runner" is a substring of both movie 1 and movie 2.
    with pytest.raises(ValueError, match="matches 2 movies"):
        service.resolve_movie("Blade Runner")


def test_resolve_unknown_title(service):
    with pytest.raises(ValueError, match="No movie found"):
        service.resolve_movie("Nonexistent Replicant")


def test_resolve_unknown_id(service):
    with pytest.raises(ValueError, match="in the catalog"):
        service.resolve_movie(9999)


def test_recommend_enriches_titles_and_scores(service):
    recommendations = service.recommend(1, n=1)
    assert len(recommendations) == 1
    top = recommendations[0]
    assert top.movie_id == 2
    assert top.title == "Blade Runner Clone (1982)"
    assert top.similarity_score == pytest.approx(1.0)


def test_recommend_on_unrated_movie_fails(service):
    """Phantom Replicant (id 4) has no ratings, so it's absent from the model."""
    resolved = service.resolve_movie(UNRATED_MOVIE_ID)
    assert resolved == UNRATED_MOVIE_ID
    with pytest.raises(ValueError, match="not found"):
        service.recommend(resolved)


@pytest.mark.parametrize("bad_n", [0, -1])
def test_recommend_rejects_non_positive_n(service, bad_n):
    # A negative n would otherwise slice out almost the whole catalog.
    with pytest.raises(ValueError, match="at least 1"):
        service.recommend(1, n=bad_n)
