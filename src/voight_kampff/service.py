"""The recommendation engine shared by the `vk` CLI and the FastAPI service.

Loading the catalog and fitting the item-item model is a small ritual; keeping
it in one place means the blade runner on the command line and the server in
the Tyrell Corporation basement interrogate exactly the same memories.
"""
from dataclasses import dataclass
from typing import List, Union

import pandas as pd

from voight_kampff.data.loader import MovieDataLoader
from voight_kampff.models.collaborative_filtering import CollaborativeFilter

# How many ambiguous title matches to name before we fall back to "+N more".
_MAX_CANDIDATES_SHOWN = 10


@dataclass
class Recommendation:
    """A recommended movie with its title resolved and its similarity score."""
    movie_id: int
    title: str
    similarity_score: float


class RecommenderService:
    """Loads the catalog, fits the model, and answers recommendation queries."""

    def __init__(self, movies_df: pd.DataFrame, model: CollaborativeFilter) -> None:
        """Wrap an already-loaded catalog and a fitted model.

        Args:
            movies_df: Movie catalog with at least `movieId` and `title` columns.
            model: A fitted item-based collaborative filtering model.
        """
        self.movies_df = movies_df
        self.model = model

    @classmethod
    def from_data_dir(cls, data_dir: str) -> "RecommenderService":
        """Load the MovieLens dataset under `data_dir` and fit the model.

        Args:
            data_dir: Directory containing an `ml-latest-small/` folder of CSVs.

        Returns:
            A ready-to-query RecommenderService.
        """
        loader = MovieDataLoader(data_dir)
        loader.load_data()
        user_item_matrix, rated_movie_ids = loader.create_user_item_matrix()
        model = CollaborativeFilter()
        model.fit(user_item_matrix, rated_movie_ids)
        return cls(loader.movies_df, model)

    def title_for(self, movie_id: int) -> str:
        """Return the catalog title for a movie ID, or a placeholder if absent."""
        matches = self.movies_df.loc[self.movies_df["movieId"] == movie_id, "title"]
        return str(matches.iloc[0]) if not matches.empty else f"movie {movie_id}"

    def resolve_movie(self, query: Union[int, str]) -> int:
        """Resolve a movie ID or title to a movie ID present in the catalog.

        A numeric query is treated as a `movieId`; anything else is matched
        against titles case-insensitively — exact match first, then a unique
        substring match.

        Args:
            query: A numeric movieId or a (partial) title.

        Returns:
            The resolved movieId.

        Raises:
            ValueError: If nothing matches, or a title is ambiguous.
        """
        text = str(query).strip()

        if text.lstrip("-").isdigit():
            movie_id = int(text)
            if (self.movies_df["movieId"] == movie_id).any():
                return movie_id
            raise ValueError(f"No movie with id {movie_id} in the catalog.")

        lowered = self.movies_df["title"].str.lower()
        needle = text.lower()

        exact = self.movies_df[lowered == needle]
        if len(exact) == 1:
            return int(exact.iloc[0]["movieId"])
        if len(exact) > 1:
            raise ValueError(self._ambiguous(text, exact))

        partial = self.movies_df[lowered.str.contains(needle, regex=False)]
        if len(partial) == 1:
            return int(partial.iloc[0]["movieId"])
        if len(partial) > 1:
            raise ValueError(self._ambiguous(text, partial))

        raise ValueError(
            f"No movie found matching '{text}'. Pass a numeric movieId or a title."
        )

    def recommend(self, movie_id: int, n: int = 5) -> List[Recommendation]:
        """Return the top-`n` movies most similar to `movie_id`.

        Args:
            movie_id: The movie to base recommendations on.
            n: Number of recommendations to return.

        Returns:
            A list of Recommendation objects with titles resolved.

        Raises:
            ValueError: If the movie has no ratings and is absent from the model.
        """
        ranked = self.model.get_recommendations(movie_id, n)
        return [
            Recommendation(int(mid), self.title_for(int(mid)), float(score))
            for mid, score in ranked
        ]

    @staticmethod
    def _ambiguous(query: str, matches: pd.DataFrame) -> str:
        """Build an error message naming the ambiguous title candidates."""
        shown = matches.head(_MAX_CANDIDATES_SHOWN)
        listing = ", ".join(
            f"{int(row.movieId)}: {row.title}" for row in shown.itertuples()
        )
        extra = len(matches) - _MAX_CANDIDATES_SHOWN
        suffix = f" (+{extra} more)" if extra > 0 else ""
        return (
            f"'{query}' matches {len(matches)} movies: {listing}{suffix}. "
            "Pass a specific movieId."
        )
