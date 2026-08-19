"""Calculate transparent genre affinity and segment-profile similarity."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import (
    GENRES,
    LIFT_HIGH,
    LIFT_LOW,
    PREFERENCE_HIGH,
    PREFERENCE_LOW,
    ProjectPaths,
)


def affinity_interpretation(lift: float, preference: float) -> str:
    engagement = "high" if lift >= LIFT_HIGH else "low" if lift <= LIFT_LOW else "neutral"
    satisfaction = (
        "positive"
        if preference >= PREFERENCE_HIGH
        else "negative"
        if preference <= PREFERENCE_LOW
        else "neutral"
    )
    return {
        ("high", "positive"): "aligned affinity",
        ("high", "negative"): "high engagement / below personal norm",
        ("high", "neutral"): "engagement-led affinity",
        ("low", "positive"): "selective appreciation",
        ("low", "negative"): "low engagement / below personal norm",
        ("low", "neutral"): "low engagement",
        ("neutral", "positive"): "satisfaction-led affinity",
        ("neutral", "negative"): "average engagement, lower satisfaction",
        ("neutral", "neutral"): "near population norm",
    }[(engagement, satisfaction)]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else np.nan


def calculate_affinity(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(paths.processed / "ratings.csv", usecols=["user_id", "movie_id", "rating"])
    movies = pd.read_csv(paths.processed / "movies.csv", usecols=["movie_id", "genre_count"])
    movie_genres = pd.read_csv(paths.processed / "movie_genres.csv")
    assignments = pd.read_csv(
        paths.model / "user_segments.csv",
        usecols=["user_id", "segment", "segment_name"],
    )

    user_summary = (
        ratings.groupby("user_id")
        .agg(user_rating_count=("rating", "size"), user_mean_rating=("rating", "mean"))
        .reset_index()
    )
    rating_genres = (
        ratings.merge(user_summary, on="user_id", how="left", validate="many_to_one")
        .merge(movies, on="movie_id", how="left", validate="many_to_one")
        .merge(movie_genres, on="movie_id", how="left", validate="many_to_many")
    )
    rating_genres = rating_genres.loc[rating_genres["genre"].isin(GENRES)].copy()
    rating_genres["fractional_genre_weight"] = 1 / rating_genres["genre_count"]
    rating_genres["user_centered_rating"] = rating_genres["rating"] - rating_genres["user_mean_rating"]
    rating_genres["weighted_centered_rating"] = (
        rating_genres["fractional_genre_weight"] * rating_genres["user_centered_rating"]
    )

    user_genre_observed = (
        rating_genres.groupby(["user_id", "genre"])
        .agg(
            fractional_genre_activity=("fractional_genre_weight", "sum"),
            weighted_centered_rating_sum=("weighted_centered_rating", "sum"),
            genre_rating_memberships=("rating", "size"),
        )
        .reset_index()
        .merge(user_summary[["user_id", "user_rating_count"]], on="user_id", how="left")
    )
    user_genre_observed["user_genre_engagement_share"] = (
        user_genre_observed["fractional_genre_activity"] / user_genre_observed["user_rating_count"]
    )
    user_genre_observed["user_centered_rating_preference"] = (
        user_genre_observed["weighted_centered_rating_sum"]
        / user_genre_observed["fractional_genre_activity"]
    )

    grid = pd.MultiIndex.from_product(
        [assignments["user_id"], GENRES], names=["user_id", "genre"]
    ).to_frame(index=False)
    grid = (
        grid.merge(user_genre_observed, on=["user_id", "genre"], how="left", validate="one_to_one")
        .merge(assignments, on="user_id", how="left", validate="many_to_one")
    )
    for column in (
        "fractional_genre_activity",
        "genre_rating_memberships",
        "user_genre_engagement_share",
    ):
        grid[column] = grid[column].fillna(0)
    # A missing preference means no evidence, not neutral preference. Aggregation
    # below therefore averages preference only among users who rated the genre.
    grid.to_csv(paths.model / "user_genre_components.csv", index=False)

    population = (
        grid.groupby("genre")
        .agg(
            population_engagement_share=("user_genre_engagement_share", "mean"),
            population_user_centered_preference=("user_centered_rating_preference", "mean"),
            population_users_with_activity=("user_centered_rating_preference", "count"),
        )
        .reset_index()
    )
    segment_sizes = (
        assignments.groupby(["segment", "segment_name"])
        .size()
        .rename("segment_users")
        .reset_index()
    )
    affinity = (
        grid.groupby(["segment", "segment_name", "genre"])
        .agg(
            segment_engagement_share=("user_genre_engagement_share", "mean"),
            user_centered_rating_preference=("user_centered_rating_preference", "mean"),
            users_with_genre_activity=("user_centered_rating_preference", "count"),
            fractional_genre_activity=("fractional_genre_activity", "sum"),
            genre_rating_memberships=("genre_rating_memberships", "sum"),
        )
        .reset_index()
        .merge(population, on="genre", how="left", validate="many_to_one")
        .merge(segment_sizes, on=["segment", "segment_name"], how="left", validate="many_to_one")
    )
    affinity["engagement_lift"] = (
        affinity["segment_engagement_share"] / affinity["population_engagement_share"]
    )
    affinity["genre_user_support_share"] = (
        affinity["users_with_genre_activity"] / affinity["segment_users"]
    )
    affinity["preference_difference_vs_population"] = (
        affinity["user_centered_rating_preference"]
        - affinity["population_user_centered_preference"]
    )
    affinity["affinity_interpretation"] = [
        affinity_interpretation(lift, preference)
        for lift, preference in zip(
            affinity["engagement_lift"], affinity["user_centered_rating_preference"]
        )
    ]
    affinity = affinity.sort_values(["segment", "genre"]).reset_index(drop=True)
    affinity.to_csv(paths.model / "segment_genre_affinity.csv", index=False)

    ranking_rows = []
    for segment, group in affinity.groupby("segment"):
        supported = group.loc[group["genre_user_support_share"].ge(0.20)]
        for ranking, metric, ascending in (
            ("top_engagement_share", "segment_engagement_share", False),
            ("top_engagement_lift", "engagement_lift", False),
            ("top_centered_preference", "user_centered_rating_preference", False),
            ("lowest_centered_preference", "user_centered_rating_preference", True),
        ):
            for rank, row in enumerate(
                supported.sort_values(metric, ascending=ascending).head(5).itertuples(), start=1
            ):
                ranking_rows.append(
                    {
                        "segment": segment,
                        "segment_name": row.segment_name,
                        "ranking": ranking,
                        "rank": rank,
                        "genre": row.genre,
                        "segment_engagement_share": row.segment_engagement_share,
                        "engagement_lift": row.engagement_lift,
                        "user_centered_rating_preference": row.user_centered_rating_preference,
                        "genre_user_support_share": row.genre_user_support_share,
                        "affinity_interpretation": row.affinity_interpretation,
                    }
                )
    pd.DataFrame(ranking_rows).to_csv(paths.model / "affinity_rankings.csv", index=False)

    lift_vectors = affinity.pivot(index="segment", columns="genre", values="engagement_lift") - 1.0
    preference_vectors = affinity.pivot(
        index="segment", columns="genre", values="preference_difference_vs_population"
    )
    name_map = assignments.drop_duplicates("segment").set_index("segment")["segment_name"].to_dict()
    pair_rows = []
    for segment_a, segment_b in combinations(sorted(name_map), 2):
        pair_rows.append(
            {
                "segment_a": segment_a,
                "segment_a_name": name_map[segment_a],
                "segment_b": segment_b,
                "segment_b_name": name_map[segment_b],
                "engagement_lift_similarity_cosine": cosine_similarity(
                    lift_vectors.loc[segment_a].to_numpy(),
                    lift_vectors.loc[segment_b].to_numpy(),
                ),
                "preference_delta_similarity_cosine": cosine_similarity(
                    preference_vectors.loc[segment_a].to_numpy(),
                    preference_vectors.loc[segment_b].to_numpy(),
                ),
            }
        )
    similarity = pd.DataFrame(pair_rows)
    similarity["engagement_similarity_rank"] = (
        similarity["engagement_lift_similarity_cosine"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    similarity["preference_similarity_rank"] = (
        similarity["preference_delta_similarity_cosine"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    similarity = similarity.sort_values("engagement_similarity_rank")
    similarity.to_csv(paths.model / "segment_similarity.csv", index=False)
    return affinity, similarity

