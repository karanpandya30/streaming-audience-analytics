"""Build the deliberately minimal user-level feature candidates."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import GENRES, OUTLIER_IQR_MULTIPLIER, ProjectPaths


def genre_slug(genre: str) -> str:
    return genre.lower().replace("'", "").replace("-", "_").replace(" ", "_")


def build_user_features(paths: ProjectPaths) -> pd.DataFrame:
    ratings = pd.read_csv(paths.processed / "ratings.csv", usecols=["user_id", "movie_id", "rating"])
    movies = pd.read_csv(paths.processed / "movies.csv", usecols=["movie_id", "genre_count"])
    movie_genres = pd.read_csv(paths.processed / "movie_genres.csv")

    user_base = (
        ratings.groupby("user_id")
        .agg(
            rating_count=("rating", "size"),
            mean_rating=("rating", "mean"),
            rating_std=("rating", "std"),
        )
        .reset_index()
    )
    user_base["log_rating_count"] = np.log1p(user_base["rating_count"])

    # Each rating contributes a total weight of one across all source genres.
    # "unknown" is excluded from interpretable features, so one affected movie
    # contributes no known-genre share rather than being relabeled speculatively.
    fractional = (
        ratings[["user_id", "movie_id"]]
        .merge(movies, on="movie_id", how="left", validate="many_to_one")
        .merge(movie_genres, on="movie_id", how="left", validate="many_to_many")
    )
    fractional = fractional.loc[fractional["genre"].isin(GENRES)].copy()
    fractional["fractional_event"] = 1 / fractional["genre_count"]
    genre_features = (
        fractional.groupby(["user_id", "genre"])["fractional_event"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=GENRES, fill_value=0)
    )
    genre_features = genre_features.div(
        user_base.set_index("user_id")["rating_count"], axis=0
    ).rename(columns=lambda value: f"genre_share_{genre_slug(value)}")

    features = user_base.set_index("user_id").join(genre_features).reset_index()
    features["outlier_flag"] = False
    features["outlier_reasons"] = ""
    threshold_rows = []
    for column in ("log_rating_count", "mean_rating", "rating_std"):
        q1 = float(features[column].quantile(0.25))
        q3 = float(features[column].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        mask = ~features[column].between(lower, upper)
        features.loc[mask, "outlier_flag"] = True
        direction = np.where(features.loc[mask, column] < lower, "low", "high")
        features.loc[mask, "outlier_reasons"] += [
            f"{column}:{value};" for value in direction
        ]
        threshold_rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "flagged_users": int(mask.sum()),
            }
        )

    genre_columns = [column for column in features if column.startswith("genre_share_")]
    feature_sets = {
        "A_genre_only": genre_columns,
        "B_genre_plus_activity": [*genre_columns, "log_rating_count"],
        "C_genre_activity_mean": [*genre_columns, "log_rating_count", "mean_rating"],
        "D_genre_activity_mean_variability": [*genre_columns, "log_rating_count", "mean_rating", "rating_std"],
        "E_behavior_minimal": ["log_rating_count", "mean_rating"],
    }
    feature_dictionary = [
        {
            "feature": "log_rating_count",
            "category": "activity",
            "definition": "Natural log of one plus the user's rating-event count.",
            "reason_included": "Represents activity while reducing long-tail leverage.",
        },
        {
            "feature": "mean_rating",
            "category": "rating behavior",
            "definition": "User's mean rating on the source 1–5 scale.",
            "reason_included": "Separates generally critical from generally positive raters.",
        },
        {
            "feature": "rating_std",
            "category": "rating behavior",
            "definition": "Within-user sample standard deviation of ratings.",
            "reason_included": "Candidate selectivity measure; tested but not automatically retained.",
        },
        {
            "feature": "genre_share_*",
            "category": "content behavior",
            "definition": "User's fractional rating activity assigned to each known genre.",
            "reason_included": "Tests whether genre composition improves segment structure.",
        },
    ]
    exclusions = [
        {
            "excluded_feature": "median_rating",
            "reason": "Overlaps strongly with mean rating and adds complexity without a distinct initial hypothesis.",
        },
        {
            "excluded_feature": "rating_entropy",
            "reason": "Another rating-spread transformation; rating standard deviation is easier to explain.",
        },
        {
            "excluded_feature": "demographics",
            "reason": "Reserved for descriptive audit; behavioral segment membership should not be determined by sensitive identity fields.",
        },
        {
            "excluded_feature": "engagement_lift",
            "reason": "Population-relative derivative of genre share; used after clustering for interpretation.",
        },
        {
            "excluded_feature": "timestamp recency",
            "reason": "Rating timestamps are often batch-submission times and are not reliable viewing chronology for all users.",
        },
    ]

    features.to_csv(paths.model / "user_features.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(paths.model / "outlier_thresholds.csv", index=False)
    pd.DataFrame(feature_dictionary).to_csv(paths.model / "feature_dictionary.csv", index=False)
    pd.DataFrame(exclusions).to_csv(paths.model / "feature_exclusions.csv", index=False)
    (paths.model / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2) + "\n")
    return features

