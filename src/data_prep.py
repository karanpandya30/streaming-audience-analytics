"""Ingest, audit, and minimally clean the canonical MovieLens 100K files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import ALL_SOURCE_GENRES, GENRE_COLUMNS, ProjectPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    """Read source files, preserve valid records, add auditable derived fields."""
    paths.ensure_output_directories()

    ratings = pd.read_csv(
        paths.raw / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "rating_timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "rating_timestamp": "int64"},
    )
    ratings["rating_datetime_utc"] = pd.to_datetime(
        ratings["rating_timestamp"], unit="s", utc=True
    )
    ratings = ratings.sort_values(
        ["user_id", "rating_datetime_utc", "movie_id"], kind="stable"
    ).reset_index(drop=True)

    users = pd.read_csv(
        paths.raw / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        dtype={
            "user_id": "int32",
            "age": "int16",
            "gender": "string",
            "occupation": "string",
            "zip_code": "string",
        },
        encoding="latin-1",
    ).sort_values("user_id")

    item_columns = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRE_COLUMNS,
    ]
    item_dtypes = {"movie_id": "int32", "movie_title": "string", "imdb_url": "string"}
    item_dtypes.update({column: "int8" for column in GENRE_COLUMNS})
    movies = pd.read_csv(
        paths.raw / "u.item",
        sep="|",
        names=item_columns,
        dtype=item_dtypes,
        encoding="latin-1",
    )
    movies["release_date"] = pd.to_datetime(
        movies["release_date"], format="%d-%b-%Y", errors="coerce"
    )
    movies["video_release_date"] = pd.to_datetime(
        movies["video_release_date"], format="%d-%b-%Y", errors="coerce"
    )
    movies["release_year"] = movies["release_date"].dt.year.astype("Int64")
    movies["title_year"] = pd.to_numeric(
        movies["movie_title"].str.extract(r"\((\d{4})\)\s*$", expand=False),
        errors="coerce",
    ).astype("Int64")
    movies["year_mismatch_flag"] = (
        movies["release_year"].notna()
        & movies["title_year"].notna()
        & movies["release_year"].ne(movies["title_year"])
    )
    movies["genre_count"] = movies[list(GENRE_COLUMNS)].sum(axis=1).astype("int8")
    movies = movies.sort_values("movie_id").reset_index(drop=True)

    genre_map = dict(zip(GENRE_COLUMNS, ALL_SOURCE_GENRES))
    movie_genres = movies[["movie_id", *GENRE_COLUMNS]].melt(
        id_vars="movie_id", var_name="genre_column", value_name="genre_flag"
    )
    movie_genres = movie_genres.loc[movie_genres["genre_flag"].eq(1)].copy()
    movie_genres["genre"] = movie_genres["genre_column"].map(genre_map)
    movie_genres = movie_genres[["movie_id", "genre"]].sort_values(["movie_id", "genre"])

    analytical = ratings.merge(users, on="user_id", how="left", validate="many_to_one").merge(
        movies.drop(columns=["video_release_date"]),
        on="movie_id",
        how="left",
        validate="many_to_one",
    )
    analytical["user_mean_rating"] = analytical.groupby("user_id")["rating"].transform("mean")
    analytical["rating_centered_user"] = analytical["rating"] - analytical["user_mean_rating"]

    user_activity = (
        ratings.groupby("user_id")
        .agg(
            rating_count=("rating", "size"),
            mean_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            first_rating_datetime_utc=("rating_datetime_utc", "min"),
            last_rating_datetime_utc=("rating_datetime_utc", "max"),
            unique_movies=("movie_id", "nunique"),
        )
        .reset_index()
    )
    user_activity["active_span_days"] = (
        user_activity["last_rating_datetime_utc"] - user_activity["first_rating_datetime_utc"]
    ).dt.total_seconds() / 86_400

    rating_distribution = (
        ratings["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="count")
    )
    rating_distribution["share"] = rating_distribution["count"] / len(ratings)
    rating_genres = ratings.merge(movie_genres, on="movie_id", how="left", validate="many_to_many")
    genre_summary = (
        rating_genres.groupby("genre", dropna=False)
        .agg(
            rating_memberships=("rating", "size"),
            users=("user_id", "nunique"),
            movies=("movie_id", "nunique"),
            mean_rating=("rating", "mean"),
        )
        .reset_index()
        .sort_values("rating_memberships", ascending=False)
    )

    decisions = [
        {
            "decision": "Use u.data as the canonical ratings table",
            "reason": "The supplied base/test files are overlapping predefined splits and must not be appended.",
            "effect": "The analytical population contains exactly 100,000 unique user-movie rating events.",
        },
        {
            "decision": "Retain all 943 users",
            "reason": "Every user has at least 20 ratings; unusual activity is valid behavior unless shown otherwise.",
            "effect": "Outliers are flagged and tested through sensitivity analysis rather than automatically removed.",
        },
        {
            "decision": "Treat ratings as rating activity, not confirmed viewing",
            "reason": "The data contains no starts, watch time, completion, impressions, or exposure.",
            "effect": "Engagement and campaign conclusions are framed as hypotheses.",
        },
        {
            "decision": "Exclude demographics from clustering",
            "reason": "The business question is behavioral, and demographic inclusion could create sensitive or non-actionable segments.",
            "effect": "Age, gender, occupation, and ZIP code remain available only for descriptive audit.",
        },
        {
            "decision": "Preserve movie IDs even when titles repeat",
            "reason": "The source identifies catalog records by movie_id; title-level deduplication could misassign ratings.",
            "effect": "Potential title duplicates remain reviewable without destructive cleaning.",
        },
        {
            "decision": "Use timestamps descriptively, not as viewing chronology",
            "reason": "Many histories were submitted in compressed batches.",
            "effect": "No recency, session, or retention claim is based on rating timestamps.",
        },
    ]
    profile = {
        "source": "MovieLens 100K",
        "source_url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "source_readme_sha256": _sha256(paths.raw / "README") if (paths.raw / "README").is_file() else None,
        "rows": {"ratings": len(ratings), "users": len(users), "movies": len(movies)},
        "rating_period_utc": [ratings["rating_datetime_utc"].min().isoformat(), ratings["rating_datetime_utc"].max().isoformat()],
        "missing_release_dates": int(movies["release_date"].isna().sum()),
        "duplicate_movie_title_rows": int(movies.duplicated("movie_title", keep=False).sum()),
        "release_title_year_mismatches": int(movies["year_mismatch_flag"].sum()),
        "movies_without_known_genre": int(movies[list(GENRE_COLUMNS[1:])].sum(axis=1).eq(0).sum()),
        "matrix_density": len(ratings) / (len(users) * len(movies)),
    }

    ratings.to_csv(paths.processed / "ratings.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    users.to_csv(paths.processed / "users.csv", index=False)
    movies.to_csv(paths.processed / "movies.csv", index=False, date_format="%Y-%m-%d")
    movie_genres.to_csv(paths.processed / "movie_genres.csv", index=False)
    analytical.to_csv(paths.processed / "analytical_ratings.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    user_activity.to_csv(paths.processed / "user_activity.csv", index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    rating_distribution.to_csv(paths.processed / "rating_distribution.csv", index=False)
    genre_summary.to_csv(paths.processed / "genre_summary.csv", index=False)
    (paths.processed / "cleaning_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    (paths.processed / "data_profile.json").write_text(json.dumps(profile, indent=2) + "\n")

    return {
        "ratings": ratings,
        "users": users,
        "movies": movies,
        "movie_genres": movie_genres,
        "analytical": analytical,
        "user_activity": user_activity,
        "rating_distribution": rating_distribution,
        "genre_summary": genre_summary,
    }

