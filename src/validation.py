"""Fail-fast validators for every pipeline stage.

Assertions are intentionally separated from transformations. A stage writes its
outputs, then its validator independently reloads and checks those outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FINAL_K, GENRES, K_VALUES, ProjectPaths, SEGMENT_NAMES


@dataclass
class ValidationReport:
    stage: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def check(self, name: str, condition: bool) -> None:
        self.checks[name] = bool(condition)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def save_and_raise_if_failed(self, path: Path) -> dict[str, object]:
        payload = {
            "stage": self.stage,
            "passed": self.passed,
            "checks": self.checks,
            "details": self.details,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        if not self.passed:
            failed = [name for name, value in self.checks.items() if not value]
            raise AssertionError(f"{self.stage} validation failed: {failed}")
        return payload


def validate_raw(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("raw_data")
    required = ["u.data", "u.user", "u.item", "u.genre", "u.info"]
    for name in required:
        report.check(f"raw_file_present_{name}", (paths.raw / name).is_file())
    if all((paths.raw / name).is_file() for name in ("u.data", "u.user", "u.item")):
        report.check("u_data_has_100000_rows", sum(1 for _ in (paths.raw / "u.data").open()) == 100_000)
        report.check("u_user_has_943_rows", sum(1 for _ in (paths.raw / "u.user").open()) == 943)
        report.check("u_item_has_1682_rows", sum(1 for _ in (paths.raw / "u.item").open(encoding="latin-1")) == 1_682)
    report.details["source"] = "MovieLens 100K canonical files"
    return report.save_and_raise_if_failed(paths.validation / "01_raw_data.json")


def validate_processed(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("processed_data")
    ratings = pd.read_csv(paths.processed / "ratings.csv")
    users = pd.read_csv(paths.processed / "users.csv", dtype={"zip_code": "string"})
    movies = pd.read_csv(paths.processed / "movies.csv")
    genres = pd.read_csv(paths.processed / "movie_genres.csv")
    analytical = pd.read_csv(paths.processed / "analytical_ratings.csv", low_memory=False)

    report.check("ratings_row_count", len(ratings) == 100_000)
    report.check("users_row_count", len(users) == 943)
    report.check("movies_row_count", len(movies) == 1_682)
    report.check("rating_user_movie_key_unique", not ratings.duplicated(["user_id", "movie_id"]).any())
    report.check("user_key_unique", users["user_id"].is_unique)
    report.check("movie_key_unique", movies["movie_id"].is_unique)
    report.check("ratings_valid", ratings["rating"].between(1, 5).all())
    report.check("all_users_joined", analytical["age"].notna().all())
    report.check("all_movies_joined", analytical["movie_title"].notna().all())
    report.check("genre_bridge_matches_flags", len(genres) == int(movies["genre_count"].sum()))
    centered_max = analytical.groupby("user_id")["rating_centered_user"].mean().abs().max()
    report.check("centered_ratings_zero_mean_per_user", centered_max < 1e-9)
    report.check("one_expected_missing_release_date", movies["release_date"].isna().sum() == 1)
    unknown_bridge_rows = genres["genre"].eq("unknown").sum()
    report.check(
        "unknown_genre_bridge_matches_source_flags",
        unknown_bridge_rows == int(movies["genre_unknown"].sum()) == 2,
    )
    report.details.update(
        {
            "rating_date_min": ratings["rating_datetime_utc"].min(),
            "rating_date_max": ratings["rating_datetime_utc"].max(),
            "movie_genre_rows": len(genres),
        }
    )
    return report.save_and_raise_if_failed(paths.validation / "02_processed_data.json")


def validate_features(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("user_features")
    features = pd.read_csv(paths.model / "user_features.csv")
    genre_columns = [column for column in features if column.startswith("genre_share_")]
    report.check("one_row_per_user", len(features) == 943 and features["user_id"].is_unique)
    report.check("expected_genre_features", len(genre_columns) == len(GENRES))
    report.check("no_model_feature_missingness", not features[["log_rating_count", "mean_rating", "rating_std", *genre_columns]].isna().any().any())
    report.check("rating_counts_at_least_20", features["rating_count"].min() >= 20)
    report.check("genre_shares_nonnegative", features[genre_columns].ge(0).all().all())
    report.check("genre_share_total_at_most_one", features[genre_columns].sum(axis=1).le(1 + 1e-10).all())
    report.check("outlier_flag_boolean", set(features["outlier_flag"].astype(str).str.lower()) <= {"true", "false"})
    report.details["outlier_users"] = int(features["outlier_flag"].sum())
    return report.save_and_raise_if_failed(paths.validation / "03_user_features.json")


def validate_experiments(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("segmentation_experiments")
    metrics = pd.read_csv(paths.model / "segmentation_experiments.csv")
    expected = 2 * 5 * len(K_VALUES)
    report.check("expected_60_experiments", len(metrics) == expected == 60)
    report.check("experiment_key_unique", not metrics.duplicated(["population", "feature_set", "k"]).any())
    report.check("all_k_values_present", set(metrics["k"]) == set(K_VALUES))
    report.check("silhouette_range", metrics["silhouette"].between(-1, 1).all())
    report.check("ari_range", metrics["cv_reference_ari_mean"].between(-1, 1).all())
    report.check("positive_distance_ratio", metrics["cv_heldout_distance_ratio_mean"].gt(0).all())
    report.check("nonempty_clusters", metrics["smallest_cluster_count"].gt(0).all())
    report.details["validation_fits_per_experiment"] = 20
    report.details["total_validation_fits"] = int(20 * len(metrics))
    return report.save_and_raise_if_failed(paths.validation / "04_segmentation_experiments.json")


def validate_final_segments(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("final_segments")
    assignments = pd.read_csv(paths.model / "user_segments.csv")
    summary = pd.read_csv(paths.model / "segment_summary.csv")
    report.check("one_assignment_per_user", len(assignments) == 943 and assignments["user_id"].is_unique)
    report.check("four_segments", set(assignments["segment"]) == set(SEGMENT_NAMES))
    report.check("all_segments_nontrivial", assignments["segment"].value_counts().min() >= 0.10 * len(assignments))
    report.check("valid_assignment_margin", assignments["assignment_margin"].between(0, 1).all())
    report.check("summary_reconciles", summary["user_count"].sum() == len(assignments))
    report.check("summary_has_final_k_rows", len(summary) == FINAL_K)
    report.details["segment_counts"] = assignments["segment"].value_counts().sort_index().to_dict()
    return report.save_and_raise_if_failed(paths.validation / "05_final_segments.json")


def validate_affinity(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("genre_affinity")
    affinity = pd.read_csv(paths.model / "segment_genre_affinity.csv")
    user_genre = pd.read_csv(paths.model / "user_genre_components.csv")
    report.check("complete_segment_genre_grid", len(affinity) == FINAL_K * len(GENRES))
    report.check("segment_genre_key_unique", not affinity.duplicated(["segment", "genre"]).any())
    report.check("complete_user_genre_grid", len(user_genre) == 943 * len(GENRES))
    report.check("lift_positive", affinity["engagement_lift"].gt(0).all())
    report.check("support_share_valid", affinity["genre_user_support_share"].between(0, 1).all())
    report.check("engagement_shares_valid", affinity["segment_engagement_share"].between(0, 1).all())
    report.check("preference_has_no_missingness", affinity["user_centered_rating_preference"].notna().all())
    report.details["interpretations"] = affinity["affinity_interpretation"].value_counts().to_dict()
    return report.save_and_raise_if_failed(paths.validation / "06_genre_affinity.json")


def validate_dashboard(paths: ProjectPaths) -> dict[str, object]:
    report = ValidationReport("dashboard_exports")
    summary = pd.read_csv(paths.dashboard / "segment_summary.csv")
    affinity = pd.read_csv(paths.dashboard / "segment_genre_affinity.csv")
    users = pd.read_csv(paths.dashboard / "user_segments.csv")
    baseline = pd.read_csv(paths.dashboard / "genre_population_baseline.csv")
    similarity = pd.read_csv(paths.dashboard / "segment_similarity.csv")
    report.check("segment_summary_grain", len(summary) == 4 and summary["segment"].is_unique)
    report.check("affinity_grain", len(affinity) == 72 and not affinity.duplicated(["segment", "genre"]).any())
    report.check("user_grain", len(users) == 943 and users["user_id"].is_unique)
    report.check("baseline_grain", len(baseline) == 18 and baseline["genre"].is_unique)
    report.check("pair_grain", len(similarity) == 6 and not similarity.duplicated(["segment_a", "segment_b"]).any())
    report.check("segment_counts_reconcile", summary["user_count"].sum() == len(users))
    report.check("user_shares_sum_one", np.isclose(summary["user_share"].sum(), 1.0))
    report.check("similarity_range", similarity[["engagement_lift_similarity_cosine", "preference_delta_similarity_cosine"]].apply(lambda s: s.between(-1, 1).all()).all())
    return report.save_and_raise_if_failed(paths.validation / "07_dashboard_exports.json")


def validate_all(paths: ProjectPaths) -> dict[str, object]:
    results = {
        "raw": validate_raw(paths),
        "processed": validate_processed(paths),
        "features": validate_features(paths),
        "experiments": validate_experiments(paths),
        "segments": validate_final_segments(paths),
        "affinity": validate_affinity(paths),
        "dashboard": validate_dashboard(paths),
    }
    summary = {
        "passed": all(result["passed"] for result in results.values()),
        "stages": {name: result["passed"] for name, result in results.items()},
    }
    (paths.validation / "00_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
