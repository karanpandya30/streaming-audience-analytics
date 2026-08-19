"""Run feature/K experiments, validate user stability, and fit final K=4 segments."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .config import (
    BOUNDARY_MARGIN_THRESHOLDS,
    FINAL_K,
    K_VALUES,
    N_INIT_REFERENCE,
    N_INIT_VALIDATION,
    ProjectPaths,
    RANDOM_SEED,
    SEGMENT_NAMES,
    VALIDATION_FOLDS,
    VALIDATION_REPEATS,
)


def block_weights(columns: list[str]) -> np.ndarray:
    """Give feature families comparable total Euclidean influence.

    Without weighting, 18 standardized genre shares can overwhelm one activity
    feature merely because the genre block contains more columns.
    """
    weights = np.ones(len(columns), dtype=float)
    genre_indices = [i for i, column in enumerate(columns) if column.startswith("genre_share_")]
    rating_indices = [i for i, column in enumerate(columns) if column in {"mean_rating", "rating_std"}]
    if genre_indices:
        weights[genre_indices] = 1 / math.sqrt(len(genre_indices))
    if rating_indices:
        weights[rating_indices] = 1 / math.sqrt(len(rating_indices))
    return weights


def _fit_transform(raw: np.ndarray, columns: list[str]) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw)
    weights = block_weights(columns)
    return scaler, standardized * weights, weights


def _transform(raw: np.ndarray, scaler: StandardScaler, weights: np.ndarray) -> np.ndarray:
    return scaler.transform(raw) * weights


def _fold_ids(activity: pd.Series, repeat: int) -> np.ndarray:
    """Create 10% user folds stratified by activity quintile."""
    rng = np.random.default_rng(RANDOM_SEED + repeat)
    activity_bands = pd.qcut(activity.rank(method="first"), q=5, labels=False)
    folds = np.empty(len(activity), dtype=int)
    for band in range(5):
        indices = np.flatnonzero(np.asarray(activity_bands) == band)
        rng.shuffle(indices)
        folds[indices] = np.arange(len(indices)) % VALIDATION_FOLDS
    return folds


def _assignment_diagnostics(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    closest = distances[np.arange(len(x)), order[:, 0]]
    second = distances[np.arange(len(x)), order[:, 1]]
    margin = (second - closest) / np.maximum(second, 1e-12)
    return order[:, 0], closest, margin


def _cross_validate_users(
    raw: np.ndarray,
    columns: list[str],
    activity: pd.Series,
    k: int,
    reference_labels: np.ndarray,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    weights = block_weights(columns)
    for repeat in range(VALIDATION_REPEATS):
        folds = _fold_ids(activity, repeat)
        for fold in range(VALIDATION_FOLDS):
            train = folds != fold
            test = ~train
            scaler = StandardScaler().fit(raw[train])
            train_x = _transform(raw[train], scaler, weights)
            test_x = _transform(raw[test], scaler, weights)
            all_x = _transform(raw, scaler, weights)
            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_SEED + repeat * 100 + fold,
                n_init=N_INIT_VALIDATION,
                algorithm="lloyd",
            ).fit(train_x)
            test_labels, closest, margins = _assignment_diagnostics(test_x, model.cluster_centers_)
            all_labels = model.predict(all_x)
            train_distance = np.linalg.norm(
                train_x - model.cluster_centers_[model.labels_], axis=1
            ).mean()
            records.append(
                {
                    "heldout_distance_ratio": float(closest.mean() / train_distance),
                    "heldout_margin": float(margins.mean()),
                    "reference_ari": float(adjusted_rand_score(reference_labels, all_labels)),
                    "heldout_min_cluster_share": float(
                        np.bincount(test_labels, minlength=k).min() / test.sum()
                    ),
                }
            )
    frame = pd.DataFrame(records)
    summary = {
        f"cv_{column}_mean": float(frame[column].mean()) for column in frame.columns
    }
    summary.update(
        {f"cv_{column}_std": float(frame[column].std(ddof=1)) for column in frame.columns}
    )
    return summary


def run_segmentation_experiments(paths: ProjectPaths) -> pd.DataFrame:
    """Run 60 configurations: 2 populations × 5 feature sets × 6 K values."""
    features = pd.read_csv(paths.model / "user_features.csv")
    feature_sets = json.loads((paths.model / "feature_sets.json").read_text())
    experiment_rows: list[dict[str, float | int | str]] = []

    for population_name, population_mask in {
        "all_users": np.ones(len(features), dtype=bool),
        "iqr_outliers_excluded": ~features["outlier_flag"].to_numpy(dtype=bool),
    }.items():
        population = features.loc[population_mask].reset_index(drop=True)
        for feature_set_name, columns in feature_sets.items():
            raw = population[columns].to_numpy(dtype=float)
            _, x, _ = _fit_transform(raw, columns)
            for k in K_VALUES:
                reference = KMeans(
                    n_clusters=k,
                    random_state=RANDOM_SEED,
                    n_init=N_INIT_REFERENCE,
                    algorithm="lloyd",
                ).fit(x)
                counts = np.bincount(reference.labels_, minlength=k)
                seed_aris = []
                for seed_offset in range(10):
                    single = KMeans(
                        n_clusters=k,
                        random_state=RANDOM_SEED + seed_offset,
                        n_init=1,
                        algorithm="lloyd",
                    ).fit_predict(x)
                    seed_aris.append(adjusted_rand_score(reference.labels_, single))
                cross_validation = _cross_validate_users(
                    raw,
                    columns,
                    population["rating_count"],
                    k,
                    reference.labels_,
                )
                experiment_rows.append(
                    {
                        "population": population_name,
                        "feature_set": feature_set_name,
                        "feature_count": len(columns),
                        "k": k,
                        "user_count": len(population),
                        "inertia_per_user": float(reference.inertia_ / len(population)),
                        "silhouette": float(silhouette_score(x, reference.labels_)),
                        "calinski_harabasz": float(calinski_harabasz_score(x, reference.labels_)),
                        "davies_bouldin": float(davies_bouldin_score(x, reference.labels_)),
                        "smallest_cluster_count": int(counts.min()),
                        "smallest_cluster_share": float(counts.min() / len(population)),
                        "largest_cluster_share": float(counts.max() / len(population)),
                        "seed_ari_mean": float(np.mean(seed_aris)),
                        "seed_ari_min": float(np.min(seed_aris)),
                        **cross_validation,
                    }
                )

    experiments = pd.DataFrame(experiment_rows).sort_values(
        ["population", "feature_set", "k"]
    )
    experiments.to_csv(paths.model / "segmentation_experiments.csv", index=False)
    return experiments


def _stable_k4_mapping(raw_centers: pd.DataFrame) -> dict[int, str]:
    """Map arbitrary K-means IDs to semantic IDs using activity and rating levels."""
    ordered_by_rating = raw_centers.sort_values("mean_rating").index.tolist()
    low_rating, high_rating = ordered_by_rating[:2], ordered_by_rating[2:]
    low_rating = sorted(low_rating, key=lambda index: raw_centers.loc[index, "log_rating_count"], reverse=True)
    high_rating = sorted(high_rating, key=lambda index: raw_centers.loc[index, "log_rating_count"], reverse=True)
    return {
        low_rating[0]: "S1",
        low_rating[1]: "S2",
        high_rating[0]: "S3",
        high_rating[1]: "S4",
    }


def fit_final_segments(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the selected K=4 behavioral model and save assignment diagnostics."""
    features = pd.read_csv(paths.model / "user_features.csv")
    users = pd.read_csv(paths.processed / "users.csv", dtype={"zip_code": "string"})
    columns = ["log_rating_count", "mean_rating"]
    raw = features[columns].to_numpy(dtype=float)
    scaler, x, weights = _fit_transform(raw, columns)
    model = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_SEED,
        n_init=50,
        algorithm="lloyd",
    ).fit(x)
    raw_centers_array = scaler.inverse_transform(model.cluster_centers_ / weights)
    raw_centers = pd.DataFrame(raw_centers_array, columns=columns)
    mapping = _stable_k4_mapping(raw_centers)
    raw_labels, closest, margins = _assignment_diagnostics(x, model.cluster_centers_)

    assignments = features.copy()
    assignments["segment"] = pd.Series(raw_labels).map(mapping)
    assignments["segment_name"] = assignments["segment"].map(SEGMENT_NAMES)
    assignments["assignment_distance"] = closest
    assignments["assignment_margin"] = margins
    assignments["boundary_flag_margin_lt_0_20"] = margins < BOUNDARY_MARGIN_THRESHOLDS[0]
    assignments["boundary_flag_margin_lt_0_30"] = margins < BOUNDARY_MARGIN_THRESHOLDS[1]
    assignments = assignments.merge(users, on="user_id", how="left", validate="one_to_one")
    assignments.to_csv(paths.model / "user_segments.csv", index=False)

    summary = (
        assignments.groupby(["segment", "segment_name"], as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            median_rating_count=("rating_count", "median"),
            mean_rating_count=("rating_count", "mean"),
            mean_user_rating=("mean_rating", "mean"),
            mean_rating_variability=("rating_std", "mean"),
            median_assignment_margin=("assignment_margin", "median"),
            boundary_user_count_margin_lt_0_20=("boundary_flag_margin_lt_0_20", "sum"),
            boundary_user_count_margin_lt_0_30=("boundary_flag_margin_lt_0_30", "sum"),
            outlier_user_count=("outlier_flag", "sum"),
        )
        .sort_values("segment")
    )
    summary["user_share"] = summary["user_count"] / len(assignments)
    summary["boundary_user_share_margin_lt_0_20"] = summary["boundary_user_count_margin_lt_0_20"] / summary["user_count"]
    summary["boundary_user_share_margin_lt_0_30"] = summary["boundary_user_count_margin_lt_0_30"] / summary["user_count"]
    summary["outlier_user_share"] = summary["outlier_user_count"] / summary["user_count"]
    summary.to_csv(paths.model / "segment_summary.csv", index=False)

    center_rows = []
    for raw_cluster, segment in mapping.items():
        center_rows.append(
            {
                "raw_cluster": raw_cluster,
                "segment": segment,
                "segment_name": SEGMENT_NAMES[segment],
                "log_rating_count_center": raw_centers.loc[raw_cluster, "log_rating_count"],
                "mean_rating_center": raw_centers.loc[raw_cluster, "mean_rating"],
            }
        )
    pd.DataFrame(center_rows).sort_values("segment").to_csv(paths.model / "final_cluster_centers.csv", index=False)

    experiments = pd.read_csv(paths.model / "segmentation_experiments.csv")
    comparison = experiments.loc[
        experiments["population"].eq("all_users")
        & experiments["feature_set"].eq("E_behavior_minimal")
        & experiments["k"].isin([3, 4]),
        [
            "k",
            "silhouette",
            "seed_ari_mean",
            "cv_reference_ari_mean",
            "smallest_cluster_share",
            "cv_heldout_distance_ratio_mean",
        ],
    ].copy()
    comparison["selected"] = comparison["k"].eq(FINAL_K)
    comparison["decision_note"] = np.where(
        comparison["k"].eq(FINAL_K),
        "Selected: strong user-fold stability and actionable high-activity rating-style split; lower single-start stability is mitigated with 50 starts.",
        "Statistical baseline: slightly higher silhouette but combines high-activity critical and positive users.",
    )
    comparison.to_csv(paths.model / "k3_k4_decision.csv", index=False)

    metadata = {
        "algorithm": "sklearn.cluster.KMeans",
        "final_k": FINAL_K,
        "features": columns,
        "scaling": "population z-score; no learned weighting needed because each feature is a separate family",
        "random_seed": RANDOM_SEED,
        "n_init": 50,
        "population": "all 943 users; IQR outliers retained and separately flagged",
        "selection_reason": "K=4 gives actionable differentiation and strong user-fold stability. K=3 has better silhouette and single-start stability and remains the statistical parsimony benchmark; the final K=4 fit uses 50 starts to mitigate initialization sensitivity.",
        "important_limitation": "Clusters describe rating activity and rating tendency; they do not predict retention or campaign response.",
    }
    (paths.model / "final_model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return assignments, summary
