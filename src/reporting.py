"""Create minimal dashboard exports and static notebook figures."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/movielens-audience-analytics-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import MIN_GENRE_SUPPORT_SHARE, ProjectPaths


def create_dashboard_exports(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
    users = pd.read_csv(paths.model / "user_segments.csv")
    base_summary = pd.read_csv(paths.model / "segment_summary.csv")
    affinity = pd.read_csv(paths.model / "segment_genre_affinity.csv")
    similarity = pd.read_csv(paths.model / "segment_similarity.csv")

    # Demographics (age, gender, occupation) pass through as descriptive
    # overlays and filters only; they were never clustering inputs. zip_code
    # stays excluded from the reporting layer, consistent with the
    # sensitive-fields-for-audit-only stance.
    dashboard_users = users[
        [
            "user_id",
            "segment",
            "segment_name",
            "rating_count",
            "mean_rating",
            "rating_std",
            "age",
            "gender",
            "occupation",
            "assignment_distance",
            "assignment_margin",
            "boundary_flag_margin_lt_0_20",
            "boundary_flag_margin_lt_0_30",
            "outlier_flag",
            "outlier_reasons",
        ]
    ].sort_values("user_id")

    dashboard_affinity = affinity.copy()
    for output_column, metric in (
        ("engagement_share_rank", "segment_engagement_share"),
        ("engagement_lift_rank", "engagement_lift"),
        ("centered_preference_rank", "user_centered_rating_preference"),
    ):
        dashboard_affinity[output_column] = (
            dashboard_affinity.groupby("segment")[metric]
            .rank(method="dense", ascending=False)
            .astype(int)
        )
    dashboard_affinity["material_support_flag"] = (
        dashboard_affinity["genre_user_support_share"] >= MIN_GENRE_SUPPORT_SHARE
    )
    dashboard_affinity["aligned_affinity_flag"] = dashboard_affinity[
        "affinity_interpretation"
    ].eq("aligned affinity")
    dashboard_affinity["engagement_tension_flag"] = dashboard_affinity[
        "affinity_interpretation"
    ].eq("high engagement / below personal norm")
    dashboard_affinity["selective_appreciation_flag"] = dashboard_affinity[
        "affinity_interpretation"
    ].eq("selective appreciation")
    dashboard_affinity = dashboard_affinity.sort_values(
        ["segment", "engagement_share_rank", "genre"]
    )

    population_users = len(dashboard_users)
    baseline = (
        dashboard_affinity[
            [
                "genre",
                "population_engagement_share",
                "population_user_centered_preference",
                "population_users_with_activity",
            ]
        ]
        .drop_duplicates()
        .sort_values("population_engagement_share", ascending=False)
    )
    baseline["population_user_support_share"] = (
        baseline["population_users_with_activity"] / population_users
    )

    supported = dashboard_affinity.loc[dashboard_affinity["material_support_flag"]].copy()
    top_engaged = (
        dashboard_affinity.sort_values(
            ["segment", "segment_engagement_share", "genre"], ascending=[True, False, True]
        )
        .groupby("segment", as_index=False)
        .first()[
            [
                "segment",
                "genre",
                "segment_engagement_share",
                "engagement_lift",
                "user_centered_rating_preference",
                "affinity_interpretation",
            ]
        ]
        .rename(
            columns={
                "genre": "top_engagement_genre",
                "segment_engagement_share": "top_engagement_share",
                "engagement_lift": "top_engagement_genre_lift",
                "user_centered_rating_preference": "top_engagement_genre_centered_preference",
                "affinity_interpretation": "top_engagement_genre_interpretation",
            }
        )
    )
    top_overindexed = (
        supported.sort_values(["segment", "engagement_lift", "genre"], ascending=[True, False, True])
        .groupby("segment", as_index=False)
        .first()[
            [
                "segment",
                "genre",
                "segment_engagement_share",
                "engagement_lift",
                "genre_user_support_share",
                "affinity_interpretation",
            ]
        ]
        .rename(
            columns={
                "genre": "top_overindexed_genre",
                "segment_engagement_share": "top_overindexed_genre_engagement_share",
                "engagement_lift": "top_overindexed_genre_lift",
                "genre_user_support_share": "top_overindexed_genre_user_support_share",
                "affinity_interpretation": "top_overindexed_genre_interpretation",
            }
        )
    )
    top_preferred = (
        supported.sort_values(
            ["segment", "user_centered_rating_preference", "genre"],
            ascending=[True, False, True],
        )
        .groupby("segment", as_index=False)
        .first()[
            [
                "segment",
                "genre",
                "segment_engagement_share",
                "user_centered_rating_preference",
                "engagement_lift",
                "genre_user_support_share",
                "affinity_interpretation",
            ]
        ]
        .rename(
            columns={
                "genre": "top_preferred_genre",
                "segment_engagement_share": "top_preferred_genre_engagement_share",
                "user_centered_rating_preference": "top_preferred_genre_centered_preference",
                "engagement_lift": "top_preferred_genre_lift",
                "genre_user_support_share": "top_preferred_genre_user_support_share",
                "affinity_interpretation": "top_preferred_genre_interpretation",
            }
        )
    )
    tension_counts = (
        dashboard_affinity.groupby("segment", as_index=False)["engagement_tension_flag"]
        .sum()
        .rename(columns={"engagement_tension_flag": "engagement_tension_genre_count"})
    )
    executive_summary = (
        base_summary.merge(top_engaged, on="segment")
        .merge(top_overindexed, on="segment")
        .merge(top_preferred, on="segment")
        .merge(tension_counts, on="segment")
        .sort_values("segment")
    )

    # Demographic overlays. The dataset skews male and toward students, so
    # segment composition should be read against the population baseline
    # rather than as absolute shares.
    segment_demographics = (
        users.groupby(["segment", "segment_name"], as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            median_age=("age", "median"),
            mean_age=("age", "mean"),
            share_male=("gender", lambda s: s.eq("M").mean()),
            share_female=("gender", lambda s: s.eq("F").mean()),
        )
        .sort_values("segment")
    )

    occupation_counts = (
        users.groupby(["segment", "segment_name", "occupation"])
        .size()
        .rename("users")
        .reset_index()
    )
    occupation_counts["occupation_share"] = occupation_counts[
        "users"
    ] / occupation_counts.groupby("segment")["users"].transform("sum")
    top_occupations = (
        occupation_counts.sort_values(
            ["segment", "users", "occupation"], ascending=[True, False, True]
        )
        .groupby("segment", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_occupations["occupation_rank"] = (
        top_occupations.groupby("segment")["users"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    population_occupations = (
        users.groupby("occupation")
        .size()
        .rename("users")
        .reset_index()
        .sort_values(["users", "occupation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    population_occupations["occupation_share"] = (
        population_occupations["users"] / population_occupations["users"].sum()
    )
    demographic_baseline = pd.DataFrame(
        [
            {
                "population_users": len(users),
                "median_age": users["age"].median(),
                "mean_age": users["age"].mean(),
                "share_male": users["gender"].eq("M").mean(),
                "share_female": users["gender"].eq("F").mean(),
                "top_occupation": population_occupations.loc[0, "occupation"],
                "top_occupation_share": population_occupations.loc[0, "occupation_share"],
            }
        ]
    )

    manifest = pd.DataFrame(
        [
            ("segment_summary.csv", "One row per segment", "Executive cards and segment comparison"),
            ("segment_genre_affinity.csv", "One row per segment and genre", "Affinity heatmaps, opportunity quadrants, genre drill-downs"),
            ("user_segments.csv", "One row per user", "Segment filters, demographic overlays, uncertainty distribution, outlier review"),
            ("genre_population_baseline.csv", "One row per genre", "Population benchmarks and tooltips"),
            ("segment_similarity.csv", "One row per unordered segment pair", "Lookalike audience comparison"),
            ("segment_demographics.csv", "One row per segment", "Descriptive demographic composition versus population baseline"),
            ("segment_top_occupations.csv", "One row per segment and top-5 occupation", "Occupation mix overlays for segment cards and drill-downs"),
            ("demographic_population_baseline.csv", "One row (population)", "Baseline for composition-versus-population demographic framing"),
            ("population_occupations.csv", "One row per occupation", "Full occupation baseline for drill-down comparison"),
        ],
        columns=["file_name", "grain", "primary_dashboard_use"],
    )

    outputs = {
        "segment_summary.csv": executive_summary,
        "segment_genre_affinity.csv": dashboard_affinity,
        "user_segments.csv": dashboard_users,
        "genre_population_baseline.csv": baseline,
        "segment_similarity.csv": similarity,
        "segment_demographics.csv": segment_demographics,
        "segment_top_occupations.csv": top_occupations,
        "demographic_population_baseline.csv": demographic_baseline,
        "population_occupations.csv": population_occupations,
        "dashboard_data_manifest.csv": manifest,
    }
    for file_name, frame in outputs.items():
        frame.to_csv(paths.dashboard / file_name, index=False)
    return outputs


def create_figures(paths: ProjectPaths) -> None:
    """Create a small set of decision-relevant static figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    features = pd.read_csv(paths.model / "user_features.csv")
    experiments = pd.read_csv(paths.model / "segmentation_experiments.csv")
    segments = pd.read_csv(paths.model / "segment_summary.csv")
    affinity = pd.read_csv(paths.model / "segment_genre_affinity.csv")
    similarity = pd.read_csv(paths.model / "segment_similarity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(features["rating_count"], bins=40, ax=axes[0], color="#3567A8")
    axes[0].set_title("User activity is strongly right-skewed")
    axes[0].set_xlabel("Ratings per user")
    sns.histplot(features["log_rating_count"], bins=30, ax=axes[1], color="#4C9F70")
    axes[1].set_title("Log activity is more suitable for distance modeling")
    axes[1].set_xlabel("log(1 + ratings per user)")
    fig.tight_layout()
    fig.savefig(paths.figures / "01_user_activity_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    selected = experiments.loc[
        experiments["population"].eq("all_users")
        & experiments["feature_set"].isin(["C_genre_activity_mean", "E_behavior_minimal"])
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=selected, x="k", y="silhouette", hue="feature_set", marker="o", ax=ax)
    ax.axvline(4, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Separation by feature set and number of segments")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()
    fig.savefig(paths.figures / "02_model_selection.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=segments, x="segment", y="user_share", hue="segment", legend=False, ax=ax)
    ax.set_title("Final audience-segment sizes")
    ax.set_ylabel("Share of users")
    ax.set_xlabel("")
    ax.set_ylim(0, max(segments["user_share"]) * 1.2)
    fig.tight_layout()
    fig.savefig(paths.figures / "03_segment_sizes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lift = affinity.pivot(index="segment", columns="genre", values="engagement_lift")
    fig, ax = plt.subplots(figsize=(14, 4.2))
    sns.heatmap(lift, center=1, cmap="RdBu", annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "Engagement lift"})
    ax.set_title("Genre engagement lift by segment (1.00 = population norm)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(paths.figures / "04_engagement_lift_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    preference = affinity.pivot(index="segment", columns="genre", values="user_centered_rating_preference")
    fig, ax = plt.subplots(figsize=(14, 4.2))
    sns.heatmap(preference, center=0, cmap="RdBu", annot=True, fmt="+.2f", ax=ax, cbar_kws={"label": "Centered preference"})
    ax.set_title("Genre rating preference relative to each user's own mean")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(paths.figures / "05_centered_preference_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    labels = sorted(segments["segment"])
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for row in similarity.itertuples(index=False):
        matrix.loc[row.segment_a, row.segment_b] = row.engagement_lift_similarity_cosine
        matrix.loc[row.segment_b, row.segment_a] = row.engagement_lift_similarity_cosine
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, vmin=-1, vmax=1, center=0, cmap="RdBu", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Lookalike similarity of genre lift patterns")
    fig.tight_layout()
    fig.savefig(paths.figures / "06_segment_similarity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

