"""Run the complete MovieLens audience-segmentation workflow."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/movielens-audience-analytics-mpl")

from src.affinity import calculate_affinity
from src.config import ProjectPaths
from src.data_prep import prepare_data
from src.feature_engineering import build_user_features
from src.reporting import create_dashboard_exports, create_figures
from src.segmentation import fit_final_segments, run_segmentation_experiments
from src.validation import (
    validate_affinity,
    validate_all,
    validate_dashboard,
    validate_experiments,
    validate_features,
    validate_final_segments,
    validate_processed,
    validate_raw,
)


def run_pipeline() -> dict[str, object]:
    paths = ProjectPaths(Path(__file__).resolve().parent)
    paths.ensure_output_directories()
    stages = [
        ("raw data", lambda: validate_raw(paths)),
        ("ingestion and cleaning", lambda: (prepare_data(paths), validate_processed(paths))[1]),
        ("feature engineering", lambda: (build_user_features(paths), validate_features(paths))[1]),
        ("segmentation experiments", lambda: (run_segmentation_experiments(paths), validate_experiments(paths))[1]),
        ("final segmentation", lambda: (fit_final_segments(paths), validate_final_segments(paths))[1]),
        ("genre affinity", lambda: (calculate_affinity(paths), validate_affinity(paths))[1]),
        ("dashboard exports", lambda: (create_dashboard_exports(paths), validate_dashboard(paths))[1]),
        ("figures", lambda: create_figures(paths)),
    ]
    for stage_name, stage_function in stages:
        started = perf_counter()
        stage_function()
        print(f"PASS {stage_name} ({perf_counter() - started:.1f}s)")
    summary = validate_all(paths)
    print(f"PASS end-to-end validation: {summary}")
    return summary


if __name__ == "__main__":
    run_pipeline()
