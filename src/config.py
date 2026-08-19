"""Central configuration and explicitly documented analytical choices."""

from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 42
FINAL_K = 4
K_VALUES = tuple(range(3, 9))
VALIDATION_REPEATS = 2
VALIDATION_FOLDS = 10
N_INIT_REFERENCE = 20
N_INIT_VALIDATION = 5
OUTLIER_IQR_MULTIPLIER = 1.5
MIN_GENRE_SUPPORT_SHARE = 0.20
BOUNDARY_MARGIN_THRESHOLDS = (0.20, 0.30)

# These are interpretation thresholds, not learned cutoffs. They are deliberately
# symmetric around the population norm and remain visible in the report.
LIFT_HIGH = 1.05
LIFT_LOW = 0.95
PREFERENCE_HIGH = 0.05
PREFERENCE_LOW = -0.05

GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

ALL_SOURCE_GENRES = ("unknown", *GENRES)

GENRE_COLUMNS = (
    "genre_unknown",
    "genre_action",
    "genre_adventure",
    "genre_animation",
    "genre_childrens",
    "genre_comedy",
    "genre_crime",
    "genre_documentary",
    "genre_drama",
    "genre_fantasy",
    "genre_film_noir",
    "genre_horror",
    "genre_musical",
    "genre_mystery",
    "genre_romance",
    "genre_sci_fi",
    "genre_thriller",
    "genre_war",
    "genre_western",
)

# Names follow a standardized 2x2 behavioral scheme matching the two clustering
# features (activity level x rating disposition). Genre descriptors are deliberately
# excluded from names: genre affinity is a post-clustering overlay, and names should
# claim only what the model measured. "Critic" means a lower rating tendency relative
# to the population norm, not a negative rater.
SEGMENT_NAMES = {
    "S1": "Heavy Critics",
    "S2": "Casual Critics",
    "S3": "Heavy Enthusiasts",
    "S4": "Casual Enthusiasts",
}


@dataclass(frozen=True)
class ProjectPaths:
    """All pipeline paths, rooted at the portable project directory."""

    root: Path

    @classmethod
    def from_file(cls, file: str | Path) -> "ProjectPaths":
        return cls(Path(file).resolve().parents[1])

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "ml-100k"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def model(self) -> Path:
        return self.root / "data" / "model"

    @property
    def dashboard(self) -> Path:
        return self.root / "data" / "final_output_csvs"

    @property
    def validation(self) -> Path:
        return self.root / "reports" / "validation"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"

    def ensure_output_directories(self) -> None:
        for directory in (
            self.processed,
            self.model,
            self.dashboard,
            self.validation,
            self.figures,
        ):
            directory.mkdir(parents=True, exist_ok=True)

