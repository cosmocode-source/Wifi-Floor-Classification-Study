"""Shared configuration, constants, and path contracts for the WiFi Floor Classification project."""

from __future__ import annotations

from pathlib import Path

# Repository root and standard directory locations
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "Dataset"
# Compatibility aliases; all dataset defaults now resolve to Dataset, not data/raw.
DATA_DIR: Path = DATASET_DIR
RAW_DATA_DIR: Path = DATASET_DIR
LEGACY_DATASET_DIR: Path = DATASET_DIR
DEFAULT_TRAINING_CSV: Path = DATASET_DIR / "trainingData.csv"
DEFAULT_VALIDATION_CSV: Path = DATASET_DIR / "validationData.csv"

SHARED_DIR: Path = PROJECT_ROOT / "shared"
MANIFESTS_DIR: Path = SHARED_DIR / "manifests"
DEFAULT_SPLIT_MANIFEST_CSV: Path = MANIFESTS_DIR / "development_splits.csv"
DEFAULT_SPLIT_MANIFEST_META_JSON: Path = MANIFESTS_DIR / "development_splits_metadata.json"
DEFAULT_DATA_AUDIT_JSON: Path = MANIFESTS_DIR / "dataset_audit_report.json"

MODELS_DIR: Path = PROJECT_ROOT / "models"
COMPARISON_DIR: Path = PROJECT_ROOT / "comparison"
COMPARISON_RESULTS_DIR: Path = COMPARISON_DIR / "results"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
PRESENTATION_DIR: Path = PROJECT_ROOT / "presentation"

# Predictor and metadata column contracts
NUM_WAP_FEATURES: int = 520
WAP_COLUMNS: tuple[str, ...] = tuple(f"WAP{i:03d}" for i in range(1, NUM_WAP_FEATURES + 1))
WAP_COLUMN_SET: frozenset[str] = frozenset(WAP_COLUMNS)

TARGET_COLUMN: str = "FLOOR"
BUILDING_COLUMN: str = "BUILDINGID"
ROW_ID_COLUMN: str = "row_id"

METADATA_COLUMNS: tuple[str, ...] = (
    "LONGITUDE",
    "LATITUDE",
    "FLOOR",
    "BUILDINGID",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
)
NON_PREDICTOR_METADATA_COLUMNS: tuple[str, ...] = (
    "LONGITUDE",
    "LATITUDE",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
)
FORBIDDEN_PREDICTOR_COLUMNS: frozenset[str] = frozenset(
    (*METADATA_COLUMNS, ROW_ID_COLUMN, "source_row_index", "split_role", "fold", "fold_3", "fold_5")
)

EXPECTED_COLUMNS_ORDERED: tuple[str, ...] = (*WAP_COLUMNS, *METADATA_COLUMNS)
EXPECTED_TOTAL_COLUMNS: int = len(EXPECTED_COLUMNS_ORDERED)  # 529

# Expected public UJIIndoorLoc specifications (reference expectations, not hardcoded local claims)
EXPECTED_TRAINING_ROWS: int = 19937
EXPECTED_VALIDATION_ROWS: int = 1111
EXPECTED_BUILDINGS: tuple[int, ...] = (0, 1, 2)
EXPECTED_FLOORS_GLOBAL: tuple[int, ...] = (0, 1, 2, 3, 4)

# Signal strength and preprocessing contracts
UNDETECTED_SENTINEL: int = 100
# NOTE: -110.0 is our documented project assumption for undetected WAPs (weaker than -104 dBm min observed),
# NOT a verified setting from the reference paper.
DEFAULT_UNDETECTED_REPLACEMENT: float = -110.0
MIN_OBSERVED_RSSI: float = -104.0
MAX_OBSERVED_RSSI: float = 0.0

# Split and evaluation contracts
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_INTERNAL_HOLDOUT_FRACTION: float = 0.20
DEFAULT_LIGHTWEIGHT_CV_FOLDS: int = 3
DEFAULT_FINAL_CV_FOLDS: int = 5

SPLIT_ROLE_DEV_TRAIN: str = "dev_train"
SPLIT_ROLE_INTERNAL_HOLDOUT: str = "internal_holdout"
SPLIT_ROLE_OFFICIAL_VALIDATION: str = "official_validation"

EVALUATION_ROLE_CV_VAL: str = "cv_val"
EVALUATION_ROLE_INTERNAL_HOLDOUT: str = "internal_holdout"
EVALUATION_ROLE_OFFICIAL_VALIDATION: str = "official_validation"
VALID_EVALUATION_ROLES: tuple[str, ...] = (
    EVALUATION_ROLE_CV_VAL,
    EVALUATION_ROLE_INTERNAL_HOLDOUT,
    EVALUATION_ROLE_OFFICIAL_VALIDATION,
)

MODEL_ID_PCA_KNN: str = "pca_knn"
MODEL_ID_PCA_WEIGHTED_KNN: str = "pca_weighted_knn"
MODEL_ID_GRADIENT_BOOSTING: str = "gradient_boosting"
VALID_MODEL_IDS: tuple[str, ...] = (
    MODEL_ID_PCA_KNN,
    MODEL_ID_PCA_WEIGHTED_KNN,
    MODEL_ID_GRADIENT_BOOSTING,
)

RESULT_SCHEMA_VERSION: str = "1.0"
VALID_RUN_STATUSES: tuple[str, ...] = ("completed", "failed", "skipped", "unexecuted")

# Fixed label policy for pooled macro-F1 across buildings:
# UJIIndoorLoc buildings have varying floor counts (0..3 or 0..4).
# When pooling predictions across buildings, macro-F1 is computed over the fixed ordered union of
# global floor labels present (defaulting to canonical 0..4 when all floors exist in the evaluation set)
# with zero_division=0.0, alongside diagnostic building-weighted macro-F1.
POOLED_MACRO_F1_LABEL_POLICY: str = "fixed_global_floor_union_with_zero_division_0"
POOLED_GLOBAL_FLOOR_LABELS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Hardware policy: CPU-compatible scikit-learn execution without nested parallelism
DEFAULT_N_JOBS: int = 1
