# dskit/__init__.py
"""
dskit — A Reusable Data Science Toolkit

A collection of production-ready modules for every stage of the
data science pipeline: loading, exploring, preprocessing, splitting,
modelling, artifact management, configuration, and performance.

Quick start:
    from dskit import load_dataset, create_split, run_full_pipeline
    from dskit import PreprocessingPipeline, ModelRegistry
    from dskit import ExperimentRegistry, ArtifactStore

Full module access:
    import dskit.data_io as data_io
    import dskit.modeling as modeling
"""

__version__ = "1.0.0"
__author__  = "Shouke Wei"
__license__ = "MIT"

# ── Data access ─────────────────────────────────────────────────────
from dskit.data_io import (
    load_dataset,
    save_dataset,
    validate_columns,
    validate_dtypes,
)

# ── Exploratory analysis ─────────────────────────────────────────────
from dskit.eda import (
    describe_df,
    missing_summary,
    skewness_summary,
    correlation_summary,
)

# ── Preprocessing ────────────────────────────────────────────────────
from dskit.preprocessing import (
    fill_missing,
    compute_fill_values,
    add_missing_indicator,
    detect_outliers_iqr,
    detect_outliers_zscore,
    outlier_summary,
    cap_outliers,
    cap_with_bounds,
    compute_iqr_bounds,
    log_transform,
    compute_scaling_params,
    apply_scaling,
)

# ── Splitting ────────────────────────────────────────────────────────
from dskit.splitting import (
    create_split,
    create_three_way_split,
    create_stratified_split,
    create_time_split,
    save_split,
    load_split,
    SplitResult,
)

# ── Pipeline ─────────────────────────────────────────────────────────
from dskit.pipeline import (
    PreprocessingPipeline,
    run_full_pipeline,
    run_feature_engineering,
    summarise_pipeline,

)

# ── Feature engineering ──────────────────────────────────────────────
from dskit.feature_engineering import (
    add_family_features,
    add_polynomial_features,
    fit_onehot_encoder,
    apply_onehot_encoder,
    fit_ordinal_encoder,
    apply_ordinal_encoder,
    fit_target_encoder,
    apply_target_encoder,
    log_transform_feature,
    bin_feature,
)

# ── Modeling ─────────────────────────────────────────────────────────
from dskit.modeling import (
    train_model,
    evaluate_regression,
    evaluate_classification,
    compute_vif,
    select_significant_features,
    save_model,
    load_model,
    ModelRegistry,
)

# ── Persistence ──────────────────────────────────────────────────────
from dskit.persistence import (
    save_joblib,
    load_joblib,
    save_pickle,
    load_pickle,
    save_metadata,
    load_metadata,
    ArtifactStore,
)

# ── Artifact management ──────────────────────────────────────────────
from dskit.artifacts import (
    save_experiment,
    load_experiment,
    promote_experiment,
    archive_experiment,
    compute_metrics,
    ExperimentRegistry,
)

# ── Reproducibility ──────────────────────────────────────────────────
from dskit.reproducibility import (
    set_global_seed,
    run_experiment,
    RunLogger,
)

# ── Configuration ────────────────────────────────────────────────────
from dskit.config import (
    validate_config,
    merge_configs,
    get_default_config,
    create_experiment_config,
    apply_environment_profile,
    save_config,
    load_config,
)

# ── Performance ──────────────────────────────────────────────────────
from dskit.performance import (
    timer,
    memory_profile,
    optimize_dtypes,
    memory_report,
    process_in_chunks,
    parallel_standardize,
    PipelineCache,
)

__all__ = [
    # Data access
    "load_dataset", "save_dataset", "validate_columns", "validate_dtypes",
    # EDA
    "describe_df", "missing_summary", "skewness_summary", "correlation_summary",
    # Preprocessing
    "fill_missing", "compute_fill_values", "add_missing_indicator",
    "detect_outliers_iqr", "detect_outliers_zscore", "outlier_summary",
    "cap_outliers", "cap_with_bounds", "compute_iqr_bounds",
    "log_transform", "compute_scaling_params", "apply_scaling",
    # Splitting
    "create_split", "create_three_way_split", "create_stratified_split",
    "create_time_split", "save_split", "load_split", "SplitResult",
    # Pipeline
    "PreprocessingPipeline", "run_full_pipeline", "run_feature_engineering", 
    "summarise_pipeline",
    # Feature engineering
    "add_family_features", "add_polynomial_features",
    "fit_onehot_encoder", "apply_onehot_encoder",
    "fit_ordinal_encoder", "apply_ordinal_encoder",
    "fit_target_encoder", "apply_target_encoder",
    "log_transform_feature", "bin_feature",
    # Modeling
    "train_model", "evaluate_regression", "evaluate_classification",
    "compute_vif", "select_significant_features",
    "save_model", "load_model", "ModelRegistry",
    # Persistence
    "save_joblib", "load_joblib", "save_pickle", "load_pickle",
    "save_metadata", "load_metadata", "ArtifactStore",
    # Artifacts
    "save_experiment", "load_experiment", "promote_experiment",
    "archive_experiment", "compute_metrics", "ExperimentRegistry",
    # Reproducibility
    "set_global_seed", "run_experiment", "RunLogger",
    # Config
    "validate_config", "merge_configs", "get_default_config",
    "save_config", "load_config", "create_experiment_config",
    "apply_environment_profile",
    # Performance
    "timer", "memory_profile", "optimize_dtypes", "memory_report",
    "process_in_chunks", "parallel_standardize", "PipelineCache",
    # Pipeline
    "run_full_pipeline",
    # Meta
    "__version__", "__author__",
]