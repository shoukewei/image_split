# modules/config.py
"""
Centralised configuration management for the data science system.

Provides:
  - Default configuration values
  - Schema validation
  - Typed configuration dataclasses
  - Config loading, saving, and merging utilities
  - Environment profile management
"""

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Global Allowed Options ───────────────────────────────────────────
VALID_SCALING_METHODS = {"standard", "minmax", "robust", "maxabs"}
VALID_OUTLIER_METHODS = {"iqr", "zscore"}
VALID_MODEL_CLASSES   = {
    "LinearRegression", "Ridge", "Lasso",
    "RandomForestRegressor", "RandomForestClassifier",
    "GradientBoostingRegressor", "GradientBoostingClassifier",
    "LogisticRegression",
}

REQUIRED_KEYS = {
    "experiment_id": str,
    "seed":          int,
    "data":          dict,
    "splitting":     dict,
    "preprocessing": dict,
    "output":        dict,
}

DATA_REQUIRED_KEYS     = {"url": str, "target": str}
SPLITTING_REQUIRED_KEYS = {"test_size": float, "random_state": int}
OUTPUT_REQUIRED_KEYS    = {"experiments_dir": str, "registry_path": str}

# ── Default configurations ───────────────────────────────────────────
DEFAULT_SPLITTING = {
    "test_size":    0.2,
    "val_size":     0.1,
    "stratified":   False,
    "random_state": 42,
}

DEFAULT_PREPROCESSING = {
    "missing": {
        "strategies":        {},
        "indicator_columns": [],
    },
    "outliers": {
        "columns":    [],
        "method":     "iqr",
        "multiplier": 1.5,
    },
    "scaling": {
        "columns": [],
        "method":  "standard",
    },
}

DEFAULT_OUTPUT = {
    "experiments_dir": "experiments",
    "registry_path":   "registry/experiments.json",
    "production_dir":  "production",
    "configs_dir":     "configs",
    "logs_dir":        "logs",
}

BASE_CONFIG = {
    "seed":          42,
    "splitting":     DEFAULT_SPLITTING,
    "preprocessing": DEFAULT_PREPROCESSING,
    "output":        DEFAULT_OUTPUT,
    "models":        {},
    "notes":         "",
}

ENVIRONMENT_PROFILES = {
    "development": {
        "output": {
            "experiments_dir": "experiments_dev",
            "registry_path":   "registry/experiments_dev.json",
            "production_dir":  "production_dev",
            "logs_dir":        "logs_dev",
        },
        "_dev_mode": True,
    },
    "testing": {
        "splitting": {"test_size": 0.3},
        "output": {
            "experiments_dir": "experiments_test",
            "registry_path":   "registry/experiments_test.json",
        },
        "_test_mode": True,
    },
    "production": {
        "output": {
            "experiments_dir": "experiments",
            "registry_path":   "registry/experiments.json",
            "production_dir":  "production",
        },
    },
}


# ── Typed Configuration Dataclasses ──────────────────────────────────

@dataclass
class DataConfig:
    """
    Configuration for the data access layer.

    Parameters
    ----------
    url : str
        The remote source path or database connection string.
    target : str
        The name of the target column for prediction.
    index_col : int, optional
        Column index to use as row label (default is 0).
    required_columns : list of str, optional
        A structural list of column names that must be present in dataset schema.
    """
    url:              str
    target:           str
    index_col:        Optional[int]  = 0
    required_columns: Optional[List[str]] = None

    def validate(self) -> List[str]:
        """
        Validate data settings constraints.

        Returns
        -------
        list of str
            A list containing any validation error strings. Empty if valid.
        """
        errors = []
        if not self.url:
            errors.append("'data.url' cannot be empty.")
        if not self.target:
            errors.append("'data.target' cannot be empty.")
        return errors


@dataclass
class SplittingConfig:
    """
    Configuration for the dataset splitting layer.

    Parameters
    ----------
    test_size : float, optional
        Proportion of dataset to partition into the test set (default 0.2).
    val_size : float, optional
        Proportion of dataset to partition into validation set (default 0.1).
    stratified : bool, optional
        Whether to perform stratified sampling on target distribution (default False).
    random_state : int, optional
        PRNG seed value to ensure deterministic splitting operations (default 42).
    """
    test_size:    float = 0.2
    val_size:     float = 0.1
    stratified:   bool  = False
    random_state: int   = 42

    def validate(self) -> List[str]:
        """
        Validate test size boundaries and split constraints.

        Returns
        -------
        list of str
            A list containing validation errors. Empty if valid.
        """
        errors = []
        if not (0 < self.test_size < 1):
            errors.append(f"test_size must be in (0, 1), got {self.test_size}")
        if self.val_size and not (0 <= self.val_size < 1):
            errors.append(f"val_size must be in [0, 1), got {self.val_size}")
        if self.test_size + self.val_size >= 1:
            errors.append(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) "
                f"must be < 1"
            )
        return errors


@dataclass
class ScalingConfig:
    """
    Configuration for column feature scaling.

    Parameters
    ----------
    columns : list of str
        Target structural features to undergo transformation.
    method : str, optional
        The mathematical scaler mechanism variant name (default "standard").
    """
    columns: List[str] = field(default_factory=list)
    method:  str       = "standard"

    def validate(self) -> List[str]:
        """
        Validate scaler methods and field parameters.

        Returns
        -------
        list of str
            A list containing validation error strings. Empty if valid.
        """
        errors = []
        if self.method not in VALID_SCALING_METHODS:
            errors.append(
                f"scaling.method must be one of {VALID_SCALING_METHODS}, "
                f"got '{self.method}'"
            )
        if not self.columns:
            errors.append("scaling.columns cannot be empty.")
        return errors


@dataclass
class PreprocessingConfig:
    """
    Configuration for the data transformation pipeline.

    Parameters
    ----------
    missing_strategies : dict, optional
        A dictionary mapping columns to explicit substitution strategies.
    indicator_columns : list of str, optional
        A list of column names for flag attributes tracking missingness.
    outlier_columns : list of str, optional
        Target structural variables evaluated during outlier detection.
    outlier_method : str, optional
        Statistical metric formula identifier for outlier clipping (default "iqr").
    outlier_multiplier : float, optional
        The deviation ceiling distance threshold factor (default 1.5).
    scaling : ScalingConfig, optional
        Sub-configuration containing features mapping rules.
    """
    missing_strategies:  Dict[str, Any] = field(default_factory=dict)
    indicator_columns:   List[str]      = field(default_factory=list)
    outlier_columns:     List[str]      = field(default_factory=list)
    outlier_method:      str            = "iqr"
    outlier_multiplier:  float          = 1.5
    scaling:             ScalingConfig  = field(default_factory=ScalingConfig)

    def validate(self) -> List[str]:
        """
        Validate pipeline parameters and nested scaling definitions.

        Returns
        -------
        list of str
            A list containing validation error strings. Empty if valid.
        """
        errors = []
        if self.outlier_method not in VALID_OUTLIER_METHODS:
            errors.append(
                f"outlier_method must be one of {VALID_OUTLIER_METHODS}, "
                f"got '{self.outlier_method}'"
            )
        if self.outlier_multiplier <= 0:
            errors.append(
                f"outlier_multiplier must be positive, got {self.outlier_multiplier}"
            )
        errors.extend(self.scaling.validate())
        return errors


@dataclass
class OutputConfig:
    """
    Configuration for output artifacts and artifact directories.

    Parameters
    ----------
    experiments_dir : str, optional
        Path location of stored experiment outputs.
    registry_path : str, optional
        Central registry file pathway for persistent parameters storage.
    production_dir : str, optional
        Directory destination for optimized inference serialization pipelines.
    configs_dir : str, optional
        Target tracking subdirectory for serialized JSON runtime configuration outputs.
    logs_dir : str, optional
        Output pipeline tracking stream subdirectory logs path.
    """
    experiments_dir: str = "experiments"
    registry_path:   str = "registry/experiments.json"
    production_dir:  str = "production"
    configs_dir:     str = "configs"
    logs_dir:        str = "logs"


@dataclass
class ExperimentConfig:
    """
    Complete, typed experiment master configuration tree.

    Parameters
    ----------
    experiment_id : str
        Unique alphanumeric identifying signature string for this run.
    data : DataConfig
        Nested target details specifying dataset structure.
    seed : int, optional
        An administrative seed used across model partitions (default 42).
    splitting : SplittingConfig, optional
        Structure specifying proportions for data separation blocks.
    preprocessing : PreprocessingConfig, optional
        Transformations settings applied to structural steps.
    models : dict, optional
        The evaluation targets model hyperparameter configurations matrix mapping.
    output : OutputConfig, optional
        Output workspace targeting rules paths mapping block.
    notes : str, optional
        Human readable summary context about this configuration run strategy.
    """
    experiment_id:  str
    data:           DataConfig
    seed:           int                 = 42
    splitting:      SplittingConfig     = field(default_factory=SplittingConfig)
    preprocessing:  PreprocessingConfig = field(default_factory=PreprocessingConfig)
    models:         Dict[str, Dict]     = field(default_factory=dict)
    output:         OutputConfig        = field(default_factory=OutputConfig)
    notes:          str                 = ""

    def validate(self) -> List[str]:
        """
        Perform a comprehensive sub-component field configuration constraint validation.

        Returns
        -------
        list of str
            A consolidated tracking list containing system error strings encountered.
        """
        errors = []
        if not self.experiment_id:
            errors.append("'experiment_id' cannot be empty.")
        if self.seed < 0:
            errors.append(f"'seed' must be non-negative, got {self.seed}")
        errors.extend(self.data.validate())
        errors.extend(self.splitting.validate())
        errors.extend(self.preprocessing.validate())
        for name, model_def in self.models.items():
            class_name = model_def.get("class", "")
            if class_name not in VALID_MODEL_CLASSES:
                errors.append(
                    f"models.{name}.class = '{class_name}' is not recognised."
                )
        return errors

    def to_dict(self) -> dict:
        """
        Deconstruct standard instance attributes fields tree back to structural plain text dicts.

        Returns
        -------
        dict
            Structural dictionary matching the base settings configuration mappings.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """
        Factory method providing explicit restoration pipeline to hydrate from standard dict configurations.

        Parameters
        ----------
        d : dict
            A structural plain dictionary mapping containing configuration details.

        Returns
        -------
        ExperimentConfig
            An instance populated with structural values from the source dictionary.
        """
        return cls(
            experiment_id=d["experiment_id"],
            seed=d.get("seed", 42),
            notes=d.get("notes", ""),
            data=DataConfig(**d["data"]),
            splitting=SplittingConfig(**d.get("splitting", {})),
            preprocessing=PreprocessingConfig(
                **{k: v for k, v in d.get("preprocessing", {}).items() if k != "scaling"},
                scaling=ScalingConfig(**d.get("preprocessing", {}).get("scaling", {})),
            ),
            models=d.get("models", {}),
            output=OutputConfig(**d.get("output", {})),
        )


# ── Utilities & Core Logic ───────────────────────────────────────────

def get_default_config() -> dict:
    """
    Return a deep copy of the base configuration defaults.

    Returns
    -------
    dict
        A standard base dictionary layout structure initialization mapping block.
    """
    return deepcopy(BASE_CONFIG)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base. Override values take precedence.
    
    Keys in override take precedence over keys in base.
    Nested dictionaries are merged recursively — not replaced wholesale.

    Parameters
    ----------
    base : dict
        The base configuration (e.g. DEFAULT_CONFIG or a profile config).
    override : dict
        The override configuration (e.g. experiment-specific values).

    Returns
    -------
    dict
        A new dictionary with override values merged into base.

    Examples
    --------
    >>> base    = {"seed": 42, "splitting": {"test_size": 0.2, "val_size": 0.1}}
    >>> override = {"seed": 99, "splitting": {"test_size": 0.3}}
    >>> merge_configs(base, override)
    {"seed": 99, "splitting": {"test_size": 0.3, "val_size": 0.1}}
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config: dict) -> List[str]:
    """
    Validate a config dict. Returns list of error strings (empty = valid).

    Checks for required keys, correct value types, valid values
    for enumerated fields, and semantic constraints (e.g. test_size
    must be between 0 and 1).

    Parameters
    ----------
    config : dict
        Experiment configuration to validate.

    Returns
    -------
    list of str
        A list of all validation errors found. Empty list = valid config.

    Examples
    --------
    >>> errors = validate_config(config)
    >>> if errors:
    ...     for e in errors: print(e)
    ... else:
    ...     run_full_pipeline(config)
    """
    errors = []

    # --- Top-level required keys ---
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in config:
            errors.append(f"Missing required key: '{key}'")
        elif not isinstance(config[key], expected_type):
            errors.append(
                f"'{key}' must be {expected_type.__name__}, "
                f"got {type(config[key]).__name__}"
            )

    if errors:
        return errors  # Cannot safely check sub-sections if top level is broken

    # --- Data section ---
    data_cfg = config.get("data", {})
    for key, expected_type in DATA_REQUIRED_KEYS.items():
        if key not in data_cfg:
            errors.append(f"Missing 'data.{key}'")
        elif not isinstance(data_cfg[key], expected_type):
            errors.append(f"'data.{key}' must be {expected_type.__name__}")

    # --- Splitting section ---
    split_cfg = config.get("splitting", {})
    for key, expected_type in SPLITTING_REQUIRED_KEYS.items():
         if key not in split_cfg:
            errors.append(f"Missing 'splitting.{key}'")

    test_size = split_cfg.get("test_size", 0)
    val_size  = split_cfg.get("val_size", 0)
    if not (0 < test_size < 1):
        errors.append(
            f"'splitting.test_size' must be between 0 and 1, got {test_size}"
        )
    if val_size and not (0 <= val_size < 1):
        errors.append(
            f"'splitting.val_size' must be between 0 and 1, got {val_size}"
        )
    if test_size + val_size >= 1:
        errors.append(
            f"'splitting.test_size' ({test_size}) + 'splitting.val_size' "
            f"({val_size}) must be < 1"
        )

    # --- Preprocessing section ---
    pp_cfg = config.get("preprocessing", {})

    scaling_cfg = pp_cfg.get("scaling", {})
    scaling_method = scaling_cfg.get("method", "standard")
    if scaling_method not in VALID_SCALING_METHODS:
        errors.append(
            f"'preprocessing.scaling.method' must be one of "
            f"{VALID_SCALING_METHODS}, got '{scaling_method}'"
        )
    if "columns" not in scaling_cfg:
        errors.append("Missing 'preprocessing.scaling.columns'")

    outlier_cfg = pp_cfg.get("outliers", {})
    if outlier_cfg:
        outlier_method = outlier_cfg.get("method", "iqr")
        if outlier_method not in VALID_OUTLIER_METHODS:
            errors.append(
                f"'preprocessing.outliers.method' must be one of "
                f"{VALID_OUTLIER_METHODS}, got '{outlier_method}'"
            )
        multiplier = outlier_cfg.get("multiplier", 1.5)
        if not isinstance(multiplier, (int, float)) or multiplier <= 0:
            errors.append(
                f"'preprocessing.outliers.multiplier' must be a positive number"
            )

    # --- Models section ---
    models_cfg = config.get("models", {})
    if models_cfg:
        for model_name, model_def in models_cfg.items():
            class_name = model_def.get("class", "")
            if class_name not in VALID_MODEL_CLASSES:
                errors.append(
                    f"'models.{model_name}.class' = '{class_name}' is not a "
                    f"recognised model class. Available: {VALID_MODEL_CLASSES}"
                )

    # --- Output section ---
    out_cfg = config.get("output", {})
    for key in OUTPUT_REQUIRED_KEYS:
        if key not in out_cfg:
            errors.append(f"Missing 'output.{key}'")

    # --- Seed ---
    seed = config.get("seed", 0)
    if not isinstance(seed, int) or seed < 0:
        errors.append(f"'seed' must be a non-negative integer, got {seed!r}")

    return errors


def apply_environment_profile(config: dict, environment: str = None) -> dict:
    """
    Apply an environment profile overlay. Reads DS_ENVIRONMENT if not given.
    
    The environment is determined (in order of precedence) by:
    1. The `environment` argument
    2. The `DS_ENVIRONMENT` environment variable
    3. Default: 'development'

    Parameters
    ----------
    config : dict
        Base experiment configuration.
    environment : str, optional
        Environment name: 'development', 'testing', or 'production'.
        If None, reads from the DS_ENVIRONMENT environment variable.

    Returns
    -------
    dict
        Configuration with environment profile applied.

    Raises
    ------
    ValueError
        If the environment name is not recognised.
    """
    if environment is None:
        environment = os.environ.get("DS_ENVIRONMENT", "development")

    if environment not in ENVIRONMENT_PROFILES:
        raise ValueError(
            f"Unknown environment '{environment}'. "
            f"Available: {list(ENVIRONMENT_PROFILES.keys())}"
        )

    profile = ENVIRONMENT_PROFILES[environment]
    result  = merge_configs(config, profile)
    result["_environment"] = environment

    print(f"Environment profile applied: '{environment}'")
    return result


def load_config(path: str) -> dict:
    """
    Load a configuration from a JSON or YAML file.

    The format is determined by the file extension:
    .json → JSON, .yaml or .yml → YAML.

    Parameters
    ----------
    path : str
        Path to a configuration file.

    Returns
    -------
    dict
        The loaded configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If the target configuration file is a YAML file and PyYAML is missing.
    ValueError
        If the file extension is not .json, .yaml, or .yml.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: '{path}'")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        with open(path) as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported format: '{suffix}'.")


def save_config(config: dict, path: str) -> None:
    """
    Save a configuration to a JSON or YAML file.

    The format is determined by the file extension.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    path : str
        Destination file path (.json, .yaml, or .yml).

    Raises
    ------
    ImportError
        If saving to a YAML variant path file target and PyYAML is missing.
    ValueError
        If file formatting layout target extension is not supported.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        raise ValueError(f"Unsupported format: '{suffix}'.")
    print(f"Config saved: '{path}'")


def create_experiment_config(
    experiment_id: str,
    model_configs: dict,
    overrides: dict = None,
    base: dict = None,
    notes: str = ""
) -> dict:
    """
    Create a validated experiment config by merging overrides into a base.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment.
    model_configs : dict
        Model definitions to include.
    overrides : dict, optional
        Any values to override from the base config.
    base : dict, optional
        Base configuration to inherit from. Defaults to ADVERTISING_BASE.
    notes : str, optional
        Human-readable description.

    Returns
    -------
    dict
        Complete, validated experiment configuration.

    Raises
    ------
    ValueError
        If validation finds errors in the resulting configuration.
    """
    base_cfg = deepcopy(base or get_default_config())
    override_cfg = merge_configs(overrides or {}, {
        "experiment_id": experiment_id,
        "models": model_configs,
        "notes":  notes,
    })
    config = merge_configs(base_cfg, override_cfg)
    errors = validate_config(config)
    if errors:
        raise ValueError(
            f"Invalid config for '{experiment_id}':\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    return config