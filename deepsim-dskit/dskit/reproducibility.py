# reproducibility.py

import json
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Registry of available model classes
MODEL_REGISTRY = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full pipeline reproducibility.

    Seeds Python's random module, NumPy's random generator,
    and the OS-level hash seed environment variable.

    Parameters
    ----------
    seed : int, optional
        The seed value to use. Default is 42.

    Notes
    -----
    For PyTorch or TensorFlow models, additional seeding is required:
        import torch; torch.manual_seed(seed)
        import tensorflow as tf; tf.random.set_seed(seed)
    This function covers standard scikit-learn and NumPy workflows.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed set: {seed}")

def instantiate_model(model_config: dict):
    """
    Instantiate a model from a configuration dictionary.

    Parameters
    ----------
    model_config : dict
        Must contain 'class' (str) and 'params' (dict).

    Returns
    -------
    Unfitted sklearn estimator.

    Raises
    ------
    ValueError
        If the model class is not in MODEL_REGISTRY.
    """
    class_name = model_config["class"]
    if class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class '{class_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[class_name](**model_config.get("params", {}))


class RunLogger:
    """
    A structured, time-stamped logger for experiment runs.

    Writes log entries to an in-memory list and optionally to a
    JSON Lines file (.jsonl) — one JSON object per line — which
    is both human-readable and machine-parseable.

    Parameters
    ----------
    experiment_id : str
        Identifier of the experiment being logged.
    log_file : str, optional
        Path to a .jsonl file for persistent logging.
        If None, logging is in-memory only.

    Attributes
    ----------
    entries_ : list of dict
        All log entries recorded during this run.
    """

    def __init__(self, experiment_id: str, log_file: str = None):
        self.experiment_id = experiment_id
        self.log_file      = log_file
        self.entries_      = []

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, data: dict = None, level: str = "INFO") -> None:
        """
        Record a log entry.

        Parameters
        ----------
        event : str
            Short description of the event (e.g. 'split_complete').
        data : dict, optional
            Additional structured data to record.
        level : str, optional
            Log level: 'INFO', 'WARNING', or 'ERROR'. Default is 'INFO'.
        """
        from datetime import datetime
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "level":         level,
            "event":         event,
            **(data or {}),
        }
        self.entries_.append(entry)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def summary(self) -> pd.DataFrame:
        """
        Return all log entries as a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per log entry, sorted by timestamp.
        """
        return pd.DataFrame(self.entries_)

    def warnings(self) -> list:
        """Return all WARNING and ERROR entries."""
        return [e for e in self.entries_
                if e["level"] in {"WARNING", "ERROR"}]


def run_experiment(config: dict, base_dir: str = "experiments") -> dict:
    """
    Execute a complete experiment from a configuration dictionary.

    Performs the following steps in order:
    1. Set global seed
    2. Load dataset
    3. Split into train and test sets
    4. Fit preprocessing pipeline on training data
    5. Transform both splits
    6. Train the specified model
    7. Compute evaluation metrics
    8. Save the complete experiment to disk
    9. Return the run record

    Parameters
    ----------
    config : dict
        A complete experiment configuration (see schema above).
    base_dir : str, optional
        Root directory for experiment artifacts. Default is 'experiments'.

    Returns
    -------
    dict
        Run record containing: experiment_id, metrics, artifact_dir,
        config, and status.
    """
    exp_id = config["experiment_id"]
    print(f"\n{'='*55}")
    print(f"Running experiment: {exp_id}")
    print(f"{'='*55}")

    # 1. Seed
    set_global_seed(config["seed"])

    # 2. Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["url"], index_col=0)
    target = data_cfg["target"]
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Split
    split_cfg  = config["splitting"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=split_cfg["test_size"],
        random_state=config["seed"],
    )
    print(f"Split: {len(X_train)} train, {len(X_test)} test rows")

    # 4. Preprocessing — fit on training data only
    pp_cfg  = config["preprocessing"]
    scaler  = StandardScaler()
    scaling_cols = pp_cfg["scaling"]["columns"]
    X_train_np   = scaler.fit_transform(X_train[scaling_cols])
    X_test_np    = scaler.transform(X_test[scaling_cols])
    X_train_scaled = pd.DataFrame(X_train_np, columns=scaling_cols,
                                   index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_np,  columns=scaling_cols,
                                   index=X_test.index)

    # 5. Train candidate models

    models_cfg = config["models"]

    best_model = None
    best_model_name = None
    best_score = float("inf")

    for model_name, model_cfg in models_cfg.items():

        model = instantiate_model(model_cfg)

        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, preds)

        print(f"Trained: {model_name} ({type(model).__name__}) | MSE={mse:.4f}")

        if mse < best_score:
            best_score = mse
            best_model = model
            best_model_name = model_name

    # 6. Metrics
    def _metrics(X, y, label):
        p = best_model.predict(X)
        return {
            f"{label}_r2":   round(float(r2_score(y, p)), 4),
            f"{label}_rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            f"{label}_mae":  round(float(mean_absolute_error(y, p)), 4),
        }

    metrics = {**_metrics(X_train_scaled, y_train, "train"),
               **_metrics(X_test_scaled,  y_test,  "test")}
    print(f"Test R²: {metrics['test_r2']}  |  RMSE: {metrics['test_rmse']}")

    # 7. Save
    output_cfg = config.get("output", {})
    exp_base_dir = output_cfg.get("experiments_dir", base_dir)

    exp_dir = Path(exp_base_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(best_model, exp_dir / "model.joblib", compress=3)
    joblib.dump(scaler, exp_dir / "pipeline.joblib", compress=3)

    with open(exp_dir / "features.json", "w") as f:
        json.dump({"features": scaling_cols}, f, indent=2)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    from datetime import datetime
    metadata = {
        "experiment_id": exp_id,
        "model_class": type(best_model).__name__,
        "model_params": {
            k: str(v)
            for k, v in best_model.get_params().items()
        },
        "trained_at":    datetime.now().isoformat(),
        "seed":          config["seed"],
        "notes":         config.get("notes", ""),
        "status":        "active",
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: '{exp_dir}'")

    return {
        "experiment_id":  exp_id,
        "best_model_name": best_model_name,
        "best_score":      best_score,
        "metrics":         metrics,
        "artifact_dir":    str(exp_dir),
        "config":          config,
        "status":          "success",
    }

def run_experiment_logged(
    config: dict,
    base_dir: str = "experiments",
    log_dir:  str = "logs"
) -> dict:
    """
    Execute an experiment with full structured logging.

    Parameters
    ----------
    config : dict
        Complete experiment configuration.
    base_dir : str, optional
        Root directory for experiment artifacts.
    log_dir : str, optional
        Directory for log files.

    Returns
    -------
    dict
        Run record with metrics, artifact directory, and log entries.
    """
    exp_id = config["experiment_id"]
    log_path = f"{log_dir}/{exp_id}.jsonl"
    logger = RunLogger(exp_id, log_file=log_path)

    logger.log("run_started", {"config_keys": list(config.keys())})

    # 1. Seed
    seed = config["seed"]
    set_global_seed(seed)
    logger.log("seed_set", {"seed": seed})

    # 2. Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["url"], index_col=0)
    logger.log("data_loaded", {
        "rows": len(df), "columns": list(df.columns), "url": data_cfg["url"]
    })

    # Check for missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing  = sum(missing_counts.values())
    if total_missing > 0:
        logger.log("missing_values_detected",
                   {"counts": missing_counts}, level="WARNING")
    else:
        logger.log("no_missing_values")

    # 3. Split
    target = data_cfg["target"]
    X = df.drop(columns=[target])
    y = df[target]
    split_cfg = config["splitting"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_cfg["test_size"], random_state=seed
    )
    logger.log("split_complete", {
        "train_rows": len(X_train), "test_rows": len(X_test),
        "test_size":  split_cfg["test_size"]
    })

    # 4. Scale
    pp_cfg       = config["preprocessing"]
    scaling_cols = pp_cfg["scaling"]["columns"]
    scaler       = StandardScaler()
    X_train_np   = scaler.fit_transform(X_train[scaling_cols])
    X_test_np    = scaler.transform(X_test[scaling_cols])
    X_train_sc   = pd.DataFrame(X_train_np, columns=scaling_cols, index=X_train.index)
    X_test_sc    = pd.DataFrame(X_test_np,  columns=scaling_cols, index=X_test.index)
    logger.log("scaling_complete", {
        "method":  pp_cfg["scaling"]["method"],
        "columns": scaling_cols,
    })

    # 5. Train
    model = instantiate_model(config["model"])
    model.fit(X_train_sc, y_train)
    logger.log("model_trained", {
        "model_class":  type(model).__name__,
        "model_params": {k: str(v) for k, v in model.get_params().items()},
    })

    # 6. Evaluate
    def _m(X, y, label):
        p = best_model.predict(X)
        return {
            f"{label}_r2":   round(float(r2_score(y, p)), 4),
            f"{label}_rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4),
            f"{label}_mae":  round(float(mean_absolute_error(y, p)), 4),
        }

    metrics = {**_m(X_train_sc, y_train, "train"),
               **_m(X_test_sc,  y_test,  "test")}
    logger.log("evaluation_complete", metrics)

    # 7. Save artifacts
    import joblib
    from datetime import datetime

    output_cfg = config.get("output", {})
    exp_base_dir = output_cfg.get("experiments_dir", base_dir)
    
    exp_dir = Path(exp_base_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model,  exp_dir / "model.joblib",    compress=3)
    joblib.dump(scaler, exp_dir / "pipeline.joblib", compress=3)

    for fname, data in [
        ("features.json", {"features": scaling_cols}),
        ("config.json",   config),
        ("metrics.json",  metrics),
        ("metadata.json", {
            "experiment_id": exp_id, "model_class": type(model).__name__,
            "trained_at": datetime.now().isoformat(),
            "seed": seed, "notes": config.get("notes", ""), "status": "active",
        }),
    ]:
        with open(exp_dir / fname, "w") as f:
            json.dump(data, f, indent=2, default=str)

    logger.log("artifacts_saved", {"artifact_dir": str(exp_dir)})
    logger.log("run_complete", {"status": "success"})

    return {
        "experiment_id": exp_id,
        "metrics":       metrics,
        "artifact_dir":  str(exp_dir),
        "config":        config,
        "status":        "success",
        "log":           logger,
    }
