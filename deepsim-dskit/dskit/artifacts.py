# modules/artifacts.py

import json
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(
    model,
    X_train, y_train,
    X_test,  y_test
) -> dict:
    """
    Compute a standard set of regression metrics for train and test sets.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train, y_train : training data
    X_test, y_test : test data

    Returns
    -------
    dict
        Keys: train_r2, train_rmse, train_mae,
              test_r2, test_rmse, test_mae.
    """
    def _metrics(X, y, label):
        preds = model.predict(X)
        return {
            f"{label}_r2":   round(r2_score(y, preds), 4),
            f"{label}_rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
            f"{label}_mae":  round(float(mean_absolute_error(y, preds)), 4),
        }

    return {**_metrics(X_train, y_train, "train"),
            **_metrics(X_test,  y_test,  "test")}


def save_experiment(
    experiment_id: str,
    model,
    pipeline,
    features: list,
    config: dict,
    metrics: dict,
    base_dir: str = "experiments",
    notes: str = ""
) -> str:
    """
    Save a complete experiment to a self-describing directory.

    Creates six files: model.joblib, pipeline.joblib, features.json,
    config.json, metrics.json, and metadata.json.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for this experiment (e.g. 'exp_001').
    model : fitted sklearn estimator
    pipeline : fitted scaler or PreprocessingPipeline
    features : list of str
        Feature names the model was trained on.
    config : dict
        Preprocessing and training configuration.
    metrics : dict
        Evaluation metrics (from compute_metrics()).
    base_dir : str, optional
        Root directory for experiments. Default is 'experiments'.
    notes : str, optional
        Human-readable notes about this experiment.

    Returns
    -------
    str
        Path to the saved experiment directory.
    """
    exp_dir = Path(base_dir) / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(model, exp_dir / "model.joblib", compress=3)

    # Pipeline
    joblib.dump(pipeline, exp_dir / "pipeline.joblib", compress=3)

    # Features
    with open(exp_dir / "features.json", "w") as f:
        json.dump({"features": features}, f, indent=2)

    # Config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Metrics
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Metadata
    metadata = {
        "experiment_id":  experiment_id,
        "model_class":    type(model).__name__,
        "model_params":   {k: str(v) for k, v in model.get_params().items()},
        "trained_at":     datetime.now().isoformat(),
        "training_rows":  len(features),
        "notes":          notes,
        "status":         "active",
    }
    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved experiment '{experiment_id}' → '{exp_dir}'")
    return str(exp_dir)


def load_experiment(experiment_dir: str) -> dict:
    """
    Load a complete experiment from its directory.

    Parameters
    ----------
    experiment_dir : str
        Path to a saved experiment directory.

    Returns
    -------
    dict
        Keys: model, pipeline, features, config, metrics, metadata.

    Raises
    ------
    FileNotFoundError
        If required files are missing from the directory.
    """
    path = Path(experiment_dir)

    for required in ["model.joblib", "metadata.json",
                     "features.json", "metrics.json"]:
        if not (path / required).exists():
            raise FileNotFoundError(
                f"Required file '{required}' missing from '{path}'."
            )

    model    = joblib.load(path / "model.joblib")
    pipeline = (
        joblib.load(path / "pipeline.joblib")
        if (path / "pipeline.joblib").exists() else None
    )

    with open(path / "features.json")  as f: features  = json.load(f)["features"]
    with open(path / "metrics.json")   as f: metrics   = json.load(f)
    with open(path / "metadata.json")  as f: metadata  = json.load(f)
    with open(path / "config.json")    as f: config    = json.load(f)

    print(f"Loaded experiment '{metadata['experiment_id']}' "
          f"({metadata['model_class']})")
    return {
        "model":    model,
        "pipeline": pipeline,
        "features": features,
        "metrics":  metrics,
        "metadata": metadata,
        "config":   config,
    }


def promote_experiment(
    experiment_id: str,
    registry: "ExperimentRegistry",
    production_dir: str = "production",
    base_dir: str = "experiments"
) -> None:
    """
    Promote an experiment to production.

    Copies all artifact files from the experiment directory to
    the production directory, overwriting any existing production
    artifact. Updates the experiment's status in the registry to
    'promoted' and any previously promoted experiment to 'active'.

    Parameters
    ----------
    experiment_id : str
        Identifier of the experiment to promote.
    registry : ExperimentRegistry
        The active registry instance.
    production_dir : str, optional
        Destination directory. Default is 'production'.
    base_dir : str, optional
        Root experiments directory. Default is 'experiments'.
    """
    # Demote any currently promoted experiment
    for exp in registry.experiments_:
        if exp.get("status") == "promoted":
            registry.update_status(exp["experiment_id"], "active")

    # Copy experiment to production
    src  = Path(base_dir) / experiment_id
    dest = Path(production_dir)

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    # Update status
    registry.update_status(experiment_id, "promoted")

    print(f"\nPromoted '{experiment_id}' to production.")
    print(f"Production artifact: '{dest}'")


def archive_experiment(
    experiment_id: str,
    registry: "ExperimentRegistry",
    base_dir: str = "experiments",
    archive_dir: str = "archive"
) -> None:
    """
    Move an experiment to the archive directory.

    The experiment directory is moved from base_dir to archive_dir.
    Its registry status is updated to 'archived'.

    Parameters
    ----------
    experiment_id : str
        Identifier of the experiment to archive.
    registry : ExperimentRegistry
    base_dir : str, optional
        Source root directory. Default is 'experiments'.
    archive_dir : str, optional
        Destination archive directory. Default is 'archive'.

    Raises
    ------
    ValueError
        If the experiment has status 'promoted' (production models
        should not be archived without explicit demotion).
    """
    for exp in registry.experiments_:
        if exp["experiment_id"] == experiment_id:
            if exp.get("status") == "promoted":
                raise ValueError(
                    f"Cannot archive '{experiment_id}': currently in production. "
                    "Demote it first by promoting a different experiment."
                )
            break

    src  = Path(base_dir) / experiment_id
    dest = Path(archive_dir) / experiment_id

    Path(archive_dir).mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Experiment directory '{src}' not found.")

    shutil.move(str(src), str(dest))
    registry.update_status(experiment_id, "archived")

    print(f"Archived '{experiment_id}' → '{dest}'")


class ExperimentRegistry:
    """
    A searchable registry of all experiments in a project.

    The registry maintains a central JSON index file that records
    metadata and metrics for every experiment. The full artifact
    files remain in their experiment directories.

    Parameters
    ----------
    registry_path : str
        Path to the registry JSON file. Created if it does not exist.
        Default is 'registry/experiments.json'.

    Attributes
    ----------
    registry_path : Path
    experiments_ : list of dict
        The in-memory list of experiment records.
    """

    def __init__(self, registry_path: str = "registry/experiments.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments_ = self._load_registry()

    def _load_registry(self) -> list:
        """Load the registry from disk, or return an empty list."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return []

    def _save_registry(self) -> None:
        """Persist the registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.experiments_, f, indent=2, default=str)

    def register(self, experiment_dir: str) -> "ExperimentRegistry":
        """
        Add an experiment to the registry by reading its saved files.

        Parameters
        ----------
        experiment_dir : str
            Path to a complete experiment directory (produced by
            save_experiment()).

        Returns
        -------
        ExperimentRegistry
            Returns self for method chaining.

        Raises
        ------
        FileNotFoundError
            If required files are missing from experiment_dir.
        """
        path = Path(experiment_dir)

        for required in ["metadata.json", "metrics.json", "config.json"]:
            if not (path / required).exists():
                raise FileNotFoundError(
                    f"Required file '{required}' missing from '{path}'."
                )

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        with open(path / "metrics.json") as f:
            metrics = json.load(f)
        with open(path / "config.json") as f:
            config = json.load(f)

        record = {
            "experiment_id": metadata["experiment_id"],
            "experiment_dir": str(path),
            "model_class":   metadata["model_class"],
            "trained_at":    metadata["trained_at"],
            "status":        metadata.get("status", "active"),
            "notes":         metadata.get("notes", ""),
            **metrics,
        }

        # Update existing or append
        existing = [
            i for i, e in enumerate(self.experiments_)
            if e["experiment_id"] == record["experiment_id"]
        ]
        if existing:
            self.experiments_[existing[0]] = record
        else:
            self.experiments_.append(record)

        self._save_registry()
        print(f"Registered: '{record['experiment_id']}' "
              f"({record['model_class']}, test_r2={record.get('test_r2')})")
        return self

    def all_experiments(self) -> pd.DataFrame:
        """
        Return all registered experiments as a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per experiment, sorted by test_r2 descending.
        """
        if not self.experiments_:
            return pd.DataFrame()
        return (
            pd.DataFrame(self.experiments_)
            .sort_values("test_r2", ascending=False)
            .reset_index(drop=True)
        )

    def best(
        self,
        metric: str = "test_r2",
        ascending: bool = False,
        status: str = "active"
    ) -> dict:
        """
        Return the experiment record with the best metric value.

        Parameters
        ----------
        metric : str, optional
            Metric column to rank by. Default is 'test_r2'.
        ascending : bool, optional
            If True, the lowest value is considered best (e.g. for RMSE).
            Default is False (highest is best).
        status : str, optional
            Filter to experiments with this status. Default is 'active'.

        Returns
        -------
        dict
            The best experiment record.

        Raises
        ------
        ValueError
            If no experiments match the status filter.
        """
        active = [e for e in self.experiments_ if e.get("status") == status]
        if not active:
            raise ValueError(f"No experiments with status='{status}'.")
        return sorted(active, key=lambda e: e.get(metric, 0),
                      reverse=not ascending)[0]

    def search(
        self,
        model_class: str = None,
        min_test_r2: float = None,
        max_test_rmse: float = None,
        status: str = "active"
    ) -> pd.DataFrame:
        """
        Search the registry with optional filters.

        Parameters
        ----------
        model_class : str, optional
            Filter to experiments using this model class name.
        min_test_r2 : float, optional
            Minimum test R² to include.
        max_test_rmse : float, optional
            Maximum test RMSE to include.
        status : str, optional
            Filter by experiment status. Default is 'active'.

        Returns
        -------
        pd.DataFrame
            Matching experiments sorted by test_r2 descending.
        """
        results = self.experiments_

        if status:
            results = [e for e in results if e.get("status") == status]
        if model_class:
            results = [e for e in results if e.get("model_class") == model_class]
        if min_test_r2 is not None:
            results = [e for e in results if e.get("test_r2", 0) >= min_test_r2]
        if max_test_rmse is not None:
            results = [e for e in results if e.get("test_rmse", 999) <= max_test_rmse]

        if not results:
            return pd.DataFrame()

        return (
            pd.DataFrame(results)
            .sort_values("test_r2", ascending=False)
            .reset_index(drop=True)
        )

    def update_status(
        self, experiment_id: str, status: str
    ) -> "ExperimentRegistry":
        """
        Update the status of an experiment.

        Parameters
        ----------
        experiment_id : str
            Identifier of the experiment to update.
        status : str
            New status (e.g. 'active', 'promoted', 'archived').

        Returns
        -------
        ExperimentRegistry
            Returns self for method chaining.
        """
        for exp in self.experiments_:
            if exp["experiment_id"] == experiment_id:
                exp["status"] = status
                self._save_registry()
                print(f"Updated '{experiment_id}' → status='{status}'")
                return self
        raise ValueError(f"Experiment '{experiment_id}' not found in registry.")