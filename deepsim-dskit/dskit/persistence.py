"""
modules/persistence.py

Persistence utilities and `ArtifactStore` for saving/loading models,
pipelines and metadata (joblib / pickle / json helpers).
"""
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def save_pickle(obj: object, path: str) -> None:
    """
    Save a Python object to disk using pickle.

    Parameters
    ----------
    obj : object
        Any picklable Python object.
    path : str
        Destination file path. Convention: use .pkl extension.

    Notes
    -----
    pickle files are Python-version-sensitive. A file saved with
    Python 3.10 may not load correctly in Python 3.7.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved (pickle): '{path}'  ({Path(path).stat().st_size:,} bytes)")


def load_pickle(path: str) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        Path to a .pkl file.

    Returns
    -------
    object
        The deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at '{path}'.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded (pickle): '{path}'")
    return obj

def save_joblib(obj: object, path: str, compress: int = 3) -> None:
    """
    Save a Python object to disk using joblib.

    Preferred over pickle for sklearn models and numpy-heavy objects.

    Parameters
    ----------
    obj : object
        Any joblib-serializable Python object.
    path : str
        Destination file path. Convention: use .joblib extension.
    compress : int, optional
        Compression level (0=none, 9=maximum). Default is 3.
        Higher compression reduces file size but increases save/load time.

    Notes
    -----
    joblib is significantly faster than pickle for objects
    containing large numpy arrays (e.g. sklearn models).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=compress)
    print(f"Saved (joblib): '{path}'  ({Path(path).stat().st_size:,} bytes)")

def load_joblib(path: str) -> object:
    """
    Load a Python object from a joblib file.

    Parameters
    ----------
    path : str
        Path to a .joblib file.

    Returns
    -------
    object
        The deserialized object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at '{path}'.")
    obj = joblib.load(path)
    print(f"Loaded (joblib): '{path}'")
    return obj


def save_metadata(metadata: dict, path: str) -> None:
    """
    Save artifact metadata to a JSON file.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary. All values must be JSON-serializable
        (str, int, float, list, dict, bool, None).
    path : str
        Destination file path. Convention: use .json extension.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved (JSON):   '{path}'")


def load_metadata(path: str) -> dict:
    """
    Load metadata from a JSON file.

    Parameters
    ----------
    path : str
        Path to a .json metadata file.

    Returns
    -------
    dict
        The loaded metadata dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No metadata file at '{path}'.")
    with open(path) as f:
        return json.load(f)


def list_artifact_versions(
    base_directory: str,
    model_name: str
) -> pd.DataFrame:
    """
    List all saved versions of a named artifact.

    Parameters
    ----------
    base_directory : str
        Root directory containing artifact subdirectories.
    model_name : str
        Prefix used to identify this artifact's directories.

    Returns
    -------
    pd.DataFrame
        Columns: version_dir, model_name, saved_at, test_r2.
        Sorted by saved_at descending (newest first).
    """
    base = Path(base_directory)
    rows = []

    for version_dir in sorted(base.iterdir()):
        if not version_dir.is_dir():
            continue
        meta_path = version_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = load_metadata(meta_path)
        if meta.get("model_name", "").startswith(model_name):
            rows.append({
                "version_dir": str(version_dir),
                "model_name":  meta.get("model_name"),
                "saved_at":    meta.get("saved_at"),
                "test_r2":     meta.get("test_r2"),
            })

    return (
        pd.DataFrame(rows)
        .sort_values("saved_at", ascending=False)
        .reset_index(drop=True)
    )


def load_latest_artifact(
    base_directory: str,
    model_name: str
) -> "ArtifactStore":
    """
    Load the most recently saved version of a named artifact.

    Parameters
    ----------
    base_directory : str
        Root directory containing artifact subdirectories.
    model_name : str
        Prefix used to identify this artifact's directories.

    Returns
    -------
    ArtifactStore
        The most recently saved artifact matching model_name.
    """
    versions = list_artifact_versions(base_directory, model_name)
    if versions.empty:
        raise FileNotFoundError(
            f"No artifacts matching '{model_name}' in '{base_directory}'."
        )
    latest_dir = versions.iloc[0]["version_dir"]
    print(f"Loading latest version: '{versions.iloc[0]['model_name']}'")
    return ArtifactStore.load(latest_dir)


class ArtifactStore:
    """
    A self-describing bundle of all artifacts needed for inference.

    Bundles a trained model, its preprocessing pipeline (scaler,
    encoder, etc.), the feature list it expects, and metadata
    describing what it is and how it was trained.

    Parameters
    ----------
    model_name : str
        Human-readable identifier for this artifact.

    Attributes
    ----------
    model_name : str
    model_ : fitted estimator
    pipeline_ : object (optional)
        Fitted preprocessing pipeline or scaler.
    features_ : list of str
        Feature names the model was trained on.
    metadata_ : dict
        Arbitrary metadata dictionary.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_     = None
        self.pipeline_  = None
        self.features_  = []
        self.metadata_  = {}

    def set_model(self, model) -> "ArtifactStore":
        """Attach a trained model to this store."""
        self.model_ = model
        return self

    def set_pipeline(self, pipeline) -> "ArtifactStore":
        """Attach a fitted preprocessing pipeline or scaler."""
        self.pipeline_ = pipeline
        return self

    def set_features(self, features: list) -> "ArtifactStore":
        """Record the feature names this model expects at inference time."""
        self.features_ = features
        return self

    def set_metadata(self, metadata: dict) -> "ArtifactStore":
        """Attach metadata describing the artifact."""
        self.metadata_ = metadata
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess X and generate predictions.

        Applies the stored pipeline (if any) before predicting.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature DataFrame. Must contain all columns in features_.

        Returns
        -------
        np.ndarray
            Model predictions.

        Raises
        ------
        RuntimeError
            If no model has been attached.
        ValueError
            If X is missing any expected feature columns.
        """
        if self.model_ is None:
            raise RuntimeError("No model attached. Call set_model() first.")

        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")

        X_input = X[self.features_]

        if self.pipeline_ is not None:
            X_input = self.pipeline_.transform(X_input)

        return self.model_.predict(X_input)

    def save(self, directory: str) -> None:
        """
        Save all artifacts to a directory.

        Creates four files:
        - model.joblib: the fitted model
        - pipeline.joblib: the fitted pipeline (if present)
        - features.json: the feature list
        - metadata.json: the metadata dictionary

        Parameters
        ----------
        directory : str
            Destination directory. Created if it does not exist.
        """
        if self.model_ is None:
            raise RuntimeError("No model to save. Call set_model() first.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        save_joblib(self.model_, path / "model.joblib")

        if self.pipeline_ is not None:
            save_joblib(self.pipeline_, path / "pipeline.joblib")

        save_metadata(
            {"features": self.features_},
            path / "features.json"
        )
        save_metadata(
            {**self.metadata_, "model_name": self.model_name,
             "saved_at": datetime.now().isoformat()},
            path / "metadata.json"
        )
        print(f"\nArtifact '{self.model_name}' saved to '{directory}'")

    @staticmethod
    def load(directory: str) -> "ArtifactStore":
        """
        Load an ArtifactStore from a directory.

        Parameters
        ----------
        directory : str
            Directory containing model.joblib, features.json,
            and metadata.json. pipeline.joblib is optional.

        Returns
        -------
        ArtifactStore
            The fully loaded artifact.

        Raises
        ------
        FileNotFoundError
            If model.joblib or metadata.json is missing.
        """
        path = Path(directory)

        for required in ["model.joblib", "metadata.json", "features.json"]:
            if not (path / required).exists():
                raise FileNotFoundError(
                    f"Required file '{required}' not found in '{directory}'."
                )

        meta     = load_metadata(path / "metadata.json")
        features = load_metadata(path / "features.json")["features"]
        model    = load_joblib(path / "model.joblib")

        pipeline = None
        if (path / "pipeline.joblib").exists():
            pipeline = load_joblib(path / "pipeline.joblib")

        store = ArtifactStore(meta.get("model_name", "unknown"))
        store.set_model(model)
        store.set_features(features)
        store.set_metadata(meta)
        if pipeline:
            store.set_pipeline(pipeline)

        print(f"\nLoaded artifact: '{store.model_name}' "
              f"(saved {meta.get('saved_at', 'unknown')})")
        return store
