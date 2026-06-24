# dskit/utils.py
"""
Shared utilities for the dskit data science toolkit.

Functions in this module are used across multiple other modules.
They have no dependencies on other dskit modules, making this
module the safe foundation that everything else can import from.
"""

import json
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Union
import pandas as pd
import numpy as np


__all__ = [
    "ensure_dir",
    "timestamp",
    "hash_dataframe",
    "safe_json_serialize",
    "is_numeric_series",
    "flatten_dict",
    "get_logger",
    "check_columns",
]

# ── Directory utilities ──────────────────────────────────────────────


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory and all parents if they do not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The resolved Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Timestamp utilities ──────────────────────────────────────────────


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """
    Return the current time as a formatted string.

    Parameters
    ----------
    fmt : str, optional
        strftime format string. Default is '%Y%m%d_%H%M%S',
        producing strings like '20250515_103241'.

    Returns
    -------
    str
        Formatted timestamp.

    Examples
    --------
    >>> ts = timestamp()
    >>> len(ts) == 15  # '20250515_103241'
    True
    """
    return datetime.now().strftime(fmt)


def iso_timestamp() -> str:
    """Return the current time as an ISO 8601 string."""
    return datetime.now().isoformat()


# ── Hashing utilities ────────────────────────────────────────────────


def hash_dataframe(df: pd.DataFrame, n_chars: int = 16) -> str:
    """
    Compute a short SHA-256 hash of a DataFrame's content.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash.
    n_chars : int, optional
        Length of the returned hash string. Default is 16.

    Returns
    -------
    str
        Hex digest of the DataFrame's content hash, truncated to n_chars.

    Notes
    -----
    Two DataFrames with identical values and index produce the same hash.
    A change of any single value produces a different hash.
    """
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:n_chars]


def hash_dict(d: dict, n_chars: int = 16) -> str:
    """Compute a short SHA-256 hash of a dictionary's content."""
    content = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:n_chars]


# ── JSON utilities ───────────────────────────────────────────────────


def safe_json_serialize(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects to serializable equivalents.

    Handles: numpy scalars, numpy arrays, pandas Series, Path objects,
    and datetime objects.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable equivalent.

    Raises
    ------
    TypeError
        If the object cannot be converted.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def to_json(obj: Any, path: str = None, indent: int = 2) -> str:
    """
    Serialize an object to a JSON string, with safe handling of numpy types.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    path : str, optional
        If provided, write the JSON string to this file path.
    indent : int, optional
        JSON indentation. Default is 2.

    Returns
    -------
    str
        JSON string representation.
    """
    json_str = json.dumps(obj, indent=indent, default=safe_json_serialize)
    if path:
        ensure_dir(Path(path).parent)
        with open(path, "w") as f:
            f.write(json_str)
    return json_str


def from_json(path: str) -> Any:
    """Load a JSON file and return its content."""
    with open(path) as f:
        return json.load(f)


# ── DataFrame utilities ──────────────────────────────────────────────


def is_numeric_series(series: pd.Series) -> bool:
    """Return True if a Series contains numeric (not boolean) data."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def check_columns(df: pd.DataFrame, required: list, caller: str = "") -> None:
    """
    Raise a ValueError if any required columns are missing from df.

    Parameters
    ----------
    df : pd.DataFrame
    required : list of str
        Column names that must be present.
    caller : str, optional
        Name of the calling function, included in the error message.

    Raises
    ------
    ValueError
        With a message listing all missing columns.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        prefix = f"[{caller}] " if caller else ""
        raise ValueError(
            f"{prefix}Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


# ── Dictionary utilities ─────────────────────────────────────────────


def flatten_dict(d: dict, sep: str = ".", prefix: str = "") -> dict:
    """
    Flatten a nested dictionary into a single-level dict with dotted keys.

    Parameters
    ----------
    d : dict
        Nested dictionary to flatten.
    sep : str, optional
        Separator between key levels. Default is '.'.
    prefix : str, optional
        Prefix for top-level keys.

    Returns
    -------
    dict
        Flat dictionary.

    Examples
    --------
    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items = {}
    for key, value in d.items():
        new_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, sep=sep, prefix=new_key))
        else:
            items[new_key] = value
    return items


# ── Logging utility ──────────────────────────────────────────────────


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a named logger with a consistent format.

    A thin wrapper around the logging module that applies the
    same formatter used throughout dskit. Safe to call multiple
    times — returns the existing logger if already configured.

    Parameters
    ----------
    name : str
        Logger name (typically the module name, e.g. 'dskit.data_io').
    level : str, optional
        Minimum log level. Default is 'INFO'.

    Returns
    -------
    logging.Logger
    """
    import sys
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger