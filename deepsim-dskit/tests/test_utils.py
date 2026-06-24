import pytest
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from dskit.utils import (
    ensure_dir,
    timestamp,
    iso_timestamp,
    hash_dataframe,
    hash_dict,
    safe_json_serialize,
    to_json,
    from_json,
    is_numeric_series,
    check_columns,
    flatten_dict,
    get_logger,
)

# ── Tests for Directory utilities ────────────────────────────────────

def test_ensure_dir(tmp_path):
    nested_path = tmp_path / "subdir1" / "subdir2"
    assert not nested_path.exists()
    
    returned_path = ensure_dir(nested_path)
    
    assert nested_path.exists()
    assert nested_path.is_dir()
    assert isinstance(returned_path, Path)
    assert returned_path == nested_path

    # Test passing a string path instead of a Path object
    str_path = os.path.join(str(tmp_path), "subdir3")
    returned_path_str = ensure_dir(str_path)
    assert os.path.exists(str_path)
    assert returned_path_str == Path(str_path)


# ── Tests for Timestamp utilities ────────────────────────────────────

def test_timestamp():
    ts = timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 15  # Matches default '%Y%m%d_%H%M%S' length
    
    custom_ts = timestamp("%Y-%m-%d")
    assert len(custom_ts) == 10
    assert custom_ts.count("-") == 2


def test_iso_timestamp():
    iso_ts = iso_timestamp()
    assert isinstance(iso_ts, str)
    # Check that it can be parsed back into a valid datetime object
    dt = datetime.fromisoformat(iso_ts)
    assert isinstance(dt, datetime)


# ── Tests for Hashing utilities ──────────────────────────────────────

def test_hash_dataframe():
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    df2 = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    df3 = pd.DataFrame({"A": [1, 2, 4], "B": ["x", "y", "z"]})
    
    hash1 = hash_dataframe(df1)
    hash2 = hash_dataframe(df2)
    hash3 = hash_dataframe(df3)
    
    assert hash1 == hash2  # Identical content yields identical hashes
    assert hash1 != hash3  # Content modifications yield different hashes
    assert len(hash1) == 16
    
    # Test custom truncation length
    short_hash = hash_dataframe(df1, n_chars=8)
    assert len(short_hash) == 8
    assert hash1.startswith(short_hash)


def test_hash_dict():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    d3 = {"a": 1, "b": 3}
    
    hash1 = hash_dict(d1)
    hash2 = hash_dict(d2)
    hash3 = hash_dict(d3)
    
    assert hash1 == hash2  # Keys sorted automatically before hashing
    assert hash1 != hash3
    assert len(hash1) == 16
    
    short_hash = hash_dict(d1, n_chars=5)
    assert len(short_hash) == 5
    assert hash1.startswith(short_hash)


# ── Tests for JSON utilities ─────────────────────────────────────────

def test_safe_json_serialize():
    # Standardize path comparison based on host operating system
    assert safe_json_serialize(Path("/tmp/test")) == str(Path("/tmp/test"))
    
    dt = datetime(2025, 5, 15, 10, 32, 41)
    assert safe_json_serialize(dt) == "2025-05-15T10:32:41"
    
    with pytest.raises(TypeError):
        safe_json_serialize(set([1, 2, 3]))


def test_to_json_and_from_json(tmp_path):
    data = {
        "np_int": np.int32(10),
        "np_arr": np.array([1, 2]),
        "path": Path("some/path"),
        "normal": "text"
    }
    
    json_str = to_json(data, indent=4)
    assert isinstance(json_str, str)
    parsed_str = json.loads(json_str)
    assert parsed_str["np_int"] == 10
    assert parsed_str["np_arr"] == [1, 2]
    
    # FIX: Platform-agnostic path string matching
    assert parsed_str["path"] == str(Path("some/path"))
    
    # Test serializing to file and reading back
    file_path = tmp_path / "output.json"
    to_json(data, path=str(file_path))
    assert file_path.exists()
    
    loaded_data = from_json(str(file_path))
    assert loaded_data["np_int"] == 10
    assert loaded_data["np_arr"] == [1, 2]
    # FIX: Platform-agnostic path string matching for loaded data too
    assert loaded_data["path"] == str(Path("some/path"))
    assert loaded_data["normal"] == "text"


# ── Tests for DataFrame utilities ────────────────────────────────────

def test_is_numeric_series():
    assert is_numeric_series(pd.Series([1, 2, 3])) is True
    assert is_numeric_series(pd.Series([1.1, 2.2, 3.3])) is True
    assert is_numeric_series(pd.Series([True, False, True])) is False  # Booleans are non-numeric
    assert is_numeric_series(pd.Series(["a", "b", "c"])) is False


def test_check_columns():
    df = pd.DataFrame(columns=["col1", "col2", "col3"])
    
    # Should pass without throwing exceptions
    check_columns(df, ["col1", "col2"])
    
    # Should raise a clear ValueError with details on missing columns
    with pytest.raises(ValueError) as exc_info:
        check_columns(df, ["col1", "col4", "col5"], caller="test_func")
        
    err_msg = str(exc_info.value)
    assert "[test_func]" in err_msg
    assert "Missing required columns: ['col4', 'col5']" in err_msg
    assert "Available: ['col1', 'col2', 'col3']" in err_msg


# ── Tests for Dictionary utilities ───────────────────────────────────

def test_flatten_dict():
    nested = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        },
        "f": 4
    }
    
    # Default flattening (dot separator)
    flat = flatten_dict(nested)
    expected = {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3,
        "f": 4
    }
    assert flat == expected
    
    # Custom separator character
    flat_sep = flatten_dict(nested, sep="_")
    expected_sep = {
        "a": 1,
        "b_c": 2,
        "b_d_e": 3,
        "f": 4
    }
    assert flat_sep == expected_sep
    
    # Custom global prefix
    flat_prefix = flatten_dict(nested, prefix="root")
    expected_prefix = {
        "root.a": 1,
        "root.b.c": 2,
        "root.b.d.e": 3,
        "root.f": 4
    }
    assert flat_prefix == expected_prefix


# ── Tests for Logging utility ────────────────────────────────────────

def test_get_logger():
    logger_name = "test_utils_logger"
    logger = get_logger(logger_name, level="DEBUG")
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    
    # Re-requesting the same logger should return the identical instance without repeating setup
    logger2 = get_logger(logger_name)
    assert logger is logger2
    assert len(logger.handlers) == 1  # Confirms no duplicate handlers were appended