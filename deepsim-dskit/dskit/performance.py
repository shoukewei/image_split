import io
import cProfile
import functools
import hashlib
import pstats
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy  as np
import pandas as pd


def timer(func):
    """
    Decorator that prints the execution time of a function.

    Usage:
        @timer
        def my_function(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        ms     = (time.perf_counter() - start) * 1000
        print(f"[{func.__name__}] {ms:.2f} ms")
        return result
    return wrapper


def memory_profile(func, *args, **kwargs):
    """
    Execute a function and report its peak memory allocation.

    Parameters
    ----------
    func : callable
        Function to profile.
    *args, **kwargs
        Arguments passed to func.

    Returns
    -------
    tuple of (result, peak_mb)
        The function's return value and peak memory in MB.
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1e6
    print(f"[{func.__name__}] Peak memory: {peak_mb:.1f} MB")
    return result, peak_mb


def profile_function(func, *args, n_lines: int = 15, **kwargs):
    """
    Profile a function with cProfile and print the top n_lines by cumulative time.

    Parameters
    ----------
    func : callable
    n_lines : int
        Number of top functions to display. Default is 15.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(n_lines)
    print(stream.getvalue())
    return result


def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report memory usage per column and total.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to audit.

    Returns
    -------
    pd.DataFrame
        Columns: column, dtype, memory_mb, pct_of_total.
    """
    usage = df.memory_usage(deep=True)
    total = usage.sum() / 1e6
    rows  = [
        {"column": col, "dtype": str(df[col].dtype),
         "memory_mb": round(usage[col] / 1e6, 3),
         "pct_of_total": round(usage[col] / usage.sum() * 100, 1)}
        for col in df.columns
    ]
    result = pd.DataFrame(rows).sort_values("memory_mb", ascending=False)
    result.loc[len(result)] = {
        "column": "TOTAL", "dtype": "",
        "memory_mb": round(total, 3), "pct_of_total": 100.0,
    }
    return result.reset_index(drop=True)


def optimize_dtypes(
    df: pd.DataFrame, cardinality_threshold: int = 50
) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by optimizing column dtypes.

    Converts:
    - object columns with <= cardinality_threshold unique values → Categorical
    - float64 columns → float32 (when precision loss is acceptable)
    - int64 columns → smallest appropriate int type

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cardinality_threshold : int, optional
        Maximum unique values for object→Categorical conversion.
        Default is 50.

    Returns
    -------
    pd.DataFrame
        A copy of df with optimized dtypes.
    """
    df = df.copy()
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object" and df[col].nunique() <= cardinality_threshold:
            df[col] = df[col].astype("category")
        elif dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            mn, mx = df[col].min(), df[col].max()
            if mn >= 0:
                df[col] = df[col].astype(
                    "uint8" if mx < 255 else "uint16" if mx < 65535 else "uint32"
                )
            else:
                df[col] = df[col].astype(
                    "int8" if mn > -128 and mx < 127
                    else "int16" if mn > -32768 and mx < 32767
                    else df[col].dtype
                )
    return df


def process_in_chunks(
    file_path: str, process_func, chunksize: int = 100_000
) -> pd.DataFrame:
    """
    Apply a processing function to a CSV file in chunks.

    Loads and processes the file one chunk at a time, accumulating
    results. Peak memory usage is proportional to chunksize, not
    to the total file size.

    Parameters
    ----------
    file_path : str
        Path to a CSV file.
    process_func : callable
        Function accepting a pd.DataFrame and returning a pd.DataFrame.
        Applied to each chunk independently.
    chunksize : int, optional
        Rows per chunk. Default is 100,000.
    verbose : bool, optional
        Print progress. Default is True.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all chunks.
    """
    results = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        results.append(process_func(chunk))
    return pd.concat(results, ignore_index=True)


def parallel_standardize(
    df: pd.DataFrame, columns: list, max_workers: int = 4
) -> pd.DataFrame:
    """
    Standardize multiple columns in parallel using threads.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to standardize.
    max_workers : int, optional
        Number of parallel threads. Default is 4.

    Returns
    -------
    pd.DataFrame
        A copy of df with the specified columns standardized.
    """
    df = df.copy()
    def _worker(args):
        s, name = args
        return name, (s - s.mean()) / s.std()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = list(ex.map(_worker, [(df[col], col) for col in columns]))
    for name, scaled in futures:
        df[name] = scaled
    return df


def make_cache_key(df: pd.DataFrame) -> str:
    """
    Compute a cache key for a DataFrame based on its content hash.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to hash.

    Returns
    -------
    str
        A SHA-256 hex digest of the DataFrame's byte representation.
    """
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


class PipelineCache:
    """
    An in-memory cache for expensive pipeline operations.

    Stores results keyed by a hash of the input data, avoiding
    redundant computation when the same input is encountered twice.

    Attributes
    ----------
    cache_ : dict
        Internal cache mapping key → result.
    hits_ : int
        Number of cache hits since creation.
    misses_ : int
        Number of cache misses since creation.
    """

    def __init__(self):
        self.cache_  = {}
        self.hits_   = 0
        self.misses_ = 0

    def get_or_compute(self, key: str, compute_func, *args, **kwargs):
        """
        Return the cached result for key, or compute and cache it.

        Parameters
        ----------
        key : str
            Cache key (e.g. from make_cache_key()).
        compute_func : callable
            Function to call on a cache miss.
        *args, **kwargs
            Arguments passed to compute_func.

        Returns
        -------
        Cached or freshly computed result.
        """
        if key in self.cache_:
            self.hits_ += 1
            return self.cache_[key]

        self.misses_ += 1
        result = compute_func(*args, **kwargs)
        self.cache_[key] = result
        return result

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self.hits_ + self.misses_
        return {
            "hits":       self.hits_,
            "misses":     self.misses_,
            "hit_rate":   round(self.hits_ / total, 3) if total > 0 else 0,
            "cached_keys": len(self.cache_),
        }