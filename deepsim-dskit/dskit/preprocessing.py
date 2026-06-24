# modules/preprocessing.py (missing data section)

import numpy as np
import pandas as pd


def drop_missing(
    df: pd.DataFrame,
    subset: list = None,
    threshold: float = None
) -> pd.DataFrame:
    """
    Drop rows with missing values, with explicit control over scope.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list of str, optional
        Columns to consider. If None, all columns are used.
    threshold : float, optional
        Drop rows where the proportion of missing values exceeds
        this threshold (0 to 1).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if threshold is not None:
        min_valid = int((1 - threshold) * df.shape[1])
        return df.dropna(thresh=min_valid, subset=subset)
    return df.dropna(subset=subset)


def add_missing_indicator(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Add binary indicator columns for missing values.

    Must be called before imputation.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns for which to add '{column}_missing' indicators.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def compute_fill_values(
    df: pd.DataFrame,
    strategies: dict
) -> dict:
    """
    Compute imputation fill values from a DataFrame.

    Always call this on training data only.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    strategies : dict
        Mapping of column name to 'mean', 'median', 'mode',
        or a scalar constant.

    Returns
    -------
    dict
        Mapping of column name to computed fill value.
    """
    fill_values = {}
    for col, strategy in strategies.items():
        if strategy == "mean":
            fill_values[col] = df[col].mean()
        elif strategy == "median":
            fill_values[col] = df[col].median()
        elif strategy == "mode":
            fill_values[col] = df[col].mode()[0]
        else:
            fill_values[col] = strategy
    return fill_values


def fill_missing(
    df: pd.DataFrame,
    strategies: dict
) -> pd.DataFrame:
    """
    Apply column-specific imputation strategies to a DataFrame.

    When used in a train-test pipeline, pass the output of
    compute_fill_values() as the strategies argument to ensure
    fill values are derived from training data only.

    Parameters
    ----------
    df : pd.DataFrame
    strategies : dict
        Mapping of column name to strategy string ('mean',
        'median', 'mode') or scalar fill value.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(strategy)
    return df

# modules/preprocessing.py (outlier detection and treatment section)
def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a numeric series using the IQR method.

    An observation is flagged as an outlier if it falls below
    Q1 - multiplier * IQR or above Q3 + multiplier * IQR.

    Parameters
    ----------
    series : pd.Series
        A numeric series to evaluate.
    multiplier : float, optional
        IQR multiplier for determining bounds. Default is 1.5.
        Use 3.0 for a more conservative (less sensitive) threshold.

    Returns
    -------
    pd.Series
        A boolean Series: True where a value is an outlier.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 100])
    >>> detect_outliers_iqr(s)
    0    False
    1    False
    2    False
    3    False
    4     True
    dtype: bool
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in a numeric series using the Z-score method.

    An observation is flagged if its absolute Z-score exceeds
    the specified threshold.

    Parameters
    ----------
    series : pd.Series
        A numeric series to evaluate.
    threshold : float, optional
        Z-score threshold above which a value is flagged.
        Default is 3.0 (flags values more than 3 standard
        deviations from the mean).

    Returns
    -------
    pd.Series
        A boolean Series: True where a value is an outlier.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 100])
    >>> detect_outliers_zscore(s)
    0    False
    1    False
    2    False
    3    False
    4     True
    dtype: bool
    """
    mean = series.mean()
    std  = series.std()
    z    = (series - mean) / std
    return z.abs() > threshold


def outlier_summary(
    df: pd.DataFrame,
    method: str = "iqr",
    multiplier: float = 1.5,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Return a summary of outliers across all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str, optional
        Detection method: 'iqr' or 'zscore'. Default is 'iqr'.
    multiplier : float, optional
        IQR multiplier (used when method='iqr'). Default is 1.5.
    threshold : float, optional
        Z-score threshold (used when method='zscore'). Default is 3.0.

    Returns
    -------
    pd.DataFrame
        Columns: column, outlier_count, outlier_pct, lower_bound,
        upper_bound. Only columns with at least one outlier are included.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    rows = []

    for col in numeric_cols:
        series = df[col].dropna()

        if method == "iqr":
            mask = detect_outliers_iqr(series, multiplier=multiplier)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
        else:
            mask = detect_outliers_zscore(series, threshold=threshold)
            mean, std = series.mean(), series.std()
            lower = mean - threshold * std
            upper = mean + threshold * std

        count = mask.sum()
        if count > 0:
            rows.append({
                "column":        col,
                "outlier_count": int(count),
                "outlier_pct":   round(count / len(series) * 100, 2),
                "lower_bound":   round(lower, 4),
                "upper_bound":   round(upper, 4),
            })

    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    multiplier: float = 1.5,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove rows where a column contains outlier values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to check for outliers.
    method : str, optional
        Detection method: 'iqr' or 'zscore'. Default is 'iqr'.
    multiplier : float, optional
        IQR multiplier (used when method='iqr'). Default is 1.5.
    threshold : float, optional
        Z-score threshold (used when method='zscore'). Default is 3.0.

    Returns
    -------
    pd.DataFrame
        A copy of df with outlier rows removed.
    """
    if method == "iqr":
        mask = detect_outliers_iqr(df[column], multiplier=multiplier)
    else:
        mask = detect_outliers_zscore(df[column], threshold=threshold)

    return df[~mask].copy()


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    multiplier: float = 1.5,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Cap outlier values in a column at the detection boundaries.

    Values below the lower bound are replaced with the lower bound.
    Values above the upper bound are replaced with the upper bound.
    Rows are never removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to cap.
    method : str, optional
        Detection method: 'iqr' or 'zscore'. Default is 'iqr'.
    multiplier : float, optional
        IQR multiplier. Default is 1.5.
    threshold : float, optional
        Z-score threshold. Default is 3.0.

    Returns
    -------
    pd.DataFrame
        A copy of df with the column's extreme values clipped.
    """
    df = df.copy()
    series = df[column]

    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
    else:
        mean, std = series.mean(), series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

    df[column] = series.clip(lower=lower, upper=upper)
    return df


def log_transform(
    df: pd.DataFrame,
    column: str,
    offset: float = 1.0
) -> pd.DataFrame:
    """
    Apply a log(1 + x) transformation to a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to transform.
    offset : float, optional
        Value added before taking the log. Default is 1.0, which
        handles zero values (log(1 + 0) = 0). Increase for columns
        containing negative values.

    Returns
    -------
    pd.DataFrame
        A copy of df with the column log-transformed.

    Notes
    -----
    Uses np.log1p(x) = log(1 + x), which is numerically stable
    for small values near zero.
    """
    df = df.copy()
    df[column] = np.log1p(df[column] - (offset - 1))
    return df


def compute_iqr_bounds(
    df: pd.DataFrame,
    columns: list,
    multiplier: float = 1.5
) -> dict:
    """
    Compute IQR-based outlier bounds from a DataFrame.

    Always call this on training data only. Apply the resulting
    bounds to both training and test sets using cap_outliers_with_bounds().

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    columns : list of str
        Columns for which to compute bounds.
    multiplier : float, optional
        IQR multiplier. Default is 1.5.

    Returns
    -------
    dict
        Mapping of column name to (lower_bound, upper_bound) tuple.
    """
    bounds = {}
    for col in columns:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (
            Q1 - multiplier * IQR,
            Q3 + multiplier * IQR
        )
    return bounds


def cap_with_bounds(
    df: pd.DataFrame,
    bounds: dict
) -> pd.DataFrame:
    """
    Cap columns using precomputed bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    bounds : dict
        Mapping of column name to (lower_bound, upper_bound) tuple,
        as returned by compute_iqr_bounds().

    Returns
    -------
    pd.DataFrame
        A copy of df with columns clipped to their bounds.
    """
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df

# modules/preprocessing.py (feature scaling section)

import pandas as pd


def compute_scaling_params(
    df: pd.DataFrame,
    columns: list,
    method: str = "standard"
) -> dict:
    """
    Compute scaling parameters from training data.

    Always call on training data only. Apply with apply_scaling().

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    columns : list of str
        Columns to compute parameters for.
    method : str, optional
        'standard' (mean, std) or 'minmax' (min, max).

    Returns
    -------
    dict
        Mapping of column name to parameter dict.
    """
    params = {}
    for col in columns:
        if method == "standard":
            params[col] = {"mean": df[col].mean(), "std": df[col].std()}
        elif method == "minmax":
            params[col] = {"min": df[col].min(), "max": df[col].max()}
        else:
            raise ValueError(f"Unknown method: '{method}'.")
    return params


def apply_scaling(
    df: pd.DataFrame,
    params: dict,
    method: str = "standard"
) -> pd.DataFrame:
    """
    Apply precomputed scaling parameters to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (train or test).
    params : dict
        Parameters from compute_scaling_params().
    method : str, optional
        'standard' or 'minmax'. Must match compute_scaling_params().

    Returns
    -------
    pd.DataFrame
        A copy of df with specified columns scaled.
    """
    df = df.copy()
    for col, p in params.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
        if method == "standard":
            df[col] = (df[col] - p["mean"]) / p["std"]
        elif method == "minmax":
            df[col] = (df[col] - p["min"]) / (p["max"] - p["min"])
    return df


class StandardScaler:
    """
    Standardize features to zero mean and unit variance.

    Fit on training data; transform both train and test.
    """

    def __init__(self):
        self.means_   = {}
        self.stds_    = {}
        self.columns_ = []

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        """Compute mean and std from training data."""
        self.columns_ = df.select_dtypes(include="number").columns.tolist()
        for col in self.columns_:
            self.means_[col] = df[col].mean()
            self.stds_[col]  = df[col].std()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted parameters to any split."""
        if not self.columns_:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        for col in self.columns_:
            if col in df.columns:
                df[col] = (df[col] - self.means_[col]) / self.stds_[col]
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform training data in one call."""
        return self.fit(df).transform(df)

def standardize_polars(
    df,
    columns: list
) -> "pl.DataFrame":
    """
    Standardize numeric columns in a Polars DataFrame to zero mean and unit variance.

    This function accepts either a Polars `DataFrame` or a pandas `DataFrame`.
    If given a pandas DataFrame it will be converted to Polars first. The
    result is always a Polars DataFrame.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        Input DataFrame to standardize.
    columns : list of str
        Columns to standardize.

    Returns
    -------
    polars.DataFrame
        A new Polars DataFrame with the specified columns standardized.
    """
    try:
        import polars as pl
    except Exception as exc:  # pragma: no cover - environment may not have polars
        raise ImportError("polars is required for standardize_polars: pip install polars") from exc

    # Accept pandas DataFrame input for convenience
    if isinstance(df, pd.DataFrame):
        df_pl = pl.from_pandas(df)
    else:
        df_pl = df

    exprs = []
    for col in columns:
        if col not in df_pl.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        exprs.append(((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col))

    return df_pl.with_columns(exprs)


def fill_missing_polars(
    df,
    strategies: dict
) -> "pl.DataFrame":
    """
    Impute missing values in a Polars DataFrame according to column-specific strategies.

    Strategies may be the strings 'mean', 'median', 'mode', or a literal value.
    The function accepts either a Polars or pandas DataFrame and returns a
    Polars DataFrame.

    Note: this implementation computes scalar fill values using the pandas
    helper `compute_fill_values()` for correctness and simplicity; for very
    large datasets you can compute aggregates directly in Polars if desired.
    """
    try:
        import polars as pl
    except Exception as exc:  # pragma: no cover - environment may not have polars
        raise ImportError("polars is required for fill_missing_polars: pip install polars") from exc

    # Convert to Polars if necessary
    input_was_pandas = False
    if isinstance(df, pd.DataFrame):
        input_was_pandas = True
        df_pl = pl.from_pandas(df)
    else:
        df_pl = df

    # Compute concrete fill values using the existing pandas helper for robust
    # semantics (mode handling, etc.). This keeps behavior identical to
    # fill_missing() while enabling Polars-based imputation.
    fill_values = compute_fill_values(df_pl.to_pandas(), strategies)

    exprs = []
    for col, val in fill_values.items():
        if col not in df_pl.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        exprs.append(pl.when(pl.col(col).is_null()).then(pl.lit(val)).otherwise(pl.col(col)).alias(col))

    return df_pl.with_columns(exprs)