# modules/eda.py
import pandas as pd
from descripstats import Describe


def describe_df(df: pd.DataFrame) -> dict:
    """
    Return a structured summary of a DataFrame using extended statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to describe.

    Returns
    -------
    dict
        A dictionary containing:
        - shape: (n_rows, n_cols) tuple
        - columns: list of column names
        - dtypes: dict mapping column names to dtype strings
        - missing: dict mapping column names to missing value counts
        - summary: extended descriptive statistics (via descripstats)
    """
    return {
        "shape":   df.shape,
        "columns": list(df.columns),
        "dtypes":  {col: str(df[col].dtype) for col in df.columns},
        "missing": df.isnull().sum().to_dict(),
        "summary": Describe(df),
    }

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing values for each column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: column, missing_count, missing_pct,
        sorted by missing_pct descending. Only columns with at least
        one missing value are included.
    """
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    result = pd.DataFrame({
        "column":       missing.index,
        "missing_count": missing.values,
        "missing_pct":   pct.values,
    })

    result = result[result["missing_count"] > 0]
    return result.sort_values("missing_pct", ascending=False).reset_index(drop=True)

def skewness_summary(
    df: pd.DataFrame,
    threshold: float = 1.0
) -> pd.DataFrame:
    """
    Return skewness for all numeric columns, flagging high-skew columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.
    threshold : float, optional
        Absolute skewness above this value is flagged as high.
        Default is 1.0.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: column, skewness, high_skew,
        sorted by absolute skewness descending.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    skew_vals = df[numeric_cols].skew()

    result = pd.DataFrame({
        "column":    skew_vals.index,
        "skewness":  skew_vals.values.round(4),
        "high_skew": skew_vals.abs() > threshold,
    })

    return result.sort_values("skewness", key=abs, ascending=False).reset_index(drop=True)

def correlation_summary(
    df: pd.DataFrame,
    target: str = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Return pairwise correlations, optionally filtered by a target column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze. Only numeric columns are used.
    target : str, optional
        If provided, return correlations with this column only,
        sorted by absolute value descending.
    threshold : float, optional
        If target is None, return only pairs with absolute correlation
        above this value. Default is 0.5.

    Returns
    -------
    pd.DataFrame
        A DataFrame of correlation values.
    """
    corr = df.select_dtypes(include="number").corr()

    if target:
        result = corr[[target]].drop(index=target)
        result = result.rename(columns={target: "correlation"})
        result["abs_corr"] = result["correlation"].abs()
        return result.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")

    # Return upper triangle of pairs above threshold
    rows = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.loc[cols[i], cols[j]]
            if abs(val) >= threshold:
                rows.append({
                    "feature_1":   cols[i],
                    "feature_2":   cols[j],
                    "correlation": round(val, 4),
                })
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)