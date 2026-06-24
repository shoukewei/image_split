# modules/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ratio feature
def add_ratio_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    new_name: str,
    epsilon: float = 1e-6
) -> pd.DataFrame:
    """
    Add a ratio feature (col1 / col2) to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col1 : str
        Numerator column name.
    col2 : str
        Denominator column name.
    new_name : str
        Name for the new ratio column.
    epsilon : float, optional
        Small value added to the denominator to prevent division by zero.
        Default is 1e-6.

    Returns
    -------
    pd.DataFrame
        A copy of df with the new ratio column appended.
    """
    df = df.copy()
    df[new_name] = df[col1] / (df[col2] + epsilon)
    return df


# interaction features
def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add family size and solo-traveller indicator features.

    Derived from SibSp (siblings/spouses) and Parch (parents/children).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain 'SibSp' and 'Parch' columns.

    Returns
    -------
    pd.DataFrame
        A copy of df with 'FamilySize' and 'IsAlone' columns appended.
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 for the passenger
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    return df

# polynomial features
def add_polynomial_features(
    df: pd.DataFrame,
    columns: list,
    degree: int = 2
) -> pd.DataFrame:
    """
    Add polynomial (squared, cubed, ...) versions of numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns for which to add polynomial terms.
    degree : int, optional
        Maximum polynomial degree. Default is 2 (squared terms only).

    Returns
    -------
    pd.DataFrame
        A copy of df with new columns named '{col}_pow2', '{col}_pow3', etc.
    """
    df = df.copy()
    for col in columns:
        for d in range(2, degree + 1):
            df[f"{col}_pow{d}"] = df[col] ** d
    return df

# one-hot encoding
def fit_onehot_encoder(
    df: pd.DataFrame,
    columns: list
) -> dict:
    """
    Fit a one-hot encoder by recording unique categories per column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    columns : list of str
        Categorical columns to encode.

    Returns
    -------
    dict
        Mapping of column name to list of known categories
        (excluding the reference category dropped for each column).
    """
    encoder = {}
    for col in columns:
        categories = sorted(df[col].dropna().unique().tolist())
        encoder[col] = categories[1:]  # drop first as reference
    return encoder


def apply_onehot_encoder(
    df: pd.DataFrame,
    encoder: dict
) -> pd.DataFrame:
    """
    Apply a fitted one-hot encoder to a DataFrame.

    Unknown categories in the test set are treated as the reference
    category (all indicator columns = 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (train or test).
    encoder : dict
        Fitted encoder from fit_onehot_encoder().

    Returns
    -------
    pd.DataFrame
        A copy of df with original categorical columns replaced by
        binary indicator columns.
    """
    df = df.copy()
    for col, categories in encoder.items():
        for cat in categories:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
        df = df.drop(columns=[col])
    return df

## Ordinal encoding
def fit_ordinal_encoder(
    df: pd.DataFrame,
    column: str,
    order: list
) -> dict:
    """
    Fit an ordinal encoder using a specified category order.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    column : str
        Categorical column to encode.
    order : list
        Category values in ascending order of rank.

    Returns
    -------
    dict
        Mapping of category value to integer rank.
    """
    return {cat: i for i, cat in enumerate(order)}


def apply_ordinal_encoder(
    df: pd.DataFrame,
    column: str,
    mapping: dict,
    default: int = -1
) -> pd.DataFrame:
    """
    Apply a fitted ordinal encoder to a column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to encode.
    mapping : dict
        Category-to-integer mapping from fit_ordinal_encoder().
    default : int, optional
        Value assigned to unknown categories. Default is -1.

    Returns
    -------
    pd.DataFrame
        A copy of df with the column replaced by its integer encoding.
    """
    df = df.copy()
    df[column] = df[column].map(mapping).fillna(default).astype(int)
    return df

# Target encoding
def fit_target_encoder(
    df: pd.DataFrame,
    column: str,
    target: pd.Series,
    n_folds: int = 5,
    smoothing: float = 10.0,
    random_state: int = 42
) -> dict:
    """
    Fit a smoothed, cross-validated target encoder.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    column : str
        Categorical column to encode.
    target : pd.Series
        Target variable (aligned with df).
    n_folds : int, optional
        Number of cross-validation folds. Default is 5.
    smoothing : float, optional
        Smoothing factor. Higher values pull category means toward the
        global mean, reducing overfitting for rare categories.
        Default is 10.0.
    random_state : int, optional
        Random seed. Default is 42.

    Returns
    -------
    dict
        Mapping of category to smoothed target mean for test-time encoding.
    """
    global_mean = target.mean()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Compute out-of-fold encodings (used for training)
    oof_encodings = pd.Series(index=df.index, dtype=float)
    for train_idx, val_idx in kf.split(df):
        fold_train_y = target.iloc[train_idx]
        fold_train_x = df[column].iloc[train_idx]

        # Smoothed mean per category in this fold
        cat_stats = fold_train_y.groupby(fold_train_x).agg(["mean", "count"])
        smooth = 1 / (1 + np.exp(-(cat_stats["count"] - 1) / smoothing))
        cat_means = smooth * cat_stats["mean"] + (1 - smooth) * global_mean

        oof_encodings.iloc[val_idx] = (
            df[column].iloc[val_idx].map(cat_means).fillna(global_mean)
        )

    # Compute full-dataset means for test-time encoding
    cat_stats_full = target.groupby(df[column]).agg(["mean", "count"])
    smooth_full    = 1 / (1 + np.exp(-(cat_stats_full["count"] - 1) / smoothing))
    test_mapping   = (
        smooth_full * cat_stats_full["mean"] + (1 - smooth_full) * global_mean
    ).to_dict()

    return {"test_mapping": test_mapping, "global_mean": global_mean,
            "oof_encodings": oof_encodings}


def apply_target_encoder(
    df: pd.DataFrame,
    column: str,
    encoder: dict,
    is_train: bool = False
) -> pd.DataFrame:
    """
    Apply a fitted target encoder to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to encode.
    encoder : dict
        Output of fit_target_encoder().
    is_train : bool, optional
        If True, use pre-computed out-of-fold encodings for the training set.
        If False, use the test mapping. Default is False.

    Returns
    -------
    pd.DataFrame
        A copy of df with the column replaced by its target encoding.
    """
    df = df.copy()
    if is_train:
        df[column] = encoder["oof_encodings"].values
    else:
        df[column] = (
            df[column].map(encoder["test_mapping"])
            .fillna(encoder["global_mean"])
        )
    return df

# log transformation
def log_transform_feature(
    df: pd.DataFrame,
    column: str,
    new_name: str = None
) -> pd.DataFrame:
    """
    Apply log(1 + x) transformation to a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to transform.
    new_name : str, optional
        Name for the transformed column. If None, the original
        column is replaced. Default is None.

    Returns
    -------
    pd.DataFrame
        A copy of df with the transformation applied.
    """
    df = df.copy()
    name = new_name if new_name else column
    df[name] = np.log1p(df[column])
    return df

# binning
def bin_feature(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list,
    new_name: str = None
) -> pd.DataFrame:
    """
    Bin a continuous feature into discrete categories.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to bin.
    bins : list
        Bin edges (n+1 values for n bins).
    labels : list
        Category label for each bin (n labels for n bins).
    new_name : str, optional
        Name for the binned column. Defaults to '{column}_bin'.

    Returns
    -------
    pd.DataFrame
        A copy of df with the binned column appended.
    """
    df = df.copy()
    name = new_name if new_name else f"{column}_bin"
    df[name] = pd.cut(
        df[column], bins=bins, labels=labels, include_lowest=True
    )
    return df
