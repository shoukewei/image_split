# modules/splitting.py

import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

SplitResult = namedtuple("SplitResult", ["X_train", "X_test", "y_train", "y_test"])
ThreeWaySplit = namedtuple(
    "ThreeWaySplit",
    ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
)

# Create a a basic two-way split
def create_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> SplitResult:
    """
    Split a DataFrame into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target.
    target : str
        Name of the target column.
    test_size : float, optional
        Proportion of the dataset to use as test data.
        Must be between 0 and 1. Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    SplitResult
        A named tuple with fields: X_train, X_test, y_train, y_test.

    Raises
    ------
    ValueError
        If target is not a column in df, or if test_size is
        not between 0 and 1.

    Examples
    --------
    >>> split = create_split(df, target="Sales")
    >>> split.X_train.shape
    (160, 3)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

# create a three-way split (train/val/test)
def create_three_way_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> ThreeWaySplit:
    """
    Split a DataFrame into training, validation, and test sets.

    The validation set is carved from the training portion after the
    initial train-test split.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    test_size : float, optional
        Proportion of the full dataset for the test set. Default is 0.2.
    val_size : float, optional
        Proportion of the full dataset for the validation set.
        Default is 0.1.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    ThreeWaySplit
        Named tuple with: X_train, X_val, X_test, y_train, y_val, y_test.

    Raises
    ------
    ValueError
        If test_size + val_size >= 1, leaving no training data.
    """
    if test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) must be < 1."
        )

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    X = df.drop(columns=[target])
    y = df[target]

    # First: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second: split remaining data into train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return ThreeWaySplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )

# Create a stratified split
def create_stratified_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    bins: int = None
) -> SplitResult:
    """
    Create a stratified train-test split.

    For categorical targets, stratification is applied directly.
    For continuous targets, the target is binned into quantiles
    before stratification, then the original values are used.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    test_size : float, optional
        Proportion for the test set. Default is 0.2.
    random_state : int, optional
        Random seed. Default is 42.
    bins : int, optional
        Number of quantile bins for continuous target stratification.
        If None and target is numeric, defaults to 5.

    Returns
    -------
    SplitResult
        Named tuple with X_train, X_test, y_train, y_test.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    X = df.drop(columns=[target])
    y = df[target]

    # Determine stratification variable
    if pd.api.types.is_numeric_dtype(y):
        n_bins = bins if bins is not None else 5
        strat_labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    else:
        strat_labels = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_labels
    )

    return SplitResult(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )

# Create a time-based split
def create_time_split(
    df: pd.DataFrame,
    target: str,
    date_column: str,
    test_size: float = 0.2
) -> SplitResult:
    """
    Split a time-ordered DataFrame into training and test sets.

    Observations are sorted by date_column. The last test_size
    proportion of observations (by time) form the test set.
    No random shuffling is applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date or time column.
    target : str
        Name of the target column.
    date_column : str
        Name of the column containing dates or timestamps.
        Used for sorting only — not included in features.
    test_size : float, optional
        Proportion of observations for the test set. Default is 0.2.

    Returns
    -------
    SplitResult
        Named tuple with X_train, X_test, y_train, y_test.
        The date_column is excluded from X.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found.")

    df_sorted = df.sort_values(by=date_column).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_size))

    train = df_sorted.iloc[:cutoff]
    test  = df_sorted.iloc[cutoff:]

    feature_cols = [c for c in df.columns if c not in [target, date_column]]

    return SplitResult(
        X_train=train[feature_cols],
        X_test=test[feature_cols],
        y_train=train[target],
        y_test=test[target]
    )

# Save and load splits to/from disk
def save_split(split: SplitResult, directory: str) -> None:
    """
    Save a SplitResult to a directory as CSV files.

    Creates four files: X_train.csv, X_test.csv,
    y_train.csv, y_test.csv.

    Parameters
    ----------
    split : SplitResult
        The split to save.
    directory : str
        Path to the output directory. Created if it does not exist.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    split.X_train.to_csv(path / "X_train.csv")
    split.X_test.to_csv(path / "X_test.csv")
    split.y_train.to_csv(path / "y_train.csv")
    split.y_test.to_csv(path / "y_test.csv")

    print(f"Split saved to '{directory}' "
          f"({len(split.X_train)} train, {len(split.X_test)} test rows)")


def load_split(directory: str) -> SplitResult:
    """
    Load a SplitResult from a directory of CSV files.

    Expects four files: X_train.csv, X_test.csv,
    y_train.csv, y_test.csv.

    Parameters
    ----------
    directory : str
        Path to the directory containing the split CSV files.

    Returns
    -------
    SplitResult
        The loaded split.

    Raises
    ------
    FileNotFoundError
        If any of the expected files are missing.
    """
    path = Path(directory)
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

    for fname in required:
        if not (path / fname).exists():
            raise FileNotFoundError(
                f"Expected file '{fname}' not found in '{directory}'."
            )

    return SplitResult(
        X_train=pd.read_csv(path / "X_train.csv", index_col=0),
        X_test=pd.read_csv(path / "X_test.csv",  index_col=0),
        y_train=pd.read_csv(path / "y_train.csv", index_col=0).squeeze(),
        y_test=pd.read_csv(path / "y_test.csv",  index_col=0).squeeze(),
    )

# Cross-validate a linear regression model using KFold
def cross_validate_model(
    df: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> pd.DataFrame:
    """
    Evaluate a linear regression model using k-fold cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    n_splits : int, optional
        Number of folds. Default is 5.
    random_state : int, optional
        Random seed for fold shuffling. Default is 42.
    shuffle : bool, optional
        Whether to shuffle before splitting. Default is True.
        Set to False for temporal data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with fold index, R² score, and RMSE for each fold.
    """
    X = df.drop(columns=[target])
    y = df[target]

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    model = LinearRegression()

    rows = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        r2   = model.score(X_test_cv, y_test_cv)
        rmse = np.sqrt(np.mean((y_pred - y_test_cv) ** 2))

        rows.append({"fold": fold, "r2": round(r2, 4), "rmse": round(rmse, 4)})

    results = pd.DataFrame(rows)
    results.loc[len(results)] = {
        "fold": "mean",
        "r2":   round(results["r2"].mean(), 4),
        "rmse": round(results["rmse"].mean(), 4)
    }
    return results