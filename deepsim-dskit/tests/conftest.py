# tests/conftest.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_df():
    """A small, clean DataFrame representative of the Advertising dataset."""
    return pd.DataFrame({
        "TV":        [230.1, 44.5, 17.2, 151.5, 180.8,
                       8.7, 57.5, 120.2, 8.6, 199.8],
        "Radio":     [37.8, 39.3, 45.9, 41.3, 10.8,
                      48.9, 32.8, 19.6, 2.1,  2.6],
        "Newspaper": [69.2, 45.1, 69.3, 58.5, 58.4,
                       75.1, 23.5, 11.6, 1.0,  21.2],
        "Sales":     [22.1, 10.4, 12.0, 16.5, 17.9,
                       7.2, 11.8, 13.2, 4.8, 10.6],
    })


@pytest.fixture
def sample_df_with_missing(sample_df):
    """The sample DataFrame with synthetic missing values."""
    df = sample_df.copy()
    df.loc[2, "Radio"]     = np.nan
    df.loc[5, "Newspaper"] = np.nan
    return df


@pytest.fixture
def sample_df_with_outliers(sample_df):
    """The sample DataFrame with a synthetic extreme outlier."""
    df = sample_df.copy()
    df.loc[0, "TV"] = 950.0
    return df


@pytest.fixture
def split_data(sample_df):
    """Pre-split features and target for modeling tests."""
    from sklearn.model_selection import train_test_split
    X = sample_df.drop(columns=["Sales"])
    y = sample_df["Sales"]
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def tmp_dir(tmp_path):
    """A temporary directory for artifact save/load tests."""
    return tmp_path