# tests/test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
from dskit.preprocessing import (
    fill_missing, compute_fill_values,
    add_missing_indicator,
    detect_outliers_iqr, cap_outliers,
    compute_scaling_params, apply_scaling,
)


class TestFillMissing:
    """Tests for fill_missing() and compute_fill_values()."""

    def test_median_fills_missing(self, sample_df_with_missing):
        """Median strategy fills all NaN values in the target column."""
        strategies = {"Radio": "median"}
        filled = fill_missing(sample_df_with_missing, strategies)
        assert filled["Radio"].isnull().sum() == 0

    def test_mean_fill_value_correct(self, sample_df_with_missing):
        """Mean fill value equals the column mean of non-null values."""
        fill_values = compute_fill_values(sample_df_with_missing, {"Radio": "mean"})
        expected = sample_df_with_missing["Radio"].mean()
        assert abs(fill_values["Radio"] - expected) < 1e-9

    def test_constant_fill(self, sample_df_with_missing):
        """Constant strategy fills NaN with the specified scalar."""
        filled = fill_missing(sample_df_with_missing, {"Newspaper": 0.0})
        assert filled["Newspaper"].isnull().sum() == 0
        assert (filled.loc[5, "Newspaper"] == 0.0)

    def test_does_not_modify_original(self, sample_df_with_missing):
        """fill_missing() returns a copy — original DataFrame is unchanged."""
        original_missing = sample_df_with_missing["Radio"].isnull().sum()
        fill_missing(sample_df_with_missing, {"Radio": "median"})
        assert sample_df_with_missing["Radio"].isnull().sum() == original_missing

    def test_missing_column_raises(self, sample_df):
        """ValueError raised if a strategy column does not exist."""
        with pytest.raises(ValueError, match="Column 'NonExistent' not found"):
            fill_missing(sample_df, {"NonExistent": "mean"})

    def test_split_safety(self, sample_df_with_missing):
        """Fill values computed from training only; test uses same values."""
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(sample_df_with_missing,
                                       test_size=0.3, random_state=42)
        fill_vals = compute_fill_values(train, {"Radio": "mean"})
        filled_test = fill_missing(test, fill_vals)
        assert filled_test["Radio"].isnull().sum() == 0


class TestMissingIndicator:
    """Tests for add_missing_indicator()."""

    def test_indicator_added(self, sample_df_with_missing):
        """Indicator column is added with correct name."""
        result = add_missing_indicator(sample_df_with_missing, ["Radio"])
        assert "Radio_missing" in result.columns

    def test_indicator_values_correct(self, sample_df_with_missing):
        """Indicator is 1 where original value was NaN, 0 otherwise."""
        result = add_missing_indicator(sample_df_with_missing, ["Radio"])
        assert result.loc[2, "Radio_missing"] == 1
        assert result.loc[0, "Radio_missing"] == 0

    def test_original_column_unchanged(self, sample_df_with_missing):
        """Adding the indicator does not impute the original column."""
        result = add_missing_indicator(sample_df_with_missing, ["Radio"])
        assert pd.isna(result.loc[2, "Radio"])


class TestOutlierDetection:
    """Tests for detect_outliers_iqr() and cap_outliers()."""

    def test_detects_synthetic_outlier(self, sample_df_with_outliers):
        """IQR method flags the known extreme TV value."""
        mask = detect_outliers_iqr(sample_df_with_outliers["TV"])
        assert mask.loc[0] == True  # noqa: E712

    def test_no_false_positives_on_clean_data(self, sample_df):
        """No outliers flagged in the clean sample DataFrame."""
        mask = detect_outliers_iqr(sample_df["TV"])
        assert mask.sum() == 0

    def test_cap_preserves_row_count(self, sample_df_with_outliers):
        """cap_outliers() returns the same number of rows."""
        capped = cap_outliers(sample_df_with_outliers, "TV")
        assert len(capped) == len(sample_df_with_outliers)

    def test_cap_reduces_maximum(self, sample_df_with_outliers):
        """cap_outliers() reduces the extreme maximum value."""
        original_max = sample_df_with_outliers["TV"].max()
        capped = cap_outliers(sample_df_with_outliers, "TV")
        assert capped["TV"].max() < original_max

    def test_cap_does_not_modify_original(self, sample_df_with_outliers):
        """cap_outliers() returns a copy — original is unchanged."""
        original_max = sample_df_with_outliers["TV"].max()
        cap_outliers(sample_df_with_outliers, "TV")
        assert sample_df_with_outliers["TV"].max() == original_max


class TestScaling:
    """Tests for compute_scaling_params() and apply_scaling()."""

    def test_standard_scaling_mean_zero(self, sample_df):
        """Standardized training column has mean ≈ 0."""
        params = compute_scaling_params(sample_df, ["TV"], method="standard")
        scaled = apply_scaling(sample_df, params, method="standard")
        assert abs(scaled["TV"].mean()) < 1e-9

    def test_standard_scaling_std_one(self, sample_df):
        """Standardized training column has std ≈ 1."""
        params = compute_scaling_params(sample_df, ["TV"], method="standard")
        scaled = apply_scaling(sample_df, params, method="standard")
        assert abs(scaled["TV"].std() - 1.0) < 1e-9

    def test_minmax_range(self, sample_df):
        """Min-max scaled training column spans [0, 1]."""
        params = compute_scaling_params(sample_df, ["TV"], method="minmax")
        scaled = apply_scaling(sample_df, params, method="minmax")
        assert abs(scaled["TV"].min()) < 1e-9
        assert abs(scaled["TV"].max() - 1.0) < 1e-9

    def test_does_not_modify_original(self, sample_df):
        """apply_scaling() returns a copy — original is unchanged."""
        original_mean = sample_df["TV"].mean()
        params = compute_scaling_params(sample_df, ["TV"])
        apply_scaling(sample_df, params)
        assert abs(sample_df["TV"].mean() - original_mean) < 1e-9

    @pytest.mark.parametrize("method", ["standard", "minmax"])
    def test_scaling_methods(self, sample_df, method):
        """Both scaling methods produce non-null output."""
        params = compute_scaling_params(sample_df, ["TV", "Radio"], method=method)
        scaled = apply_scaling(sample_df, params, method=method)
        assert scaled[["TV", "Radio"]].isnull().sum().sum() == 0