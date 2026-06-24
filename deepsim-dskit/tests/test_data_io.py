import pytest
import pandas as pd
import numpy as np
from dskit.data_io import load_dataset, validate_columns, validate_dtypes


class TestLoadDataset:
    """Tests for the load_dataset() function."""

    ADVERTISING_URL = (
        "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    )

    def test_loads_csv_from_url(self):
        """load_dataset() returns a non-empty DataFrame from a valid URL."""
        df = load_dataset(self.ADVERTISING_URL, index_col=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_correct_shape(self):
        """Advertising dataset has 200 rows and 4 columns."""
        df = load_dataset(self.ADVERTISING_URL, index_col=0)
        assert df.shape == (200, 4)

    def test_expected_columns(self):
        """All expected columns are present."""
        df = load_dataset(self.ADVERTISING_URL, index_col=0)
        assert set(df.columns) == {"TV", "radio", "newspaper", "sales"}

    def test_required_columns_valid(self):
        """No error when all required columns are present."""
        df = load_dataset(
            self.ADVERTISING_URL, index_col=0,
            required_columns=["TV", "radio", "sales"]
        )
        assert "TV" in df.columns

    def test_required_columns_missing_raises(self):
        """ValueError raised when a required column is absent."""
        with pytest.raises(ValueError, match="Missing required columns"):
            load_dataset(
                self.ADVERTISING_URL, index_col=0,
                required_columns=["TV", "NonExistentColumn"]
            )

    def test_no_missing_values_in_advertising(self):
        """The Advertising dataset has no missing values."""
        df = load_dataset(self.ADVERTISING_URL, index_col=0)
        assert df.isnull().sum().sum() == 0

    def test_unsupported_format_raises(self, tmp_dir):
        """ValueError raised for unsupported file formats."""
        bad_file = tmp_dir / "data.xyz"
        bad_file.write_text("col1,col2\n1,2")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(str(bad_file))


class TestValidateColumns:
    """Tests for validate_columns()."""

    def test_valid_columns_no_error(self, sample_df):
        """No error when all required columns are present."""
        validate_columns(sample_df, ["TV", "Radio"])

    def test_missing_column_raises(self, sample_df):
        """ValueError raised with the missing column name."""
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(sample_df, ["TV", "NonExistent"])

    def test_error_message_lists_missing_columns(self, sample_df):
        """Error message lists all missing columns, not just the first."""
        with pytest.raises(ValueError) as exc_info:
            validate_columns(sample_df, ["TV", "Missing1", "Missing2"])
        assert "Missing1" in str(exc_info.value)
        assert "Missing2" in str(exc_info.value)