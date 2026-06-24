# tests/test_splitting.py

import pytest
import pandas as pd
from dskit.splitting import create_split, create_stratified_split, save_split, load_split


class TestCreateSplit:
    """Tests for create_split()."""

    def test_returns_named_tuple(self, sample_df):
        """create_split() returns a SplitResult named tuple."""
        from dskit.splitting import SplitResult
        split = create_split(sample_df, target="Sales")
        assert isinstance(split, SplitResult)

    def test_correct_proportions(self, sample_df):
        """Test set size matches requested test_size."""
        split = create_split(sample_df, target="Sales", test_size=0.3)
        total = len(sample_df)
        assert len(split.X_test) == int(total * 0.3)

    def test_no_overlap_between_splits(self, sample_df):
        """Train and test sets have no index overlap."""
        split = create_split(sample_df, target="Sales")
        train_idx = set(split.X_train.index)
        test_idx  = set(split.X_test.index)
        assert len(train_idx & test_idx) == 0

    def test_target_column_excluded_from_features(self, sample_df):
        """The target column is not in X_train or X_test."""
        split = create_split(sample_df, target="Sales")
        assert "sales" not in split.X_train.columns
        assert "sales" not in split.X_test.columns

    def test_invalid_target_raises(self, sample_df):
        """ValueError raised when target column does not exist."""
        with pytest.raises(ValueError, match="Target 'NonExistent' not found"):
            create_split(sample_df, target="NonExistent")

    def test_invalid_test_size_raises(self, sample_df):
        """ValueError raised when test_size is outside (0, 1)."""
        with pytest.raises(ValueError):
            create_split(sample_df, target="Sales", test_size=1.5)

    def test_reproducible_with_seed(self, sample_df):
        """Same random_state produces identical splits."""
        split1 = create_split(sample_df, target="Sales", random_state=42)
        split2 = create_split(sample_df, target="Sales", random_state=42)
        assert split1.X_train.index.tolist() == split2.X_train.index.tolist()

    def test_different_seeds_differ(self, sample_df):
        """Different random states produce different splits."""
        split1 = create_split(sample_df, target="Sales", random_state=1)
        split2 = create_split(sample_df, target="Sales", random_state=99)
        assert split1.X_train.index.tolist() != split2.X_train.index.tolist()


class TestSaveLoadSplit:
    """Tests for save_split() and load_split()."""

    def test_save_creates_four_files(self, sample_df, tmp_dir):
        """save_split() creates X_train, X_test, y_train, y_test CSV files."""
        split = create_split(sample_df, target="Sales")
        save_split(split, str(tmp_dir))
        for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            assert (tmp_dir / fname).exists()

    def test_load_restores_identical_data(self, sample_df, tmp_dir):
        """Loaded split is identical to the original split."""
        split = create_split(sample_df, target="Sales")
        save_split(split, str(tmp_dir))
        loaded = load_split(str(tmp_dir))
        assert split.X_train.equals(loaded.X_train)
        assert split.y_test.equals(loaded.y_test)

    def test_load_missing_file_raises(self, tmp_dir):
        """FileNotFoundError raised when a required file is missing."""
        with pytest.raises(FileNotFoundError):
            load_split(str(tmp_dir))