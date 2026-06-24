"""Run a complete dskit experiment on a tiny local Advertising-style dataset."""

from pathlib import Path
import tempfile

import pandas as pd

from dskit import run_full_pipeline


def make_advertising_sample() -> pd.DataFrame:
    """Return a small offline dataset with Advertising-like columns."""
    return pd.DataFrame(
        {
            "TV": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
            "radio": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
            "newspaper": [69.2, 45.1, 69.3, 58.5, 58.4, 75.1, 23.5, 11.6, 1.0, 21.2],
            "sales": [22.1, 10.4, 12.0, 16.5, 17.9, 7.2, 11.8, 13.2, 4.8, 10.6],
        }
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "advertising.csv"
        make_advertising_sample().to_csv(data_path)

        config = {
            "experiment_id": "advertising_example",
            "seed": 42,
            "data": {
                "path": str(data_path),
                "target": "sales",
                "read_kwargs": {"index_col": 0},
            },
            "splitting": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
            "preprocessing": {
                "missing": {"strategies": {}, "indicator_columns": []},
                "outliers": {"columns": [], "method": "iqr", "multiplier": 1.5},
                "scaling": {"columns": ["TV", "radio", "newspaper"], "method": "standard"},
            },
            "models": {
                "linear": {"class": "LinearRegression", "params": {}},
                "ridge": {"class": "Ridge", "params": {"alpha": 1.0}},
            },
            "output": {
                "experiments_dir": str(tmp_path / "experiments"),
                "registry_path": str(tmp_path / "registry" / "experiments.json"),
                "production_dir":   "production",
                "configs_dir":      "configs",
                "logs_dir":         "logs",
            },
        }

        result = run_full_pipeline(config)
        print(f"Best model: {result['best_model_name']}")
        print(f"Artifacts: {result['artifact_dir']}")


if __name__ == "__main__":
    main()
