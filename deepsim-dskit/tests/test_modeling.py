from dskit.config import validate_config
from dskit.reproducibility import run_experiment


def test_run_experiment_uses_models_config_and_saves_artifacts(sample_df, tmp_dir):
    data_path = tmp_dir / "advertising.csv"
    sample_df.to_csv(data_path)

    config = {
        "experiment_id": "unit_exp",
        "seed": 42,
        "data": {
            "url": str(data_path),
            "target": "Sales",
            "read_kwargs": {"index_col": 0},
        },
        "splitting": {"test_size": 0.2, "val_size": 0.2, "random_state": 42},
        "preprocessing": {
            "missing": {"strategies": {}, "indicator_columns": []},
            "outliers": {"columns": [], "method": "iqr", "multiplier": 1.5},
            "scaling": {
                "columns": ["TV", "Radio", "Newspaper"],
                "method": "standard",
            },
        },
        "models": {
            "linear": {"class": "LinearRegression", "params": {}},
            "ridge": {"class": "Ridge", "params": {"alpha": 1.0}},
        },
        "output": {
            "experiments_dir": str(tmp_dir / "experiments"),
            "registry_path": str(tmp_dir / "registry" / "experiments.json"),
        },
    }

    assert validate_config(config) == []

    result = run_experiment(config)

    assert result["status"] == "success"
    assert result["best_model_name"] in {"linear", "ridge"}
    assert (tmp_dir / "experiments" / "unit_exp" / "model.joblib").exists()
    assert (tmp_dir / "experiments" / "unit_exp" / "pipeline.joblib").exists()
