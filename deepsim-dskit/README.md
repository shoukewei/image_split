# deepsim-dskit - A Reusable Data Science Framework

`deepsim-dskit` is an installable Python package for reproducible, configuration-driven
data science pipelines. It provides reusable building blocks for loading data,
preprocessing, splitting, modeling, artifact management, and experiment runs.

The package is still lightweight enough to use as a toolkit, but it now also has a small framework layer: you can register custom pipeline steps, declare them in configuration, and let `dskit` call them during execution.

## Installation

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "deepsim-dskit[polars]"
pip install "deepsim-dskit[yaml]"
```

## Quick Start

```python
from dskit import load_dataset, create_split

df = load_dataset("data/advertising.csv", index_col=0)
split = create_split(df, target="sales", test_size=0.2, random_state=42)
```

Run a full experiment from a config dictionary:

```python
from dskit import run_full_pipeline

config = {
    "experiment_id": "advertising_baseline",
    "seed": 42,
    "data": {
        "path": "data/advertising.csv",
        "target": "sales",
        "read_kwargs": {"index_col": 0},
    },
    "splitting": {"test_size": 0.2, "val_size": 0.1, "random_state": 42},
    "preprocessing": {
        "missing": {"strategies": {}, "indicator_columns": []},
        "outliers": {"columns": [], "method": "iqr", "multiplier": 1.5},
        "scaling": {"columns": ["tv", "radio", "newspaper"], "method": "standard"},
    },
    "models": {
        "linear": {"class": "LinearRegression", "params": {}},
        "ridge": {"class": "Ridge", "params": {"alpha": 1.0}},
    },
    "output": {
        "experiments_dir": "experiments",
        "registry_path": "registry/experiments.json",
    },
}

result = run_full_pipeline(config)
print(result["best_model_name"])
print(result["metrics"]["test_r2"])
```

## CLI

```bash
dskit-run --version
dskit-run --config configs/advertising.json --dry-run
dskit-run --config configs/advertising.json --env production
```

## Custom Framework Steps

```python
import pandas as pd
from dskit import register_function_step, PreprocessingPipeline

def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    df["total_spend"] = df["tv"] + df["radio"] + df["newspaper"]
    return df

register_function_step("total_spend", add_total_spend)

pipeline = PreprocessingPipeline({
    "steps": [{"name": "total_spend"}],
    "scaling": {"columns": ["total_spend"], "method": "standard"},
})
```

## What's Included

| Module | Purpose |
|---|---|
| `data_io` | Load, validate, and save datasets |
| `eda` | Exploratory summaries |
| `preprocessing` | Imputation, outlier treatment, scaling |
| `splitting` | Reproducible train/test/validation splits |
| `pipeline` | Fit/transform preprocessing pipeline |
| `framework` | Custom step registry and extension points |
| `feature_engineering` | Encoding and feature construction |
| `modeling` | Training, evaluation, and `ModelRegistry` |
| `persistence` | Save and load artifacts |
| `artifacts` | Experiment artifacts and registry helpers |
| `reproducibility` | Config-driven experiment execution |
| `config` | Config validation and environment profiles |
| `performance` | Profiling and optimization helpers |

## License

MIT License. See `LICENSE`.

## Author

Shouke Wei, PhD · [Deepsim Press Author Page](https://press.deepsim.ca/shouke/)
Affiliation: Deepsim Intelligence Technology Inc. [deepsim.ca](https://deepsim.ca)
