# tests/test_pipeline.py

import pandas as pd
import pytest
from dskit.pipeline import PreprocessingPipeline


# A lightweight duck-typed step class to replace the removed framework class
class MockFunctionStep:
    """A simple inline step that wraps a function for testing custom pipeline steps."""
    def __init__(self, func):
        self.func = func

    def fit(self, df: pd.DataFrame) -> "MockFunctionStep":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.func(df)


def test_pipeline_runs_custom_steps_before_scaling(sample_df):
    """
    Ensures that custom pipeline steps are executed and their engineered features
    are available and preserved for subsequent steps like scaling.
    """
    def add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
        df["total_spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
        return df

    # Configure scaling to specifically look for the column created by the custom step
    pipeline = PreprocessingPipeline(
        config={
            "scaling": {"columns": ["total_spend"], "method": "standard"},
        },
        steps=[MockFunctionStep(add_total_spend)],
    )

    # Process data (dropping 'Sales' assuming sample_df represents features + target)
    features_df = sample_df.drop(columns=["Sales"]) if "Sales" in sample_df.columns else sample_df
    result = pipeline.fit_transform(features_df)

    # Assertions
    assert "total_spend" in result.columns, "Custom step feature 'total_spend' was not created or was wiped out."
    # Standard scaled features should have a mean close to 0
    assert abs(result["total_spend"].mean()) < 1e-9, f"Scaling failed. Mean is {result['total_spend'].mean()}"


def test_pipeline_throws_error_when_transforming_unfitted():
    """
    Ensures that calling transform on an unfitted pipeline raises a RuntimeError.
    """
    pipeline = PreprocessingPipeline(config={})
    df = pd.DataFrame({"A": [1, 2, 3]})
    
    with pytest.raises(RuntimeError, match="Pipeline is not fitted"):
        pipeline.transform(df)