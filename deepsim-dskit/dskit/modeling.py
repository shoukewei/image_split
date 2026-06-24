# modules/modeling.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


def train_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "model"
):
    """
    Fit a scikit-learn compatible model on training data.

    Parameters
    ----------
    model : sklearn estimator
        An unfitted scikit-learn compatible model with a fit() method.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    model_name : str, optional
        Label for logging. Default is 'model'.

    Returns
    -------
    Fitted model object.

    Raises
    ------
    ValueError
        If X_train and y_train have different numbers of rows.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    model.fit(X_train, y_train)
    print(f"Trained: {model_name} ({type(model).__name__})")
    return model


def evaluate_regression(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "test"
) -> dict:
    """
    Evaluate a regression model and return a metrics dictionary.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained model with a predict() method.
    X : pd.DataFrame
        Features to predict on.
    y : pd.Series
        True target values.
    label : str, optional
        Label for the split being evaluated ('train' or 'test').
        Default is 'test'.

    Returns
    -------
    dict
        Keys: label, r2, rmse, mae, mape.
    """
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))

    return {
        "label": label,
        "r2":    round(r2_score(y, preds), 4),
        "rmse":  round(rmse, 4),
        "mae":   round(mean_absolute_error(y, preds), 4),
        "mape":  round(mean_absolute_percentage_error(y, preds), 4),
    }


def evaluate_classification(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "test"
) -> dict:
    """
    Evaluate a classification model and return a metrics dictionary.

    Parameters
    ----------
    model : fitted sklearn estimator
        A trained classifier with predict() and predict_proba() methods.
    X : pd.DataFrame
        Features to predict on.
    y : pd.Series
        True binary target values.
    label : str, optional
        Label for the split. Default is 'test'.

    Returns
    -------
    dict
        Keys: label, accuracy, precision, recall, f1, roc_auc.
    """
    preds = model.predict(X)
    proba = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "label":     label,
        "accuracy":  round(accuracy_score(y, preds), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall":    round(recall_score(y, preds, zero_division=0), 4),
        "f1":        round(f1_score(y, preds, zero_division=0), 4),
    }

    if proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y, proba), 4)

    return metrics


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for all columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of numeric predictor variables. Do not include the target.

    Returns
    -------
    pd.DataFrame
        Columns: feature, VIF — sorted by VIF descending.
    """
    X = df.select_dtypes(include="number").dropna()
    rows = [
        {"feature": col,
         "VIF": round(variance_inflation_factor(X.values, i), 4)}
        for i, col in enumerate(X.columns)
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )


def select_significant_features(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05
) -> tuple:
    """
    Select features with statistically significant OLS regression coefficients.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor variables (numeric).
    y : pd.Series
        Target variable.
    alpha : float, optional
        Significance threshold. Features with p-value > alpha are removed.
        Default is 0.05.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (X with only significant features, OLS summary table with p-values)
    """
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()

    summary = pd.DataFrame({
        "feature":   X_const.columns,
        "coef":      ols_model.params.round(4),
        "p_value":   ols_model.pvalues.round(4),
        "significant": ols_model.pvalues < alpha,
    }).reset_index(drop=True)

    # Exclude the constant from feature selection
    sig_features = summary[
        (summary["feature"] != "const") & (summary["significant"])
    ]["feature"].tolist()

    return X[sig_features], summary


def save_model(model, path: str) -> None:
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : fitted estimator
        Any sklearn-compatible trained model.
    path : str
        Destination file path (.joblib recommended).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to '{path}'")


def load_model(path: str):
    """
    Load a model from disk.

    Parameters
    ----------
    path : str
        Path to a saved model file.

    Returns
    -------
    Fitted model object.
    """
    model = joblib.load(path)
    print(f"Model loaded from '{path}'")
    return model

class ModelRegistry:
    """
    A registry for training and comparing multiple candidate models.

    Parameters
    ----------
    task : str
        'regression' or 'classification'. Determines the evaluation function.

    Attributes
    ----------
    models_ : dict
        Registered model instances.
    results_ : pd.DataFrame
        Evaluation results after fit_all() is called.
    best_name_ : str
        Name of the best-performing model.
    best_model_ : fitted estimator
        The best-performing fitted model.
    """

    def __init__(self, task: str = "regression"):
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self.task       = task
        self.models_    = {}
        self.results_   = None
        self.best_name_ = None
        self.best_model_ = None

    def register(self, name: str, model) -> "ModelRegistry":
        """
        Add a model to the registry.

        Parameters
        ----------
        name : str
            Identifier for this model.
        model : sklearn estimator
            An unfitted model instance.

        Returns
        -------
        ModelRegistry
            Returns self to allow method chaining.
        """
        self.models_[name] = model
        return self

    def fit_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> pd.DataFrame:
        """
        Train all registered models and evaluate on the validation set.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data

        Returns
        -------
        pd.DataFrame
            Sorted evaluation results for all models.
        """
        rows = []

        for name, model in self.models_.items():
            model = train_model(model, X_train, y_train, model_name=name)

            if self.task == "regression":
                metrics = evaluate_regression(model, X_val, y_val, label=name)
                sort_col, ascending = "rmse", True
            else:
                metrics = evaluate_classification(model, X_val, y_val, label=name)
                sort_col, ascending = "f1", False

            self.models_[name] = model  # store fitted model
            rows.append(metrics)

        self.results_ = (
            pd.DataFrame(rows)
            .rename(columns={"label": "model"})
            .sort_values(sort_col, ascending=ascending)
            .reset_index(drop=True)
        )

        best_row = self.results_.iloc[0]
        self.best_name_  = best_row["model"]
        self.best_model_ = self.models_[self.best_name_]

        return self.results_

    def get_best(self):
        """Return the name and fitted instance of the best model."""
        if self.best_model_ is None:
            raise RuntimeError("Call fit_all() before get_best().")
        return self.best_name_, self.best_model_
