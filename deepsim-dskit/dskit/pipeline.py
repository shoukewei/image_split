# modules/pipeline.py

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from dskit.preprocessing import (
    add_missing_indicator,
    compute_fill_values,
    fill_missing,
    compute_iqr_bounds,
    cap_with_bounds,
    compute_scaling_params,
    apply_scaling,
)
from dskit.feature_engineering import (
    add_family_features,
    log_transform_feature,
    add_polynomial_features,
    fit_onehot_encoder,
    apply_onehot_encoder,
)

from dskit.data_io import load_dataset
from dskit.eda import describe_df, missing_summary, skewness_summary

from dskit.splitting import create_split, save_split

# Modeling
from dskit.modeling import (
    ModelRegistry,
    train_model,
    evaluate_regression,
    compute_vif,
    select_significant_features
)

from dskit.persistence import save_joblib, save_metadata

from dskit.artifacts import (
    save_experiment, 
    ExperimentRegistry,
    promote_experiment, 
    compute_metrics)

from dskit.reproducibility import (
    set_global_seed,
    RunLogger,
)

from dskit.config import (
    save_config,
    load_config,
)

class PreprocessingPipeline:
    """
    A configurable, fit-transform preprocessing pipeline.

    Applies missing data handling, outlier capping, and feature scaling
    in sequence. All parameters are computed from training data only.

    Parameters
    ----------
    config : dict
        Pipeline configuration. Expected keys:
        - 'missing': dict with 'strategies' and optional 'indicator_columns'
        - 'outliers': dict with 'columns', 'method', and 'multiplier'
        - 'scaling': dict with 'columns' and 'method'

    Attributes
    ----------
    fill_values_ : dict
        Imputation values computed during fit (from training data).
    iqr_bounds_ : dict
        IQR outlier bounds computed during fit (from training data).
    scaling_params_ : dict
        Scaling parameters computed during fit (from training data).
    is_fitted_ : bool
        True after fit() has been called.

    Examples
    --------
    >>> pipeline = PreprocessingPipeline(config)
    >>> X_train_clean = pipeline.fit_transform(X_train)
    >>> X_test_clean  = pipeline.transform(X_test)
    """

    def __init__(self, config: dict, steps=None):
        self.config        = config
        # Optional custom preprocessing steps
        self.steps = steps or []
        self.fill_values_  = {}
        self.iqr_bounds_   = {}
        self.scaling_params_ = {}
        self.ohe_encoder_  = None
        self.is_fitted_    = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """
        Compute all preprocessing parameters from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Never call fit() on test data.

        Returns
        -------
        PreprocessingPipeline
            Returns self to allow method chaining.
        """
        # --- Feature engineering (fit-only steps) ---
        fe_cfg = self.config.get("feature_engineering", {})
        df_fe = df.copy()

        # Family features (deterministic)
        if fe_cfg.get("family_features"):
            df_fe = add_family_features(df_fe)

        # Log transforms (deterministic)
        for col in fe_cfg.get("log_columns", []):
            df_fe = log_transform_feature(df_fe, col, new_name=f"{col}_log")

        # Polynomial features (deterministic)
        poly_cfg = fe_cfg.get("polynomial", {})
        if poly_cfg.get("columns"):
            df_fe = add_polynomial_features(
                df_fe,
                columns=poly_cfg["columns"],
                degree=poly_cfg.get("degree", 2),
            )

        # One-hot encoder: fit on training data and store
        ohe_cols = fe_cfg.get("onehot_columns", [])
        if ohe_cols:
            self.ohe_encoder_ = fit_onehot_encoder(df_fe, ohe_cols)
            # Apply encoder to the engineered df so subsequent steps
            # (imputation/scaling) see the encoded columns
            df_fe = apply_onehot_encoder(df_fe, self.ohe_encoder_)

        # --- Missing data parameters ---
        missing_cfg = self.config.get("missing", {})
        strategies  = missing_cfg.get("strategies", {})
        if strategies:
            # compute fill values on the feature-engineered training frame
            self.fill_values_ = compute_fill_values(df_fe, strategies)

        # --- Outlier parameters ---
        outlier_cfg = self.config.get("outliers", {})
        outlier_cols = outlier_cfg.get("columns", [])
        if outlier_cols:
            self.iqr_bounds_ = compute_iqr_bounds(
                df_fe,
                columns=outlier_cols,
                multiplier=outlier_cfg.get("multiplier", 1.5),
            )

        # --- Custom pipeline steps ---
        df_fe = df.copy()

        for step in self.steps:
            step.fit(df_fe)
            df_fe = step.transform(df_fe)

        # --- Scaling parameters ---
        scaling_cfg  = self.config.get("scaling", {})
        scaling_cols = scaling_cfg.get("columns", [])

        if scaling_cols:
            self.scaling_params_ = compute_scaling_params(
                df_fe,
                columns=scaling_cols,
                method=scaling_cfg.get("method", "standard"),
            )

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing parameters to a DataFrame.

        Can be called on any split — train, validation, or test.
        Parameters are always those computed during fit().

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.

        Raises
        ------
        RuntimeError
            If transform() is called before fit().
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() on training data first."
            )

        df = df.copy()

        # --- Custom pipeline steps ---
        for step in self.steps:
            df = step.transform(df)

        # --- Feature engineering (apply with stored encoders) ---
        fe_cfg = self.config.get("feature_engineering", {})

        if fe_cfg.get("family_features"):
            df = add_family_features(df)

        for col in fe_cfg.get("log_columns", []):
            df = log_transform_feature(df, col, new_name=f"{col}_log")

        poly_cfg = fe_cfg.get("polynomial", {})
        if poly_cfg.get("columns"):
            df = add_polynomial_features(
                df,
                columns=poly_cfg["columns"],
                degree=poly_cfg.get("degree", 2),
            )

        ohe_cols = fe_cfg.get("onehot_columns", [])
        if ohe_cols:
            if self.ohe_encoder_ is None:
                raise RuntimeError("One-hot encoder not fitted. Call fit() first.")
            df = apply_onehot_encoder(df, self.ohe_encoder_)

        # Step 1: Add missing indicators (before imputation)
        missing_cfg      = self.config.get("missing", {})
        indicator_cols   = missing_cfg.get("indicator_columns", [])
        if indicator_cols:
            df = add_missing_indicator(df, indicator_cols)

        # Step 2: Impute missing values
        if self.fill_values_:
            df = fill_missing(df, self.fill_values_)

        # Step 3: Cap outliers
        if self.iqr_bounds_:
            df = cap_with_bounds(df, self.iqr_bounds_)

        # Step 4: Scale features
        if self.scaling_params_:
            df = apply_scaling(
                df,
                self.scaling_params_,
                method=self.config.get("scaling", {}).get("method", "standard"),
            )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on df and return the transformed result.

        Equivalent to fit(df).transform(df). Use on training data only.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        pd.DataFrame
            Preprocessed training data.
        """
        return self.fit(df).transform(df)

    def get_params(self) -> dict:
        """
        Return all fitted parameters as a dictionary.

        Returns
        -------
        dict
            Keys: fill_values, iqr_bounds, scaling_params.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted.")
        return {
            "fill_values":    self.fill_values_,
            "iqr_bounds":     self.iqr_bounds_,
            "scaling_params": self.scaling_params_,
        }

    def save(self, path: str) -> None:
        """
        Serialise the fitted pipeline to disk.

        Parameters
        ----------
        path : str
            File path for the saved pipeline (.pkl).
        """
        if not self.is_fitted_:
            raise RuntimeError("Fit the pipeline before saving.")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to '{path}'")

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        """
        Load a fitted pipeline from disk.

        Parameters
        ----------
        path : str
            Path to a saved pipeline file (.pkl).

        Returns
        -------
        PreprocessingPipeline
            The loaded, fitted pipeline.
        """
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from '{path}'")
        return pipeline
 
def run_feature_engineering(
    df: pd.DataFrame,
    config: dict,
    ohe_encoder: dict = None,
    is_train: bool = True
) -> tuple:
    """
    Apply configured feature engineering steps to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : dict
        Feature engineering config (the 'feature_engineering' sub-dict).
    ohe_encoder : dict, optional
        Fitted one-hot encoder. Required if onehot_columns is specified.
    is_train : bool, optional
        If True and ohe_encoder is None, fits the encoder on df.
        Default is True.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        (transformed DataFrame, fitted ohe_encoder)
    """
    df = df.copy()

    # Family features
    if config.get("family_features"):
        df = add_family_features(df)

    # Log transformations
    for col in config.get("log_columns", []):
        df = log_transform_feature(df, col, new_name=f"{col}_log")

    # Polynomial features
    poly_cfg = config.get("polynomial", {})
    if poly_cfg.get("columns"):
        df = add_polynomial_features(
            df,
            columns=poly_cfg["columns"],
            degree=poly_cfg.get("degree", 2)
        )

    # One-hot encoding
    ohe_cols = config.get("onehot_columns", [])
    if ohe_cols:
        if is_train and ohe_encoder is None:
            ohe_encoder = fit_onehot_encoder(df, ohe_cols)
        df = apply_onehot_encoder(df, ohe_encoder)

    return df, ohe_encoder

def summarise_pipeline(pipeline: PreprocessingPipeline) -> None:
    """Print a human-readable summary of a fitted `PreprocessingPipeline`.

    Mirrors the helper used in the chapter scripts so `workflow.py` can call
    `summarise_pipeline(pipeline)` after fitting.
    """
    if not getattr(pipeline, "is_fitted_", False):
        print("Pipeline is not fitted.")
        return

    config = pipeline.config
    params = pipeline.get_params()

    print("=" * 50)
    print("PREPROCESSING PIPELINE SUMMARY")
    print("=" * 50)

    # Missing data
    missing_cfg = config.get("missing", {})
    strategies = missing_cfg.get("strategies", {})
    indicators = missing_cfg.get("indicator_columns", [])
    if strategies:
        print("\n[1] Missing Data Imputation")
        for col, strategy in strategies.items():
            fitted_val = params["fill_values"].get(col, strategy)
            val_str = f"{fitted_val:.4f}" if isinstance(fitted_val, float) else str(fitted_val)
            print(f"    {col}: strategy='{strategy}', fill_value={val_str}")
        if indicators:
            print(f"    Indicators added for: {indicators}")

    # Outlier capping
    outlier_cfg = config.get("outliers", {})
    if outlier_cfg.get("columns"):
        print(f"\n[2] Outlier Capping "
              f"(method={outlier_cfg.get('method','iqr')}, "
              f"multiplier={outlier_cfg.get('multiplier', 1.5)})")
        for col, (lo, hi) in params["iqr_bounds"].items():
            print(f"    {col}: [{lo:.2f}, {hi:.2f}]")

    # Scaling
    scaling_cfg = config.get("scaling", {})
    if scaling_cfg.get("columns"):
        print(f"\n[3] Feature Scaling (method={scaling_cfg.get('method','standard')})")
        for col, p in params["scaling_params"].items():
            if scaling_cfg.get("method") == "standard":
                print(f"    {col}: mean={p['mean']:.4f}, std={p['std']:.4f}")
            else:
                print(f"    {col}: min={p['min']:.4f}, max={p['max']:.4f}")

    print("=" * 50)

from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor

MODEL_CLASSES={"LinearRegression":LinearRegression,"Ridge":Ridge,
           "RandomForestRegressor":RandomForestRegressor,
           "GradientBoostingRegressor":GradientBoostingRegressor}

def _resolve_model_class(name: str):
    # Prefer the sklearn classes from modeling module's registry via import
    # Fallback to dynamic lookup in sklearn if available.
    from modules.reproducibility import MODEL_REGISTRY
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    # try sklearn
    try:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        mapping = {
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }
        return mapping[name]
    except Exception:
        raise ValueError(f"Unknown model class: {name}")

def run_full_pipeline(config: dict) -> dict:
    """
    Execute a complete end-to-end data science pipeline.

    Orchestrates all system modules in the correct order:
    seed → load → EDA → split → preprocess → train →
    evaluate → save → register → promote.

    Parameters
    ----------
    config : dict
        Master pipeline configuration. See MASTER_CONFIG for the
        complete schema.

    Returns
    -------
    dict
        Pipeline result containing:
        - experiment_id: str
        - metrics: dict of all evaluation metrics
        - best_model_name: str
        - artifact_dir: str
        - eda_report: dict
        - log: RunLogger
        - status: 'success' or 'failed'
    """
    exp_id  = config["experiment_id"]
    out_cfg = config["output"]

    # --- Logger ---
    Path(out_cfg["logs_dir"]).mkdir(
        parents=True,
        exist_ok=True
    )
    logger = RunLogger(
        exp_id,
        log_file=f"{out_cfg['logs_dir']}/{exp_id}.jsonl"
    )
    logger.log("pipeline_started", {
        "experiment_id": exp_id,
        "config_keys":   list(config.keys()),
    })

    print(f"\n{'='*60}")
    print(f"  FULL PIPELINE: {exp_id}")
    print(f"{'='*60}")

    try:
        # ── Step 1: Seed ─────────────────────────────────────────
        print("\n[1/9] Setting global seed...")
        set_global_seed(config["seed"])
        logger.log("seed_set", {"seed": config["seed"]})

        # ── Step 2: Load data ────────────────────────────────────
        print("\n[2/9] Loading data...")
        data_cfg = config["data"]
        df = load_dataset(
            data_cfg["path"],
            index_col=data_cfg.get("index_col", 0),
            required_columns=data_cfg.get("required_columns"),
        )
        logger.log("data_loaded", {
            "rows": len(df), "columns": list(df.columns),
            "path": data_cfg["path"],
        })
        print(f"      Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── Step 3: EDA ──────────────────────────────────────────
        print("\n[3/9] Running exploratory analysis...")
        eda_report   = describe_df(df)
        missing      = missing_summary(df)
        skew         = skewness_summary(df)
        high_skew    = skew[skew["high_skew"]]["column"].tolist()

        logger.log("eda_complete", {
            "missing_columns":   missing["column"].tolist() if not missing.empty else [],
            "high_skew_columns": high_skew,
        })

        if not missing.empty:
            logger.log("missing_values_detected",
                       {"columns": missing["column"].tolist()}, level="WARNING")
            print(f"      WARNING: Missing values in {missing['column'].tolist()}")
        else:
            print("      No missing values detected.")

        if high_skew:
            print(f"      High-skew columns: {high_skew}")

        # ── Step 4: Split ────────────────────────────────────────
        print("\n[4/9] Splitting data...")
        split_cfg = config["splitting"]
        target    = data_cfg["target"]

        split = create_split(
            df,
            target=target,
            test_size=split_cfg["test_size"],
            random_state=split_cfg["random_state"],
        )
        save_split(split, f"data/splits/{exp_id}")
        logger.log("split_complete", {
            "train_rows": len(split.X_train),
            "test_rows":  len(split.X_test),
            "test_size":  split_cfg["test_size"],
        })
        print(f"      Train: {len(split.X_train)} rows  |  "
              f"Test: {len(split.X_test)} rows")

        # Validation split from training data
        X_tr, X_val, y_tr, y_val = train_test_split(
            split.X_train, split.y_train,
            test_size=split_cfg.get("val_size", 0.1),
            random_state=split_cfg["random_state"],
        )

        # ── Step 5: Preprocessing pipeline ──────────────────────
        print("\n[5/9] Fitting preprocessing pipeline...")
        pp_cfg = {k: v for k, v in config["preprocessing"].items()}

        pipeline      = PreprocessingPipeline(pp_cfg)
        X_tr_clean    = pipeline.fit_transform(X_tr)
        X_val_clean   = pipeline.transform(X_val)
        X_test_clean  = pipeline.transform(split.X_test)

        logger.log("preprocessing_complete", {
            "steps":          list(pp_cfg.keys()),
            "train_shape":    list(X_tr_clean.shape),
            "test_shape":     list(X_test_clean.shape),
        })
        print(f"      Preprocessed shape: {X_tr_clean.shape} (train), "
              f"{X_test_clean.shape} (test)")

        # VIF check for linear models
        vif_df = compute_vif(X_tr_clean)
        high_vif = vif_df[vif_df["VIF"] > 5]["feature"].tolist()
        if high_vif:
            logger.log("high_vif_detected", {"features": high_vif}, level="WARNING")
            print(f"      WARNING: High VIF features: {high_vif}")

        # ── Step 6: Feature significance check ──────────────────
        print("\n[6/9] Checking feature significance...")
        X_sig, sig_summary = select_significant_features(
            X_tr_clean, y_tr, alpha=0.05
        )
        insignificant = sig_summary[
            (~sig_summary["significant"]) &
            (sig_summary["feature"] != "const")
        ]["feature"].tolist()

        logger.log("significance_check", {
            "significant_features":   X_sig.columns.tolist(),
            "insignificant_features": insignificant,
        })
        print(f"      Significant: {X_sig.columns.tolist()}")
        if insignificant:
            print(f"      Insignificant (p > 0.05): {insignificant}")

        # ── Step 7: Model comparison ─────────────────────────────
        print("\n[7/9] Training and comparing models...")
        registry = ModelRegistry(task="regression")

        for name, model_cfg in config["models"].items():
            cls    = MODEL_CLASSES[model_cfg["class"]]
            params = model_cfg.get("params", {})
            registry.register(name, cls(**params))

        results = registry.fit_all(X_tr_clean, y_tr, X_val_clean, y_val)
        best_name, best_model = registry.get_best()

        logger.log("model_comparison_complete", {
            "models_compared": list(config["models"].keys()),
            "best_model":      best_name,
            "best_val_r2":     float(results.iloc[0]["r2"]),
        })
        print(f"\n      Model comparison (validation set):")
        print(results[["model", "r2", "rmse"]].to_string(index=False))
        print(f"\n      Best model: {best_name}")

        # ── Step 8: Final evaluation on test set ─────────────────
        print("\n[8/9] Final evaluation on test set...")
        # Refit best model on full training data
        X_train_full_clean = pipeline.transform(split.X_train)
        best_model.fit(X_train_full_clean, split.y_train)

        train_metrics = evaluate_regression(best_model, X_train_full_clean,
                                             split.y_train, label="train")
        test_metrics  = evaluate_regression(best_model, X_test_clean,
                                             split.y_test, label="test")
        all_metrics   = {**{f"train_{k}": v for k, v in train_metrics.items()
                             if k != "label"},
                         **{f"test_{k}": v for k, v in test_metrics.items()
                             if k != "label"}}

        logger.log("final_evaluation", all_metrics)
        print(f"      Test R²:   {test_metrics['r2']}")
        print(f"      Test RMSE: {test_metrics['rmse']}")
        print(f"      Test MAE:  {test_metrics['mae']}")

        # ── Step 9: Save, register, promote ─────────────────────
        print("\n[9/9] Saving artifacts and registering experiment...")
        artifact_dir = save_experiment(
            experiment_id=exp_id,
            model=best_model,
            pipeline=pipeline,
            features=list(X_train_full_clean.columns),
            config=config,
            metrics=all_metrics,
            base_dir=out_cfg["experiments_dir"],
            notes=config.get("notes", ""),
        )

        exp_registry = ExperimentRegistry(out_cfg["registry_path"])
        exp_registry.register(artifact_dir)
        promote_experiment(exp_id, exp_registry,
                           production_dir=out_cfg["production_dir"],
                           base_dir=out_cfg["experiments_dir"])

        logger.log("pipeline_complete", {
            "status":       "success",
            "artifact_dir": artifact_dir,
            "best_model":   best_name,
        })

        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE — {exp_id}")
        print(f"  Best model:  {best_name}")
        print(f"  Test R²:     {test_metrics['r2']}")
        print(f"  Artifact:    {artifact_dir}")
        print(f"{'='*60}\n")

        return {
            "experiment_id":  exp_id,
            "metrics":        all_metrics,
            "best_model_name": best_name,
            "artifact_dir":   artifact_dir,
            "eda_report":     eda_report,
            "log":            logger,
            "status":         "success",
        }

    except Exception as e:
        logger.log("pipeline_failed", {"error": str(e)}, level="ERROR")
        print(f"\nPIPELINE FAILED: {e}")
        raise
    