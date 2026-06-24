# dskit/toolkit.py

from pathlib import Path
from dskit.utils       import get_logger, ensure_dir
from dskit.config      import (load_config, validate_config,
                                apply_environment_profile,
                                merge_configs, get_default_config)
from dskit.data_io     import load_dataset
from dskit.eda         import describe_df, missing_summary, skewness_summary
from dskit.splitting   import create_split, save_split
from dskit.pipeline    import PreprocessingPipeline
from dskit.modeling    import ModelRegistry, evaluate_regression
from dskit.artifacts   import ExperimentRegistry, save_experiment, promote_experiment
from dskit.persistence import ArtifactStore
from dskit.reproducibility import set_global_seed


class Toolkit:
    """
    A unified, configured entry point to the dskit data science toolkit.

    Bundles configuration, logging, and registry management into one
    object. Useful for projects that want a single configured instance
    rather than managing imports and configuration separately.

    Parameters
    ----------
    config : dict or str, optional
        Experiment configuration dictionary, or path to a JSON/YAML
        config file. If None, uses default configuration.
    environment : str, optional
        Environment profile to apply ('development', 'testing',
        'production'). Default reads from DS_ENVIRONMENT env var.
    log_level : str, optional
        Logging level for all toolkit operations. Default is 'INFO'.

    Attributes
    ----------
    config : dict
        The active (merged, environment-applied) configuration.
    logger : logging.Logger
        Toolkit-level logger.
    registry : ExperimentRegistry or None
        The experiment registry (initialised on first use).

    Examples
    --------
    >>> kit = Toolkit("configs/advertising.json", environment="production")
    >>> df  = kit.load("https://...")
    >>> split = kit.split(df, target="Sales")
    >>> result = kit.run(split)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config=None,
        environment: str = None,
        log_level: str = "INFO"
    ):
        self.logger   = get_logger("dskit.toolkit", level=log_level)
        self.registry = None

        # Load config
        if isinstance(config, str):
            raw_config = load_config(config)
            self.logger.info(f"Config loaded from '{config}'")
        elif isinstance(config, dict):
            raw_config = config
        else:
            raw_config = get_default_config()
            self.logger.info("Using default configuration")

        # Apply environment profile
        self.config = apply_environment_profile(raw_config, environment)
        self.logger.info(
            f"dskit Toolkit v{self.VERSION} ready "
            f"(environment='{self.config.get('_environment', 'development')}')"
        )

    # ── Data loading ─────────────────────────────────────────────────

    def load(self, path: str, **kwargs):
        """Load a dataset using the configured backend."""
        backend = self.config.get("backend", "pandas")
        self.logger.info(f"Loading data: '{path}' (backend={backend})")
        return load_dataset(path, backend=backend, **kwargs)

    # ── EDA ──────────────────────────────────────────────────────────

    def eda(self, df, target: str = None) -> dict:
        """Run a quick EDA and return a structured report."""
        report = {
            "description":   describe_df(df),
            "missing":       missing_summary(df),
            "skewness":      skewness_summary(df),
        }
        if target:
            from dskit.eda import correlation_summary
            report["correlations"] = correlation_summary(df, target=target)
        self.logger.info(
            f"EDA complete: {df.shape}, "
            f"{report['missing'].shape[0]} missing columns, "
            f"{report['skewness']['high_skew'].sum()} high-skew columns"
        )
        return report

    # ── Splitting ────────────────────────────────────────────────────

    def split(self, df, target: str = None, **kwargs):
        """Create a train-test split using config settings."""
        target     = target or self.config.get("data", {}).get("target")
        split_cfg  = self.config.get("splitting", {})
        test_size  = kwargs.pop("test_size", split_cfg.get("test_size", 0.2))
        random_state = kwargs.pop(
            "random_state",
            split_cfg.get("random_state", self.config.get("seed", 42))
        )
        split = create_split(df, target=target,
                              test_size=test_size, random_state=random_state)
        self.logger.info(
            f"Split: {len(split.X_train)} train, {len(split.X_test)} test rows"
        )
        return split

    # ── Pipeline ─────────────────────────────────────────────────────

    def build_pipeline(self, config: dict = None) -> PreprocessingPipeline:
        """Build a preprocessing pipeline from config."""
        pp_config = config or self.config.get("preprocessing", {})
        return PreprocessingPipeline(pp_config)

    # ── Experiment registry ──────────────────────────────────────────

    def get_registry(self) -> ExperimentRegistry:
        """Return the experiment registry, initialising it if needed."""
        if self.registry is None:
            registry_path = (
                self.config.get("output", {})
                    .get("registry_path", "registry/experiments.json")
            )
            self.registry = ExperimentRegistry(registry_path)
            self.logger.info(f"Registry initialised: '{registry_path}'")
        return self.registry

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, config: dict = None) -> dict:
        """
        Execute the full pipeline from configuration.

        Parameters
        ----------
        config : dict, optional
            Override configuration. Uses self.config if None.

        Returns
        -------
        dict
            Pipeline result: experiment_id, metrics, artifact_dir, status.
        """
        from dskit.reproducibility import run_experiment
        active_config = merge_configs(self.config, config or {})
        errors = validate_config(active_config)
        if errors:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return run_experiment(active_config)

    def __repr__(self) -> str:
        exp_id = self.config.get("experiment_id", "unconfigured")
        env    = self.config.get("_environment", "development")
        return f"Toolkit(experiment='{exp_id}', environment='{env}')"
    
    # Extend Toolkit with image support (add to toolkit.py)

def split_images(
    self,
    image_dir: str,
    output_dir: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
):
    """
    Split an image dataset into train, validation, and test sets.

    Requires the image_split package (pip install image-split).

    Parameters
    ----------
    image_dir : str
        Directory containing image files, organised by class.
    output_dir : str, optional
        Output directory for split image sets. Defaults to
        'data/splits/<image_dir_name>'.
    test_size, val_size, stratify
        Same semantics as create_split() in splitting.py.
    """
    try:
        from image_split import ImageSplitter
    except ImportError:
        raise ImportError(
            "image_split is required for image splitting. "
            "Install with: pip install image-split"
        )

    output_dir = output_dir or f"data/splits/{Path(image_dir).name}"
    split_cfg  = self.config.get("splitting", {})

    splitter = ImageSplitter(
        image_dir=image_dir,
        output_dir=output_dir,
        test_size=test_size or split_cfg.get("test_size", 0.2),
        val_size=val_size or split_cfg.get("val_size", 0.1),
        random_state=self.config.get("seed", 42),
        stratify=stratify,
    )

    result = splitter.split()
    self.logger.info(
        f"Image split: {result.n_train} train, "
        f"{result.n_val} val, {result.n_test} test"
    )
    return result