"""Command-line interface for dskit."""

import argparse
import sys


def main() -> None:
    """Main entry point for the dskit-run command."""
    parser = argparse.ArgumentParser(
        description="Run a dskit experiment pipeline from a configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dskit-run --version
  dskit-run --config configs/advertising.json --dry-run
  dskit-run --config configs/advertising.json --env production
        """,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to experiment configuration file (.json or .yaml)",
    )
    parser.add_argument(
        "--env", "-e",
        default=None,
        choices=["development", "testing", "production"],
        help="Environment profile to apply",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print summary without running the pipeline",
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Print dskit version and exit",
    )

    args = parser.parse_args()

    if args.version:
        import dskit

        print(f"dskit version {dskit.__version__}")
        sys.exit(0)

    if not args.config:
        parser.error("--config is required unless --version is used")

    from dskit.config import apply_environment_profile, load_config, validate_config

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.env:
        config = apply_environment_profile(config, args.env)

    errors = validate_config(config)
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        models = config.get("models") or {"model": config.get("model", {})}
        print(f"Configuration valid: '{args.config}'")
        print(f"Experiment ID:       {config.get('experiment_id')}")
        print(f"Model(s):            {list(models.keys())}")
        print(f"Environment:         {config.get('_environment', 'development')}")
        print("Dry run complete - no pipeline executed.")
        sys.exit(0)

    from dskit.reproducibility import run_experiment

    try:
        result = run_experiment(config)
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)

    if result["status"] != "success":
        print("Pipeline failed.", file=sys.stderr)
        sys.exit(1)

    metric = result["metrics"].get(
        "test_r2", result["metrics"].get("test_accuracy", "N/A")
    )
    print(f"\nSuccess: {result['experiment_id']}")
    print(f"Best model: {result.get('best_model_name', 'N/A')}")
    print(f"Primary test metric: {metric}")
    sys.exit(0)


if __name__ == "__main__":
    main()
