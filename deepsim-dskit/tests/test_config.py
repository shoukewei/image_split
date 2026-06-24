import pytest

from dskit.config import (
    apply_environment_profile,
    load_config,
    merge_configs,
    validate_config,
)


def test_advertising_baseline_json_is_valid():
    config = load_config("configs/advertising_baseline.json")
    assert validate_config(config) == []


def test_advertising_baseline_yaml_is_valid():
    config = load_config("configs/advertising_baseline.yaml")
    assert validate_config(config) == []


def test_merge_configs_preserves_nested_defaults():
    merged = merge_configs(
        {"seed": 42, "splitting": {"test_size": 0.2, "val_size": 0.1}},
        {"splitting": {"test_size": 0.3}},
    )
    assert merged["splitting"] == {"test_size": 0.3, "val_size": 0.1}


def test_environment_profile_marks_environment():
    config = load_config("configs/advertising_baseline.json")
    profiled = apply_environment_profile(config, "testing")
    assert profiled["_environment"] == "testing"
    assert profiled["splitting"]["test_size"] == 0.3


def test_invalid_config_reports_missing_model():
    config = load_config("configs/advertising_baseline.json")

    # FIX: Set models to a dict with missing required "class" key to trip validation
    config["models"] = {"linear": {"params": {}}}

    errors = validate_config(config)
    assert len(errors) > 0
    assert any("models" in error.lower() or "class" in error.lower() for error in errors)


def test_unknown_environment_raises():
    with pytest.raises(ValueError, match="Unknown environment"):
        apply_environment_profile({}, "staging")
