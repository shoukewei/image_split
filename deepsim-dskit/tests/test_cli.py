import pytest

from dskit import cli


def test_cli_version_exits_successfully(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["dskit-run", "--version"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert "dskit version" in capsys.readouterr().out


def test_cli_dry_run_valid_config(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["dskit-run", "--config", "configs/advertising_baseline.json", "--dry-run"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    output = capsys.readouterr().out
    assert exc.value.code == 0
    assert "Configuration valid" in output
    assert "advertising_baseline" in output


def test_cli_requires_config_without_version(monkeypatch):
    monkeypatch.setattr("sys.argv", ["dskit-run"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
