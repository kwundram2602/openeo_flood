"""Offline tests for the runner's step logic and the CLI's argument handling."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from flood_pipeline import runner
from flood_pipeline.cli import cli
from flood_pipeline.config import load_config
from flood_pipeline.runner import UnknownStepError, normalize_steps, run_pipeline
from flood_pipeline.steps import StepOutcome

VALID_CONFIG = {
    "project": {"name": "t", "gee_project": "ee-test"},
    "aoi": {"path": "aoi.geojson"},
    "flexth": {"enabled": True},
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(VALID_CONFIG), encoding="utf-8")
    (tmp_path / "aoi.geojson").write_text("{}", encoding="utf-8")
    return path


def test_normalize_steps_orders_and_dedupes() -> None:
    assert normalize_steps(["flexth", "dem", "dem", " gfm "]) == ["dem", "gfm", "flexth"]
    assert normalize_steps(["GFM"]) == ["gfm"]
    assert normalize_steps([]) == []


def test_normalize_steps_rejects_unknown() -> None:
    with pytest.raises(UnknownStepError, match="water_ballet"):
        normalize_steps(["dem", "water_ballet"])


def _fake_runners(monkeypatch, outcomes: dict[str, StepOutcome | Exception]):
    """Replace the heavy step modules with canned outcomes."""

    def fake_step_runner(step: str):
        def run_step(cfg, log):
            outcome = outcomes[step]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return run_step

    monkeypatch.setattr(runner, "_step_runner", fake_step_runner)


def test_run_pipeline_happy_path(config_file: Path, monkeypatch) -> None:
    cfg = load_config(config_file)
    _fake_runners(monkeypatch, {step: StepOutcome() for step in runner.STEP_ORDER})
    lines: list[str] = []
    assert run_pipeline(cfg, runner.STEP_ORDER, log=lines.append) == 0
    assert "##[step:dem] start" in lines
    assert "##[step:flexth] done" in lines
    assert "##[pipeline] finished" in lines


def test_run_pipeline_halts_after_drive_export(config_file: Path, monkeypatch) -> None:
    cfg = load_config(config_file)
    _fake_runners(
        monkeypatch,
        {
            "dem": StepOutcome(halt=True, message="download from Drive first"),
            "gfm": StepOutcome(),
            "flexth": StepOutcome(),
        },
    )
    lines: list[str] = []
    assert run_pipeline(cfg, runner.STEP_ORDER, log=lines.append) == 0
    assert any("halted: download from Drive first" in line for line in lines)
    assert "##[step:gfm] start" not in lines  # nothing after the halt


def test_run_pipeline_reports_step_failure(config_file: Path, monkeypatch) -> None:
    cfg = load_config(config_file)
    _fake_runners(
        monkeypatch,
        {
            "dem": StepOutcome(),
            "gfm": RuntimeError("no items found"),
            "flexth": StepOutcome(),
        },
    )
    lines: list[str] = []
    assert run_pipeline(cfg, runner.STEP_ORDER, log=lines.append) == 1
    assert any("##[step:gfm] failed: no items found" in line for line in lines)
    assert "##[step:flexth] start" not in lines


def test_run_pipeline_skips_disabled_steps(config_file: Path, monkeypatch) -> None:
    cfg = load_config(config_file)
    cfg.dem.enabled = False
    _fake_runners(monkeypatch, {step: StepOutcome() for step in runner.STEP_ORDER})
    lines: list[str] = []
    assert run_pipeline(cfg, runner.STEP_ORDER, log=lines.append) == 0
    assert "##[step:dem] skipped (disabled in config)" in lines
    assert "##[step:dem] start" not in lines


def test_run_pipeline_rejects_invalid_config(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.dem.delivery = "carrier_pigeon"
    lines: list[str] = []
    assert run_pipeline(cfg, ["flexth"], log=lines.append) == 1
    assert lines[0] == "##[pipeline] invalid config:"


def test_cli_run_rejects_unknown_step(config_file: Path) -> None:
    result = CliRunner().invoke(cli, ["run", str(config_file), "--steps", "dem,nope"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_cli_run_requires_existing_config(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["run", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0


def test_cli_run_surfaces_config_validation(config_file: Path, monkeypatch) -> None:
    broken = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    broken["dem"] = {"delivery": "carrier_pigeon"}
    config_file.write_text(yaml.safe_dump(broken), encoding="utf-8")
    result = CliRunner().invoke(cli, ["run", str(config_file), "--steps", "flexth"])
    assert result.exit_code == 1
    assert "invalid config" in result.output
