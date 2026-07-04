"""Execute every dashboard page headless via Streamlit's AppTest.

This catches import errors, widget misuse and config-handling bugs without a
browser. Pages behave read-only here: no buttons are clicked, so no pipeline
runs and no network calls happen (the scenes page only searches on click).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from flood_pipeline.cli import CONFIG_ENV_VAR
from flood_pipeline.config import load_config
from flood_pipeline.steps import flexth_step

REPO_ROOT = Path(__file__).parent.parent
APP_DIR = REPO_ROOT / "src" / "flood_pipeline" / "app"
DEMO_CONFIG = REPO_ROOT / "projects" / "austria_demo" / "config.yaml"
PAGE_TIMEOUT_SECONDS = 120

PAGES = [
    APP_DIR / "Home.py",
    APP_DIR / "pages" / "1_Config.py",
    APP_DIR / "pages" / "2_AOI.py",
    APP_DIR / "pages" / "3_GFM_Scenes.py",
    APP_DIR / "pages" / "4_Run.py",
]


def _run_page(page: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(DEMO_CONFIG))
    test = AppTest.from_file(str(page), default_timeout=PAGE_TIMEOUT_SECONDS)
    test.run()
    return test


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_exception(page: Path, monkeypatch) -> None:
    result = _run_page(page, monkeypatch)
    assert not result.exception, f"{page.name} raised: {result.exception}"


@pytest.mark.skipif(
    not flexth_step.find_outputs(load_config(DEMO_CONFIG).output_dir),
    reason="needs FLEXTH outputs in the project's output dir (run the pipeline first)",
)
def test_results_page_renders_with_outputs(monkeypatch) -> None:
    result = _run_page(APP_DIR / "pages" / "5_Results.py", monkeypatch)
    assert not result.exception, f"5_Results.py raised: {result.exception}"


def test_create_project_from_template(tmp_path: Path) -> None:
    from flood_pipeline.app import ui

    template = {
        "project": {"name": "old", "gee_project": "ee-test"},
        "aoi": {"path": "somewhere.gpkg"},
        "gfm": {"temporal_extent": ["2024-09-21", "2024-09-22"]},
    }
    created = ui.create_project("new_region", tmp_path / "runs", template)
    assert created == tmp_path / "runs" / "new_region" / "config.yaml"

    cfg = load_config(created)
    assert cfg.project.name == "new_region"
    assert cfg.project.gee_project == "ee-test"  # inherited from the template
    assert cfg.aoi_path == "aoi_drawn.geojson"  # AOI starts fresh
    assert cfg.gfm.temporal_extent == ["2024-09-21", "2024-09-22"]
    assert template["project"]["name"] == "old"  # template not mutated

    with pytest.raises(FileExistsError):
        ui.create_project("new_region", tmp_path / "runs", template)
