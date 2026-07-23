"""GFM step: writes one raster per cube time-slice plus the whole-time max."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import yaml

from flood_pipeline.config import load_config
from flood_pipeline.steps import gfm

CONFIG = {
    "project": {"name": "gfm_test", "gee_project": "ee-test"},
    "aoi": {"path": "aoi.geojson"},
    "gfm": {"temporal_extent": ["2024-09-15", "2024-09-20"], "aggregation": "max"},
}


@pytest.fixture
def cfg(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    (tmp_path / "aoi.geojson").write_text("{}", encoding="utf-8")
    return load_config(path)


def _fake_cube() -> xr.DataArray:
    times = pd.to_datetime(["2024-09-16T05:12:30", "2024-09-18T05:20:00"])
    values = np.arange(2 * 3 * 3, dtype="float32").reshape(2, 3, 3)
    cube = xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )
    return cube.rio.write_crs("EPSG:4326")


def test_run_writes_one_raster_per_time_slice_plus_max(cfg, monkeypatch) -> None:
    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object(), object()])
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: _fake_cube())

    outcome = gfm.run(cfg, log=lambda _line: None)

    scene_a = cfg.gfm_scene_path("2024-09-16_051230")
    scene_b = cfg.gfm_scene_path("2024-09-18_052000")
    assert scene_a.exists()
    assert scene_b.exists()
    assert cfg.gfm_mask_path().exists()
    assert set(cfg.gfm_scene_paths()) == {scene_a, scene_b}
    assert cfg.gfm_mask_path() in outcome.outputs
    assert scene_a in outcome.outputs
