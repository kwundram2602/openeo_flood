"""GFM step: one raster per flooded time-slice, empty scenes skipped."""

import datetime as dt
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
    """Three slices: one with flood, one all-zero (no flood), one all-NaN."""
    times = pd.to_datetime(
        ["2024-09-16T05:12:30", "2024-09-18T05:20:00", "2024-09-19T05:00:00"]
    )
    flooded = np.ones((3, 3), dtype="float32")
    zeros = np.zeros((3, 3), dtype="float32")
    nans = np.full((3, 3), np.nan, dtype="float32")
    values = np.stack([flooded, zeros, nans])
    cube = xr.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={"time": times, "y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]},
    )
    return cube.rio.write_crs("EPSG:4326")


def test_run_writes_only_flooded_scenes(cfg, monkeypatch) -> None:
    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object()] * 3)
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: _fake_cube())

    outcome = gfm.run(cfg, log=lambda _line: None)

    flooded = cfg.gfm_scene_path("2024-09-16_051230")
    empty_zero = cfg.gfm_scene_path("2024-09-18_052000")
    empty_nan = cfg.gfm_scene_path("2024-09-19_050000")
    assert flooded.exists()
    assert not empty_zero.exists()  # covered, no flood -> skipped
    assert not empty_nan.exists()  # not covered -> skipped
    assert cfg.gfm_mask_path().exists()  # whole-time max always written
    assert cfg.gfm_scene_paths() == [flooded]
    assert flooded in outcome.outputs


def test_run_removes_stale_scene_rasters(cfg, monkeypatch) -> None:
    """A per-scene raster from a previous run is cleared before writing."""
    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object()] * 3)
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: _fake_cube())

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    stale = cfg.gfm_scene_path("2024-01-01_000000")
    stale.write_bytes(b"")

    gfm.run(cfg, log=lambda _line: None)
    assert not stale.exists()


def test_has_flood_predicate() -> None:
    grid = xr.DataArray(np.array([[0.0, 1.0], [np.nan, 0.0]], dtype="float32"))
    assert gfm._has_flood(grid) is True
    assert gfm._has_flood(grid.where(grid > 5)) is False  # all-NaN
    assert gfm._has_flood(xr.zeros_like(grid)) is False


class _Item:
    def __init__(self, when):
        self.datetime = when


def test_cap_items_keeps_newest_and_warns() -> None:
    items = [
        _Item(dt.datetime(2024, 9, d, tzinfo=dt.timezone.utc)) for d in (3, 10, 20)
    ]
    logs: list[str] = []
    kept = gfm._cap_items(items, 2, logs.append)
    assert [i.datetime.day for i in kept] == [20, 10]
    assert any("max_items" in m for m in logs)


def test_cap_items_unlimited_returns_all() -> None:
    items = [_Item(dt.datetime(2024, 9, 3, tzinfo=dt.timezone.utc))]
    logs: list[str] = []
    assert gfm._cap_items(items, 0, logs.append) == items
    assert logs == []  # no truncation warning
