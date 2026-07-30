"""GFM step: one raster per flooded time-slice, empty scenes skipped."""

import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
import yaml

from flood_pipeline.config import (
    FLOOD_AREA_LAYER,
    GFM_ALGORITHM_BANDS,
    load_config,
    vector_path,
)
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


FLOODED = "2024-09-16_051230"
EMPTY_ZERO = "2024-09-18_052000"
EMPTY_NAN = "2024-09-19_050000"


def _patch(monkeypatch) -> None:
    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object()] * 3)
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: _fake_cube())


def test_run_writes_only_flooded_scenes_single_band(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    outcome = gfm.run(cfg, log=lambda _line: None)

    flooded = cfg.gfm_scene_path("ensemble", FLOODED)
    assert flooded.exists()
    assert not cfg.gfm_scene_dir("ensemble", EMPTY_ZERO).exists()
    assert not cfg.gfm_scene_dir("ensemble", EMPTY_NAN).exists()
    assert cfg.gfm_mask_path("ensemble").exists()
    assert cfg.gfm_scene_stamps("ensemble") == [FLOODED]
    assert cfg.gfm_exclusion_path("ensemble", FLOODED).exists()
    assert flooded in outcome.outputs


def test_run_raises_when_no_band_has_flood_pixels(cfg, monkeypatch) -> None:
    """FLEXTH needs flood pixels; an all-empty run must say so, not fail later."""
    _patch(monkeypatch)
    empty = _fake_cube().where(lambda cube: cube < 0)
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: empty)
    with pytest.raises(RuntimeError, match="flood extent is zero"):
        gfm.run(cfg, log=lambda _line: None)


def test_run_keeps_going_when_only_one_algorithm_band_is_empty(cfg, monkeypatch) -> None:
    """Comparing algorithms: one empty algorithm must not abort the others."""
    _patch(monkeypatch)
    cfg.gfm.compare_algorithms = True
    empty = _fake_cube().where(lambda cube: cube < 0)
    filled = _fake_cube()

    def _cube(*_a, band: str = "", **_k):
        return empty if band == GFM_ALGORITHM_BANDS["dlr"] else filled

    monkeypatch.setattr(gfm, "load_flood_cube", _cube)
    gfm.run(cfg, log=lambda _line: None)
    assert cfg.gfm_mask_path("ensemble").exists()


def test_run_writes_every_algorithm_band_when_comparing(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    cfg.gfm.compare_algorithms = True
    gfm.run(cfg, log=lambda _line: None)
    for key in ("ensemble", "dlr", "tuw", "list"):
        assert cfg.gfm_scene_path(key, FLOODED).exists()
        assert cfg.gfm_mask_path(key).exists()


def test_run_writes_flood_polygons_beside_scene_and_max(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    outcome = gfm.run(cfg, log=lambda _line: None)
    scene_polygons = vector_path(cfg.gfm_scene_path("ensemble", FLOODED))
    max_polygons = vector_path(cfg.gfm_mask_path("ensemble"))
    assert scene_polygons.exists() and scene_polygons in outcome.outputs
    assert max_polygons.exists() and max_polygons in outcome.outputs
    assert not vector_path(cfg.gfm_exclusion_path("ensemble", FLOODED)).exists()

    areas = gpd.read_file(scene_polygons, layer=FLOOD_AREA_LAYER)
    assert list(areas["area_id"]) == [1]
    assert set(areas["scene"]) == {FLOODED}


def test_run_skips_polygons_for_the_sum_raster(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    cfg.gfm.aggregation = "both"
    gfm.run(cfg, log=lambda _line: None)
    assert cfg.gfm_sum_path("ensemble").exists()
    assert not vector_path(cfg.gfm_sum_path("ensemble")).exists()


def test_run_removes_stale_scene_folders(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    stale_dir = cfg.gfm_scene_dir("ensemble", "2024-01-01_000000")
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "gfm_flood.tif").write_bytes(b"")
    gfm.run(cfg, log=lambda _line: None)
    assert not stale_dir.exists()


def test_run_removes_legacy_flat_layout(cfg, monkeypatch) -> None:
    _patch(monkeypatch)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    legacy = cfg.data_dir / "gfm_flood_2024-01-01_000000.tif"
    legacy_poly = cfg.data_dir / "gfm_flood_2024-01-01_000000.gpkg"
    legacy_excl = cfg.data_dir / "gfm_exclusion_2024-01-01_000000.tif"
    for f in (legacy, legacy_poly, legacy_excl):
        f.write_bytes(b"")
    gfm.run(cfg, log=lambda _line: None)
    assert not legacy.exists() and not legacy_poly.exists() and not legacy_excl.exists()


def test_binarize_likelihood_thresholds_and_keeps_nodata() -> None:
    grid = xr.DataArray(
        np.array([[0.0, 25.0], [80.0, np.nan]], dtype="float32"),
        dims=("y", "x"),
    )
    out = gfm._binarize_likelihood(grid, 25)
    assert out.values[0, 0] == 0.0   # below threshold
    assert out.values[0, 1] == 1.0   # == threshold -> flooded
    assert out.values[1, 0] == 1.0
    assert bool(np.isnan(out.values[1, 1]))  # nodata preserved


def test_run_thresholds_likelihood_band(cfg, monkeypatch) -> None:
    times = pd.to_datetime(["2024-09-16T05:12:30"])
    values = np.array([[[0.0, 25.0, 80.0]]], dtype="float32")  # 1 time, 1x3
    cube = xr.DataArray(
        values, dims=("time", "y", "x"),
        coords={"time": times, "y": [0.0], "x": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")

    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object()])
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: cube)
    cfg.gfm.band = "ensemble_likelihood"
    cfg.gfm.likelihood_threshold = 25

    gfm.run(cfg, log=lambda _line: None)

    scene = cfg.gfm_scene_path("ensemble_likelihood", "2024-09-16_051230")
    assert scene.exists()
    vals = set(np.unique(rioxarray.open_rasterio(scene).values).tolist())
    assert vals <= {0.0, 1.0}  # thresholded to a binary extent


def test_run_writes_raw_likelihood_for_likelihood_band(cfg, monkeypatch) -> None:
    times = pd.to_datetime(["2024-09-16T05:12:30"])
    values = np.array([[[0.0, 25.0, 80.0]]], dtype="float32")
    cube = xr.DataArray(
        values, dims=("time", "y", "x"),
        coords={"time": times, "y": [0.0], "x": [0.0, 1.0, 2.0]},
    ).rio.write_crs("EPSG:4326")
    monkeypatch.setattr(gfm, "aoi_bbox_4326", lambda _p: (0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(gfm, "search_gfm_items", lambda *a, **k: [object()])
    monkeypatch.setattr(gfm, "load_flood_cube", lambda *a, **k: cube)
    cfg.gfm.band = "ensemble_likelihood"
    cfg.gfm.likelihood_threshold = 25

    gfm.run(cfg, log=lambda _line: None)

    lk = cfg.gfm_likelihood_path("ensemble_likelihood", "2024-09-16_051230")
    assert lk.exists()
    vals = set(np.unique(rioxarray.open_rasterio(lk).values).tolist())
    assert 80.0 in vals and 25.0 in vals  # raw, not binarized


def test_run_writes_no_likelihood_for_binary_band(cfg, monkeypatch) -> None:
    _patch(monkeypatch)  # fake 0/1 cube, default ensemble_flood_extent band
    gfm.run(cfg, log=lambda _line: None)
    assert not cfg.gfm_likelihood_path("ensemble", FLOODED).exists()


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
