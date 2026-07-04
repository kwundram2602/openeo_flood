"""Offline tests for the dem step's cached-DEM reuse logic.

A cached fabdem.tif is only reused when it covers the configured AOI —
drawing a new AOI must trigger a fresh download instead of silently pairing
a stale DEM with the new flood mask.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_bounds
from shapely.geometry import box

from flood_pipeline import gee
from flood_pipeline.config import load_config
from flood_pipeline.steps import dem


def _write_raster(path: Path, bounds: tuple[float, float, float, float]) -> None:
    west, south, east, north = bounds
    width = height = 50
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, width, height),
    ) as dst:
        dst.write(np.zeros((1, height, width), dtype="float32"))


def _write_aoi(path: Path, bounds: tuple[float, float, float, float]) -> None:
    frame = gpd.GeoDataFrame(geometry=[box(*bounds)], crs="EPSG:4326")
    frame.to_file(path, driver="GeoJSON")


@pytest.fixture
def cfg(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"gee_project": "ee-test"},
                "aoi": {"path": "aoi.geojson"},
            }
        ),
        encoding="utf-8",
    )
    _write_aoi(tmp_path / "aoi.geojson", (15.0, 48.0, 15.2, 48.2))
    return load_config(config_path)


def test_covers_aoi_true_for_containing_raster(tmp_path: Path) -> None:
    raster = tmp_path / "dem.tif"
    _write_raster(raster, (14.9, 47.9, 15.3, 48.3))
    assert dem.covers_aoi(raster, (15.0, 48.0, 15.2, 48.2)) is True


def test_covers_aoi_false_for_disjoint_raster(tmp_path: Path) -> None:
    raster = tmp_path / "dem.tif"
    _write_raster(raster, (10.0, 50.0, 10.5, 50.5))
    assert dem.covers_aoi(raster, (15.0, 48.0, 15.2, 48.2)) is False


def test_covers_aoi_tolerates_one_pixel(tmp_path: Path) -> None:
    raster = tmp_path / "dem.tif"
    _write_raster(raster, (15.0, 48.0, 15.2, 48.2))  # exactly the AOI bounds
    assert dem.covers_aoi(raster, (15.0, 48.0, 15.2, 48.2)) is True


def test_run_skips_when_cached_dem_covers_aoi(cfg, monkeypatch) -> None:
    cfg.data_dir.mkdir(parents=True)
    _write_raster(cfg.dem_path(), (14.9, 47.9, 15.3, 48.3))

    def fail_if_called(_project: str) -> None:
        raise AssertionError("GEE must not be initialized when the cache is valid")

    monkeypatch.setattr(gee, "init_gee", fail_if_called)
    lines: list[str] = []
    outcome = dem.run(cfg, log=lines.append)
    assert outcome.outputs == [cfg.dem_path()]
    assert any("skipping download" in line for line in lines)


def test_run_redownloads_when_cached_dem_misses_aoi(cfg, monkeypatch) -> None:
    cfg.data_dir.mkdir(parents=True)
    _write_raster(cfg.dem_path(), (10.0, 50.0, 10.5, 50.5))  # elsewhere

    class GeeInitCalled(Exception):
        pass

    def record_call(_project: str) -> None:
        raise GeeInitCalled

    monkeypatch.setattr(gee, "init_gee", record_call)
    lines: list[str] = []
    with pytest.raises(GeeInitCalled):
        dem.run(cfg, log=lines.append)
    assert any("does not cover the AOI" in line for line in lines)
