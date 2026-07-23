"""Connected flood areas from a flood raster, as polygons.

The synthetic rasters are in UTM 33N with 30 m pixels, so one pixel is exactly
900 m2 = 0.09 ha and every expected area is a round multiple of that.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from flood_pipeline import polygonize
from flood_pipeline.config import FLOOD_AREA_LAYER, vector_path

PIXEL_HA = 0.09  # 30 m x 30 m


def _write_raster(path: Path, mask: np.ndarray, *, pixel: float = 30.0) -> Path:
    """Write ``mask`` as a uint8 GeoTIFF in EPSG:32633 with square pixels."""
    height, width = mask.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500_000.0, 5_000_000.0, pixel, pixel),
    ) as dst:
        dst.write(mask.astype("uint8"), 1)
    return path


@pytest.fixture
def raster(tmp_path: Path) -> Path:
    """Three separated flood areas: 16 px (1.44 ha), 4 px (0.36 ha), 1 px speck."""
    mask = np.zeros((12, 12), dtype="uint8")
    mask[1:5, 1:5] = 1  # big blob
    mask[7:9, 1:3] = 1  # small blob
    mask[10, 10] = 1  # single-pixel speck
    return _write_raster(tmp_path / "gfm_flood_max.tif", mask)


def test_finds_every_connected_area(raster: Path) -> None:
    areas = polygonize.flood_polygons(raster, min_area_ha=0.0, scene="max")
    assert len(areas) == 3


def test_area_ids_rank_the_areas_by_size(raster: Path) -> None:
    areas = polygonize.flood_polygons(raster, min_area_ha=0.0, scene="max")
    assert list(areas["area_id"]) == [1, 2, 3]
    assert list(areas["area_ha"]) == sorted(areas["area_ha"], reverse=True)
    assert areas["area_ha"].iloc[0] == pytest.approx(16 * PIXEL_HA, rel=0.01)
    assert areas["area_ha"].iloc[1] == pytest.approx(4 * PIXEL_HA, rel=0.01)
    assert areas["area_ha"].iloc[2] == pytest.approx(PIXEL_HA, rel=0.01)


def test_min_area_drops_the_smaller_areas(raster: Path) -> None:
    areas = polygonize.flood_polygons(raster, min_area_ha=0.5, scene="max")
    assert len(areas) == 1
    assert list(areas["area_id"]) == [1]  # ids are re-assigned after filtering


def test_result_is_wgs84_and_carries_the_scene(raster: Path) -> None:
    areas = polygonize.flood_polygons(raster, min_area_ha=0.0, scene="2024-09-16_051230")
    assert areas.crs.to_epsg() == 4326
    assert set(areas["scene"]) == {"2024-09-16_051230"}


def test_diagonally_touching_pixels_are_one_area(tmp_path: Path) -> None:
    """Connectivity 8, matching flexth's flood_processing.params.connectivity."""
    mask = np.zeros((5, 5), dtype="uint8")
    mask[1, 1] = 1
    mask[2, 2] = 1  # touches the first pixel only at a corner
    raster = _write_raster(tmp_path / "diagonal.tif", mask)

    areas = polygonize.flood_polygons(raster, min_area_ha=0.0, scene="max")

    assert len(areas) == 1
    assert areas["area_ha"].iloc[0] == pytest.approx(2 * PIXEL_HA, rel=0.01)


def test_nodata_is_not_flooded(tmp_path: Path) -> None:
    """GFM writes 255 for nodata; only positive flood values count."""
    mask = np.full((5, 5), 255, dtype="uint8")
    mask[1:3, 1:3] = 1
    raster = _write_raster(tmp_path / "with_nodata.tif", mask)
    with rasterio.open(raster, "r+") as dst:
        dst.nodata = 255

    areas = polygonize.flood_polygons(raster, min_area_ha=0.0, scene="max")

    assert len(areas) == 1
    assert areas["area_ha"].iloc[0] == pytest.approx(4 * PIXEL_HA, rel=0.01)


def test_write_creates_a_readable_layer_next_to_the_raster(raster: Path) -> None:
    written = polygonize.write_flood_polygons(
        raster, min_area_ha=0.0, scene="max", log=lambda _line: None
    )

    assert written == vector_path(raster)
    reloaded = gpd.read_file(written, layer=FLOOD_AREA_LAYER)
    assert list(reloaded["area_id"]) == [1, 2, 3]
    assert reloaded.crs.to_epsg() == 4326


def test_write_skips_the_file_when_nothing_survives_the_filter(raster: Path) -> None:
    lines: list[str] = []

    written = polygonize.write_flood_polygons(
        raster, min_area_ha=1000.0, scene="max", log=lines.append
    )

    assert written is None
    assert not vector_path(raster).exists()
    assert any("no flood area" in line for line in lines)
