"""Offline tests for fractional WorldPop exposure calculations."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from flood_pipeline.steps.population import calculate_exposure


def _write_raster(path: Path, values: np.ndarray, pixel_size: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:3857",
        transform=from_origin(0, values.shape[0] * pixel_size, pixel_size, pixel_size),
    ) as dst:
        dst.write(values, 1)


def test_calculate_exposure_uses_fractional_cell_coverage(tmp_path: Path) -> None:
    population_path = tmp_path / "population.tif"
    depth_path = tmp_path / "depth.tif"
    _write_raster(population_path, np.array([[100]], dtype="float32"), 100)
    # Two of four sub-cells are flooded; 999 permanent water is excluded.
    _write_raster(
        depth_path,
        np.array([[10, 10], [0, 999]], dtype="uint16"),
        50,
    )

    stats = calculate_exposure(population_path, depth_path)

    assert stats.total_population == 100
    assert stats.exposed_population == 50
    assert stats.exposed_percent == 50
