"""Download WorldPop and combine it with FLEXTH depth for exposure metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from flood_pipeline import gee
from flood_pipeline.config import PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome
from flood_pipeline.steps.dem import covers_aoi, load_aoi


def build_population_image(
    collection_id: str, year: int, band: str, aoi_ee
) -> ee.Image:
    """Mosaic country images for one year and clip them to the project AOI."""
    collection = (
        ee.ImageCollection(collection_id)
        .filter(ee.Filter.eq("year", year))
        .filterBounds(aoi_ee)
        .select(band)
    )
    return collection.mosaic().rename("population").clip(aoi_ee)


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Download the configured WorldPop population-count raster."""
    out_path = cfg.population_path()
    aoi = load_aoi(cfg.aoi_abs_path)
    bounds = tuple(round(float(value), 5) for value in aoi.total_bounds)

    if out_path.exists() and not cfg.population.overwrite:
        if covers_aoi(out_path, bounds):
            log(f"WorldPop raster already covers the AOI, skipping: {out_path}")
            return StepOutcome(outputs=[out_path])
        log(f"existing {out_path.name} does not cover the AOI; downloading it again")

    gee.init_gee(cfg.project.gee_project)
    aoi_ee = geemap.geopandas_to_ee(aoi)
    image = build_population_image(
        cfg.population.collection,
        cfg.population.year,
        cfg.population.band,
        aoi_ee,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(
        f"downloading WorldPop {cfg.population.year} at "
        f"{cfg.population.scale} m to {out_path} ..."
    )
    geemap.download_ee_image(
        image,
        str(out_path),
        region=aoi_ee.geometry(),
        scale=cfg.population.scale,
        crs=cfg.population.crs,
    )
    log(f"WorldPop written: {out_path}")
    return StepOutcome(outputs=[out_path])


@dataclass(frozen=True)
class ExposureStats:
    """Population totals for the downloaded AOI and modeled flood footprint."""

    total_population: float
    exposed_population: float

    @property
    def exposed_percent(self) -> float:
        if self.total_population <= 0:
            return 0.0
        return 100.0 * self.exposed_population / self.total_population


def calculate_exposure(population_path: Path, depth_path: Path) -> ExposureStats:
    """Estimate exposure using fractional flood coverage per population cell.

    FLEXTH depth is converted to a binary flood mask (0/nodata and permanent
    water sentinel 999 are excluded). ``average`` reprojection calculates the
    fraction of each coarser WorldPop cell covered by modeled flooding.
    """
    with rasterio.open(population_path) as pop_src:
        population = pop_src.read(1, masked=True).filled(0).astype("float64")
        population = np.where(np.isfinite(population) & (population > 0), population, 0)
        flooded_fraction = np.zeros(population.shape, dtype="float32")

        with rasterio.open(depth_path) as depth_src:
            depth = depth_src.read(1, masked=True).filled(0)
            flooded = ((depth > 0) & (depth != 999)).astype("float32")
            reproject(
                source=flooded,
                destination=flooded_fraction,
                src_transform=depth_src.transform,
                src_crs=depth_src.crs,
                dst_transform=pop_src.transform,
                dst_crs=pop_src.crs,
                src_nodata=None,
                dst_nodata=0,
                resampling=Resampling.average,
            )

    return ExposureStats(
        total_population=float(population.sum()),
        exposed_population=float((population * flooded_fraction).sum()),
    )
