"""FABDEM DEM step: mosaic from Google Earth Engine, local download or Drive export.

"""

from __future__ import annotations

from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp

from flood_pipeline import gee
from flood_pipeline.config import PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

FABDEM_COLLECTION = "projects/sat-io/open-datasets/FABDEM"


def load_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    """Read the AOI vector file and bring it to EPSG:4326."""
    return gpd.read_file(aoi_path).to_crs(epsg=4326)


def build_fabdem_image(aoi: gpd.GeoDataFrame) -> tuple[ee.Image, ee.Geometry]:
    """FABDEM mosaic clipped to the AOI, plus the AOI as an EE geometry.
    """
    aoi_ee = geemap.geopandas_to_ee(aoi)
    mosaic = ee.ImageCollection(FABDEM_COLLECTION).filterBounds(aoi_ee).mosaic()
    return mosaic.clip(aoi_ee), aoi_ee.geometry()


def covers_aoi(dem_path: Path, aoi_bounds_4326: tuple[float, float, float, float]) -> bool:
    """Whether an existing DEM raster's extent contains the AOI bounds.
    """
    with rasterio.open(dem_path) as source:
        left, bottom, right, top = rasterio.warp.transform_bounds(
            "EPSG:4326", source.crs, *aoi_bounds_4326
        )
        tolerance_x = abs(source.transform.a)
        tolerance_y = abs(source.transform.e)
        return (
            source.bounds.left <= left + tolerance_x
            and source.bounds.bottom <= bottom + tolerance_y
            and source.bounds.right >= right - tolerance_x
            and source.bounds.top >= top - tolerance_y
        )


def normalize_nodata(dem_path: Path, log: LogFn = print) -> bool:
    """Rewrite an infinite nodata value to NaN in place. True if it changed.
    """
    with rasterio.open(dem_path) as source:
        if source.nodata is None or not np.isinf(source.nodata):
            return False
        if not np.issubdtype(np.dtype(source.dtypes[0]), np.floating):
            return False  # an integer band cannot hold NaN
        profile = source.profile
        values = source.read()
    values = np.where(np.isinf(values), np.nan, values).astype(profile["dtype"])
    profile.update(nodata=np.nan)
    with rasterio.open(dem_path, "w", **profile) as destination:
        destination.write(values)
    log(f"rewrote infinite nodata to NaN in {dem_path.name}")
    return True


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Fetch the FABDEM DEM for the AOI according to ``cfg.dem``."""
    out_path = cfg.dem_path()
    aoi = load_aoi(cfg.aoi_abs_path)
    bounds = tuple(round(float(value), 5) for value in aoi.total_bounds)
    log(f"AOI bounds (EPSG:4326): {bounds}")

    if out_path.exists() and not cfg.dem.overwrite:
        if covers_aoi(out_path, bounds):
            log(f"DEM already present and covers the AOI, skipping download: {out_path}")
            # Also repairs a DEM downloaded before nodata normalization existed
            # (including one delivered by hand from a Drive export).
            normalize_nodata(out_path, log)
            return StepOutcome(outputs=[out_path])
        log(f"existing {out_path.name} does not cover the AOI — downloading a new DEM")

    gee.init_gee(cfg.project.gee_project)
    image, region = build_fabdem_image(aoi)

    if cfg.dem.delivery == "drive":
        return _export_to_drive(cfg, image, region, log)
    return _download_local(cfg, image, region, out_path, log)


def _download_local(
    cfg: PipelineConfig,
    image: ee.Image,
    region: ee.Geometry,
    out_path: Path,
    log: LogFn,
) -> StepOutcome:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading FABDEM at {cfg.dem.scale} m resolution to {out_path} ...")
    # geedim-backed: splits the request into tiles below the EE size limit
    # and reassembles them into one GeoTIFF.
    geemap.download_ee_image(
        image,
        str(out_path),
        region=region,
        scale=cfg.dem.scale,
        crs=cfg.dem.crs,
    )
    normalize_nodata(out_path, log)
    log(f"DEM written: {out_path}")
    return StepOutcome(outputs=[out_path])


def _export_to_drive(
    cfg: PipelineConfig,
    image: ee.Image,
    region: ee.Geometry,
    log: LogFn,
) -> StepOutcome:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"{cfg.dem.drive_prefix}_export",
        folder=cfg.dem.drive_folder,
        fileNamePrefix=cfg.dem.drive_prefix,
        scale=cfg.dem.scale,
        region=region,
        fileFormat="GeoTIFF",
    )
    task.start()
    message = (
        f"Drive export task started (id={task.id}). When it finishes, download "
        f"'{cfg.dem.drive_prefix}*.tif' from the Drive folder "
        f"'{cfg.dem.drive_folder}' to {cfg.dem_path()}, then re-run the "
        "pipeline: the dem step will detect the file and continue with gfm/flexth."
    )
    log(message)
    return StepOutcome(halt=True, message=message)
