"""GHSL settlement processing step: fetches built-up data and combines with flood depth rasters."""

from pathlib import Path
from typing import Callable

import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from flood_pipeline import gee
from flood_pipeline.config import PipelineConfig
from flood_pipeline.steps.depth_damage_prov import (
    DamageTable,
    damage_fraction_for_pixels,
    load_damage_table,
    resolve_continent,
)
from flood_pipeline.steps import LogFn, StepOutcome, flexth_step


def _combine_ghsl_and_depth(
    ghsl_path: Path,
    wd_path: Path,
    out_path: Path,
    *,
    jrc_column: str,
    damage_table: DamageTable,
    log: LogFn = print,
) -> None:
    """Combine GHSL, flood depth, and relative flood damage into one GeoTIFF.

    Band 1: GHSL characteristics masked to FLEXTH depth intersection (float32)
    Band 2: Water depth, same units as FLEXTH's WD_*.tif (float32)
    Band 3: Relative flood damage fraction, 0-1, from the JRC/Huizinga
            depth-damage functions for ``jrc_column``'s continent (float32).
            Depth is converted from WD's native centimeters to meters before
            the table lookup, since the JRC table is indexed in meters.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ghsl_path) as ghsl_src, rasterio.open(wd_path) as wd_src:
        ghsl_data = ghsl_src.read(1)
        ghsl_meta = ghsl_src.meta.copy()

        wd_reprojected = np.zeros(
            (ghsl_src.height, ghsl_src.width), dtype=np.float32
        )

        reproject(
            source=rasterio.band(wd_src, 1),
            destination=wd_reprojected,
            src_transform=wd_src.transform,
            src_crs=wd_src.crs,
            dst_transform=ghsl_src.transform,
            dst_crs=ghsl_src.crs,
            resampling=Resampling.nearest,
        )

        # Create a boolean mask of valid water depth pixels
        # FLEXTH depth convention: 0 is nodata, 999 is permanent water
        valid_depth_mask = (wd_reprojected > 0) & (wd_reprojected < 999)

        # Mask Band 1 so GHSL only keeps pixels intersecting FLEXTH water data
        ghsl_data_masked = np.where(valid_depth_mask, ghsl_data, 0)

        # Zero out invalid depth pixels in Band 2 for clean storage
        wd_reprojected_masked = np.where(valid_depth_mask, wd_reprojected, 0.0)

        # Band 3: relative damage fraction. WD_*.tif is uint16 centimeters
        # (see flexth_step docs), so convert to meters for the JRC lookup,
        # which is indexed in meters.
        depth_m = wd_reprojected_masked / 100.0
        damage_fraction = damage_fraction_for_pixels(
            ghsl_data_masked.astype("int32"), depth_m, jrc_column, damage_table
        )
        damage_fraction = np.where(valid_depth_mask, damage_fraction, 0.0)

        ghsl_meta.update(
            {
                "count": 3,
                "dtype": rasterio.float32,
                "nodata": 0,
            }
        )

        log(f"Writing combined GHSL+Depth+Damage raster to {out_path.name}")
        with rasterio.open(out_path, "w", **ghsl_meta) as dst:
            dst.write(ghsl_data_masked.astype(np.float32), 1)
            dst.write(wd_reprojected_masked, 2)
            dst.write(damage_fraction.astype(np.float32), 3)


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Run the GHSL fetching and depth/damage combination pipeline step."""
    if not cfg.ghsl.enabled:
        log("GHSL step is disabled in the config; skipping.")
        return StepOutcome()

    log("Starting GHSL processing step...")

    # Path setup
    ghsl_path = cfg.ghsl_path()
    output_files = []

    # Step 1: Fetch GHSL asset from GEE if not already locally cached
    if not ghsl_path.exists():
        log(f"Initializing GEE project: {cfg.project.gee_project}")
        gee.init_gee(cfg.project.gee_project)

        log(f"Downloading GHSL asset ({cfg.ghsl.asset}) via Earth Engine...")
        aoi_gdf = gpd.read_file(cfg.aoi_abs_path).to_crs(epsg=4326)
        aoi_ee = geemap.geopandas_to_ee(aoi_gdf)

        # GHSL is a static single image asset (or image collection filtered/selected)
        image = ee.Image(cfg.ghsl.asset).select(cfg.ghsl.band).clip(aoi_ee)

        ghsl_path.parent.mkdir(parents=True, exist_ok=True)
        geemap.download_ee_image(
            image,
            str(ghsl_path),
            region=aoi_ee.geometry(),
            scale=cfg.ghsl.scale,
            crs=cfg.ghsl.crs,
        )
    else:
        log(f"Using cached GHSL raster: {ghsl_path.name}")

    # Step 2: Resolve the AOI's continent once, for the depth-damage lookup.
    # The JRC/Huizinga functions are broken out by continent; GHSL's class
    # codes alone don't carry location, so this is done once per run rather
    # than per scene.
    country_match = resolve_continent(cfg.aoi_abs_path)
    log(
        f"AOI resolved to {country_match.country_name or 'unknown country'} "
        f"-> JRC damage-function column '{country_match.jrc_column}'"
    )
    damage_table = load_damage_table()

    # Step 3: Overlay with water depth outputs from FLEXTH
    scenes = flexth_step.find_scene_outputs(cfg.output_dir)
    if not scenes:
        log("No water depth (WD_) rasters found to combine with GHSL.")
        return StepOutcome(outputs=[ghsl_path])

    for stamp, paths in scenes.items():
        wd_path = paths.get("WD")
        if wd_path and wd_path.exists():
            out_path = cfg.scene_ghsl_depth_path(stamp)
            _combine_ghsl_and_depth(
                ghsl_path,
                wd_path,
                out_path,
                jrc_column=country_match.jrc_column,
                damage_table=damage_table,
                log=log,
            )
            output_files.append(out_path)

    log("GHSL step completed successfully.")
    return StepOutcome(outputs=[ghsl_path] + output_files)