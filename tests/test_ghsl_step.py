"""GHSL settlement processing step: fetches built-up data and combines with flood depth rasters."""

from pathlib import Path
from typing import Any, Callable
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from flood_pipeline.steps import flexth_step


def _combine_ghsl_and_depth(
    ghsl_path: Path,
    wd_path: Path,
    out_path: Path,
    log: Callable[[str], None] = print,
) -> None:
    """Combine GHSL raster band with flood water depth raster band into a multi-band GeoTIFF.

    Band 1: GHSL characteristics (uint8)
    Band 2: Water depth (float32)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ghsl_path) as ghsl_src, rasterio.open(wd_path) as wd_src:
        # Read base GHSL data
        ghsl_data = ghsl_src.read(1)
        ghsl_meta = ghsl_src.meta.copy()

        # Reproject water depth to match GHSL grid
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

        # Update metadata for 2-band target
        ghsl_meta.update(
            {
                "count": 2,
                "dtype": rasterio.float32,
            }
        )

        log(f"Writing combined GHSL+Depth raster to {out_path.name}")
        with rasterio.open(out_path, "w", **ghsl_meta) as dst:
            dst.write(ghsl_data, 1)
            dst.write(wd_reprojected, 2)


def run(cfg: Any, log: Callable[[str], None] = print) -> None:
    """Run the GHSL fetching and depth combination pipeline step."""
    if not cfg.ghsl.enabled:
        log("GHSL step is disabled in the config; skipping.")
        return

    log("Starting GHSL processing step...")

    # Path setup
    ghsl_base_path = cfg.ghsl_base_path()

    # Step 1: Fetch GHSL asset from GEE if not already locally cached
    if not ghsl_base_path.exists():
        from flood_pipeline.gee import download_ee_raster

        log(f"Downloading GHSL asset ({cfg.ghsl.asset}) via Earth Engine...")
        download_ee_raster(
            asset_id=cfg.ghsl.asset,
            band=cfg.ghsl.band,
            scale=cfg.ghsl.scale,
            crs=cfg.ghsl.crs,
            aoi_path=cfg.aoi_abs_path,
            output_path=ghsl_base_path,
            project=cfg.project.gee_project,
            log=log,
        )
    else:
        log(f"Using cached GHSL raster: {ghsl_base_path.name}")

    # Step 2: Overlay with water depth outputs from FLEXTH
    scenes = flexth_step.find_scene_outputs(cfg.output_dir)
    if not scenes:
        log("No water depth (WD_) rasters found to combine with GHSL.")
        return

    for stamp, paths in scenes.items():
        wd_path = paths.get("WD")
        if wd_path and wd_path.exists():
            out_path = cfg.scene_ghsl_depth_path(stamp)
            _combine_ghsl_and_depth(ghsl_base_path, wd_path, out_path, log=log)

    log("GHSL step completed successfully.")