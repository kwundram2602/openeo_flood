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
    GHSL_TO_DAMAGE_CLASS,
    MaxDamage,
    damage_fraction_for_pixels,
    load_damage_table,
    load_max_damage_table,
    resolve_continent,
)
from flood_pipeline.steps import LogFn, StepOutcome, flexth_step

# Which GHSL class ids count as "residential" vs "commercial" for EUR-damage
# purposes (Band 4 below), derived from the single source of truth so this
# can't drift out of sync with the relative-damage class mapping.
_RESIDENTIAL_GHSL_CLASSES = frozenset(
    class_id
    for class_id, damage_class in GHSL_TO_DAMAGE_CLASS.items()
    if damage_class == "Residential buildings"
)
_COMMERCIAL_GHSL_CLASSES = frozenset(
    class_id
    for class_id, damage_class in GHSL_TO_DAMAGE_CLASS.items()
    if damage_class == "Commercial buildings"
)

_METERS_PER_DEGREE_LAT = 111_320.0  # local equirectangular approximation


def _pixel_area_m2(src: rasterio.DatasetReader) -> np.ndarray | float:
    """Per-pixel ground area in m^2 for ``src``'s grid.

    Returns a (height, 1) array broadcastable against a (height, width) array
    when ``src``'s CRS is geographic (pixel width in meters shrinks toward
    the poles as longitude degrees converge with latitude), or a plain float
    when it's already a projected/metric CRS (cfg.ghsl.crs defaults to
    EPSG:4326, but is user-configurable).

    The geographic case uses a local equirectangular approximation -- accurate
    at the scale of a single GHSL pixel (10s of meters), not for continental
    distances, which is exactly the scale this needs it at.
    """
    transform = src.transform
    px_width = abs(transform.a)
    px_height = abs(transform.e)
    if src.crs is not None and src.crs.is_geographic:
        rows = np.arange(src.height)
        row_center_lat = transform.f + transform.e * (rows + 0.5)
        meters_per_degree_lon = _METERS_PER_DEGREE_LAT * np.cos(np.radians(row_center_lat))
        row_area = (px_height * _METERS_PER_DEGREE_LAT) * (px_width * meters_per_degree_lon)
        return row_area.astype("float64")[:, np.newaxis]
    return float(px_width * px_height)


def _combine_ghsl_and_depth(
    ghsl_path: Path,
    wd_path: Path,
    out_path: Path,
    *,
    jrc_column: str,
    damage_table: DamageTable,
    max_damage: MaxDamage,
    log: LogFn = print,
) -> None:
    """Combine GHSL, flood depth, relative damage and EUR damage into one GeoTIFF.

    Band 1: GHSL characteristics where water depth and buildings overlap (float32)
    Band 2: Water depth, same units as FLEXTH's WD_*.tif (float32)
    Band 3: Relative flood damage fraction, 0-1, from the JRC/Huizinga
            depth-damage functions for ``jrc_column``'s continent (float32).
            Depth is converted from WD's native centimeters to meters before
            the table lookup, since the JRC table is indexed in meters.
    Band 4: Absolute EUR damage estimate (float32), residential/commercial
            building pixels only (0 elsewhere, including non-building GHSL
            classes such as roads/agriculture): relative damage fraction x
            JRC MaxDamage EUR/m^2 for ``max_damage``'s matched country x
            pixel area in m^2. MaxDamage values are 2010 prices, not
            inflation-adjusted -- treat as order-of-magnitude, not a
            calibrated present-day valuation. Agriculture and infrastructure
            are out of scope (see flood_pipeline.steps.max_damage).

    All three of the masked bands share the same valid-pixel footprint (a
    building present in GHSL AND FLEXTH water depth present) so a pixel that
    is zero in one is zero in all three; Band 4 is additionally zero for any
    non-building GHSL class within that footprint.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ghsl_path) as ghsl_src, rasterio.open(wd_path) as wd_src:
        ghsl_data = ghsl_src.read(1)
        ghsl_meta = ghsl_src.meta.copy()
        pixel_area_m2 = _pixel_area_m2(ghsl_src)

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

        # Define valid water footprint (where FLEXTH actually produced data)
        # FLEXTH depth convention: 0 is nodata, 999 is permanent water
        valid_depth_mask = (wd_reprojected > 0) & (wd_reprojected < 999)

        # Define valid buildings
        valid_ghsl_mask = ghsl_data > 0

        # Keep pixels where BOTH a building exists AND water depth data is
        # present. All three output bands share this footprint so they stay
        # mutually consistent (a pixel zero in one band is zero in all three).
        combined_valid_mask = valid_ghsl_mask & valid_depth_mask

        ghsl_data_masked = np.where(combined_valid_mask, ghsl_data, 0)
        wd_reprojected_masked = np.where(combined_valid_mask, wd_reprojected, 0.0)

        # Band 3: relative damage fraction. WD_*.tif is uint16 centimeters
        # (see flexth_step docs), so convert to meters for the JRC lookup,
        # which is indexed in meters.
        depth_m = wd_reprojected_masked / 100.0
        damage_fraction = damage_fraction_for_pixels(
            ghsl_data_masked.astype("int32"), depth_m, jrc_column, damage_table
        )
        damage_fraction = np.where(combined_valid_mask, damage_fraction, 0.0)

        # Band 4: EUR/m^2 varies by building type; 0 for any non-building
        # GHSL class (open space/roads/agriculture), even if it's within the
        # combined_valid_mask footprint used for Bands 1-3.
        eur_per_m2 = np.zeros(ghsl_data_masked.shape, dtype=np.float32)
        eur_per_m2[np.isin(ghsl_data_masked, list(_RESIDENTIAL_GHSL_CLASSES))] = (
            max_damage.residential_eur_per_m2
        )
        eur_per_m2[np.isin(ghsl_data_masked, list(_COMMERCIAL_GHSL_CLASSES))] = (
            max_damage.commercial_eur_per_m2
        )
        euro_damage = (damage_fraction * eur_per_m2 * pixel_area_m2).astype(np.float32)

        ghsl_meta.update(
            {
                "count": 4,
                "dtype": rasterio.float32,
                "nodata": 0,
            }
        )

        log(f"Writing combined GHSL+Damage+EUR raster to {out_path.name}")
        with rasterio.open(out_path, "w", **ghsl_meta) as dst:
            dst.write(ghsl_data_masked.astype(np.float32), 1)
            dst.write(wd_reprojected_masked.astype(np.float32), 2)
            dst.write(damage_fraction.astype(np.float32), 3)
            dst.write(euro_damage, 4)


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Run the GHSL fetching and depth/damage combination pipeline step.

    GHSL itself is fetched once per project (it doesn't vary by GFM
    algorithm), then combined against every band's FLEXTH output -- one
    combined GHSL+Depth+Damage raster per (band, scene) pair, matching the
    per-band directory layout the rest of the pipeline now uses.
    """
    if not cfg.ghsl.enabled:
        log("GHSL step is disabled in the config; skipping.")
        return StepOutcome()

    log("Starting GHSL processing step...")

    # GEE is needed both for the GHSL download below and for resolve_continent()'s
    # Earth Engine country lookup, so initialize it unconditionally -- a cached
    # GHSL raster used to skip this entirely, which broke resolve_continent().
    log(f"Initializing GEE project: {cfg.project.gee_project}")
    gee.init_gee(cfg.project.gee_project)

    # Step 1: Fetch GHSL asset from GEE if not already locally cached
    ghsl_path = cfg.ghsl_path()

    if not ghsl_path.exists():
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

    # Step 2: Resolve the AOI's continent once, for the depth-damage lookup,
    # and its country once, for the EUR MaxDamage lookup. The JRC functions
    # are broken out by continent and MaxDamage by country; GHSL's class
    # codes alone don't carry location, so both are done once per run rather
    # than once per band/scene.
    country_match = resolve_continent(cfg.aoi_abs_path)
    log(
        f"AOI resolved to {country_match.country_name or 'unknown country'} "
        f"-> JRC damage-function column '{country_match.jrc_column}'"
    )
    damage_table = load_damage_table()

    max_damage_table = load_max_damage_table()
    max_damage = max_damage_table.lookup(country_match.country_name)
    if max_damage.matched:
        log(
            f"MaxDamage matched to '{max_damage.country_name}': "
            f"residential {max_damage.residential_eur_per_m2:.0f} EUR/m2, "
            f"commercial {max_damage.commercial_eur_per_m2:.0f} EUR/m2 (2010 prices)"
        )
    else:
        log(
            f"WARNING: no MaxDamage entry for '{country_match.country_name}'; "
            f"using the World fallback: residential "
            f"{max_damage.residential_eur_per_m2:.0f} EUR/m2, commercial "
            f"{max_damage.commercial_eur_per_m2:.0f} EUR/m2 (2010 prices)"
        )

    # Step 3: Overlay with water depth outputs from FLEXTH, once per resolved
    # algorithm/likelihood band configured in the pipeline.
    output_files: list[Path] = [ghsl_path]
    resolved_keys = [key for key, _ in cfg.resolved_bands()]

    if not resolved_keys:
        log("No active GFM bands resolved in config; nothing to combine.")
        return StepOutcome(outputs=output_files)

    for band in resolved_keys:
        scenes = flexth_step.find_scene_outputs(cfg.scene_output_root(band))
        if not scenes:
            log(f"##[band:{band}] no water depth (WD_) rasters found; skipping.")
            continue
        for stamp, paths in scenes.items():
            wd_path = paths.get("WD")
            if wd_path and wd_path.exists():
                out_path = cfg.scene_ghsl_depth_path(band, stamp)
                _combine_ghsl_and_depth(
                    ghsl_path,
                    wd_path,
                    out_path,
                    jrc_column=country_match.jrc_column,
                    damage_table=damage_table,
                    max_damage=max_damage,
                    log=log,
                )
                output_files.append(out_path)
                log(f"##[band:{band}][scene:{stamp}] output: {out_path}")

    log("GHSL step completed successfully.")
    return StepOutcome(outputs=output_files)