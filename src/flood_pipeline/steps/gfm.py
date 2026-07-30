"""GFM flood-extent step: ensemble flood extent from the EODC STAC catalog.

One raster is written per acquisition timestamp (cube time-slice); FLEXTH then
consumes each per-scene mask. The whole-time maximum is always written too (as a
reference/overlay) and the temporal sum is optional (``gfm.aggregation: sum`` or
``both``).
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.stac
import odc.stac # used to create a (time, y, x) cube from the GFM STAC items
import pandas as pd
import pystac # just using for return type hint; no STAC operations here
import pystac_client # open stac catalog and search for items
import rioxarray
import xarray as xr

from flood_pipeline import polygonize
from flood_pipeline.config import (
    SCENE_STAMP_FORMAT,
    PipelineConfig,
    is_likelihood_band,
)
from flood_pipeline.steps import LogFn, StepOutcome

GFM_NODATA = 255
# Static per-AOI permanent-water reference band; downloaded once (max over time).
GFM_REFERENCE_WATER_BAND = "reference_water_mask"
# Per-acquisition mask of pixels GFM could not evaluate (urban layover/shadow,
# dense vegetation). Depends on the pass geometry, so it is written per scene —
# unlike the reference water, a whole-time max would blur pass-specific gaps.
GFM_EXCLUSION_BAND = "exclusion_mask"


def aoi_bbox_4326(aoi_path: Path) -> tuple[float, float, float, float]:
    """Bounding box (west, south, east, north) of the AOI in EPSG:4326."""
    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    west, south, east, north = (float(value) for value in aoi.total_bounds)
    return (west, south, east, north)


def search_gfm_items(
    bbox: tuple[float, float, float, float],
    temporal_extent: list[str],
    *,
    stac_url: str,
    collection: str,
) -> pystac.ItemCollection:
    """Search the STAC catalog for *all* GFM items covering bbox and time range.

    No ``max_items`` cap here: the EODC API returns items newest-first, so a cap
    would silently drop the *oldest* acquisitions in the window (see
    :func:`_cap_items` for an opt-in, warned cap). Standalone so the dashboard
    scene browser can reuse it (public API, no authentication).
    """
    catalog = pystac_client.Client.open(stac_url)
    search = catalog.search(
        bbox=list(bbox),
        datetime=list(temporal_extent),
        collections=[collection],
    )
    return search.item_collection()


def load_flood_cube(
    items: pystac.ItemCollection,
    bbox: tuple[float, float, float, float],
    *,
    band: str,
    resolution: float,
) -> xr.DataArray:
    """Load the flood-extent band as a (time, y, x) cube with nodata as NaN.

    odc-stac reprojects correctly from GFM's native Equi7Grid projection.
    """
    cube = odc.stac.load(
        items,
        bands=[band],
        crs="EPSG:4326",
        resolution=resolution,
        bbox=bbox,
        resampling="nearest",
    )
    flood = cube[band].astype("float32")
    return flood.where(flood != GFM_NODATA)


def _cap_items(items, max_items: int, log: LogFn) -> list:
    """Return items, optionally capped to the newest ``max_items`` (0 = no cap).

    The cap is a safety valve, not the default: when it truncates, it warns and
    keeps the newest acquisitions (the API's own order), dropping older ones.
    """
    items = list(items)
    if not max_items or len(items) <= max_items:
        return items
    ordered = sorted(
        items,
        key=lambda it: it.datetime or dt.datetime.min.replace(tzinfo=dt.timezone.utc),
        reverse=True,
    )
    log(
        f"WARNING: {len(items)} GFM items available but gfm.max_items={max_items}; "
        f"keeping the newest {max_items}, dropping {len(items) - max_items} older "
        "scene(s) — set gfm.max_items to 0 to keep all"
    )
    return ordered[:max_items]


def _binarize_likelihood(cube: xr.DataArray, threshold: int) -> xr.DataArray:
    """Threshold a 0-100 likelihood cube into a 0/1 flood extent, keeping NaN."""
    return (cube >= threshold).astype("float32").where(cube.notnull())


def _has_flood(scene: xr.DataArray) -> bool:
    """True if the scene has any flood pixel (value > 0) over the AOI.

    Empty scenes (all nodata/NaN or all-zero) come from overpass frames that do
    not cover the AOI or saw no flood; FLEXTH would produce an empty depth map.
    """
    return bool((scene > 0).any())


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Fetch GFM flood extent + water depth inputs for each resolved band.

    Writes one folder per flooded acquisition under flood_data/<band>/<stamp>/
    (flood mask + polygons + exclusion mask) plus per-band whole-time
    aggregates. Single-band and compare-all-algorithms share this one loop.
    """
    bbox = aoi_bbox_4326(cfg.aoi_abs_path)
    log(
        f"searching {cfg.gfm.collection} at {cfg.gfm.stac_url} for bbox "
        f"{tuple(round(value, 5) for value in bbox)}, dates {cfg.gfm.temporal_extent}"
    )
    items = search_gfm_items(
        bbox,
        cfg.gfm.temporal_extent,
        stac_url=cfg.gfm.stac_url,
        collection=cfg.gfm.collection,
    )
    if len(items) == 0:
        raise RuntimeError(
            f"no {cfg.gfm.collection} items found for bbox {bbox} in "
            f"{cfg.gfm.temporal_extent}; widen gfm.temporal_extent or check the AOI"
        )
    items = _cap_items(items, cfg.gfm.max_items, log)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_flat_layout(cfg, log)

    outputs: list[Path] = []
    flood_pixels = 0
    for key, band_name in cfg.resolved_bands():
        log(f"== band {key} ({band_name}): loading {len(items)} items ...")
        band_outputs, band_pixels = _run_band(cfg, items, bbox, key, band_name, log)
        outputs.extend(band_outputs)
        flood_pixels += band_pixels
    if flood_pixels == 0:
        # Only when no band detected anything: with compare_algorithms a single
        # empty algorithm is a real result, not a broken run.
        raise RuntimeError(
            "GFM returned scenes, but the flood extent is zero throughout the AOI "
            "and date range. FLEXTH cannot estimate water depth without flood "
            "pixels. Check the GFM raster/scene browser, use a known flooded AOI "
            "or another date range; increasing max_items alone does not create "
            "detections."
        )
    return StepOutcome(outputs=outputs)


def _run_band(
    cfg: PipelineConfig,
    items: pystac.ItemCollection,
    bbox: tuple[float, float, float, float],
    key: str,
    band_name: str,
    log: LogFn,
) -> tuple[list[Path], int]:
    """Write one band's per-scene rasters and whole-time aggregates.

    Returns the written paths and the band's flood pixel count in the temporal
    maximum, which the caller uses to detect a run without any detections.
    """
    flood = load_flood_cube(items, bbox, band=band_name, resolution=cfg.gfm.resolution)
    # Keep the raw likelihood so its scenes can be shown/probed; the extent used
    # downstream is the thresholded version.
    likelihood = flood if is_likelihood_band(band_name) else None
    if likelihood is not None:
        flood = _binarize_likelihood(flood, cfg.gfm.likelihood_threshold)
    # Same items/grid/time axis as the flood cube, so each flood scene has a
    # matching exclusion slice under the same acquisition timestamp.
    exclusion = load_flood_cube(
        items, bbox, band=GFM_EXCLUSION_BAND, resolution=cfg.gfm.resolution
    )
    cfg.gfm_band_dir(key).mkdir(parents=True, exist_ok=True)
    _clear_band_scene_dirs(cfg, key)

    outputs: list[Path] = []
    scene_count = 0
    skipped = 0
    for time_value in flood["time"].values:
        stamp = pd.Timestamp(time_value).strftime(SCENE_STAMP_FORMAT)
        scene = flood.sel(time=time_value, drop=True)
        if not _has_flood(scene):
            log(f"skipped {key}/{stamp}: no flood pixels over the AOI")
            skipped += 1
            continue
        scene_count += 1
        cfg.gfm_scene_dir(key, stamp).mkdir(parents=True, exist_ok=True)
        outputs.extend(
            _write_scene(cfg, scene, cfg.gfm_scene_path(key, stamp), stamp, log)
        )
        # Overlay only (no flood polygons): the exclusion mask of this same pass.
        excl_scene = exclusion.sel(time=time_value, drop=True)
        outputs.append(
            _write_geotiff(excl_scene, cfg.gfm_exclusion_path(key, stamp), log)
        )
        if likelihood is not None:
            outputs.append(
                _write_geotiff(
                    likelihood.sel(time=time_value, drop=True),
                    cfg.gfm_likelihood_path(key, stamp),
                    log,
                )
            )

    log(f"band {key}: wrote {scene_count} scene(s), skipped {skipped} empty")
    flood_max = flood.max(dim="time")
    flood_pixel_count = int(np.count_nonzero(np.nan_to_num(flood_max.values) > 0))
    log(f"band {key}: flood pixels in temporal maximum: {flood_pixel_count}")
    outputs.extend(_write_scene(cfg, flood_max, cfg.gfm_mask_path(key), "max", log))
    if cfg.gfm.aggregation in ("sum", "both"):
        # No polygons here: sum > 0 covers exactly the same pixels as max > 0.
        outputs.append(
            _write_geotiff(flood.sum(dim="time"), cfg.gfm_sum_path(key), log)
        )
    # Static per AOI, so the whole-time max over the same items covers the full
    # AOI in a single raster (reference/overlay layer, not a FLEXTH input).
    reference = load_flood_cube(
        items, bbox, band=GFM_REFERENCE_WATER_BAND, resolution=cfg.gfm.resolution
    )
    outputs.append(
        _write_geotiff(
            reference.max(dim="time"), cfg.gfm_reference_water_path(key), log
        )
    )
    return outputs, flood_pixel_count


def _clear_band_scene_dirs(cfg: PipelineConfig, band: str) -> None:
    """Drop a band's per-scene folders from a previous run before rewriting.

    Skipped/empty scenes must not linger and re-feed FLEXTH; the exclusion mask
    and flood polygons live inside the folder, so they go with it.
    """
    for stamp in cfg.gfm_scene_stamps(band):
        shutil.rmtree(cfg.gfm_scene_dir(band, stamp))


def _remove_legacy_flat_layout(cfg: PipelineConfig, log: LogFn) -> None:
    """Remove pre-band flat-layout GFM files left directly in data_dir."""
    if not cfg.data_dir.exists():
        return
    for pattern in ("gfm_flood_*.tif", "gfm_flood_*.gpkg", "gfm_exclusion_*.tif"):
        for stale in cfg.data_dir.glob(pattern):
            if stale.is_file():
                stale.unlink()
                log(f"removed legacy flat-layout file: {stale.name}")


def _write_scene(
    cfg: PipelineConfig,
    data: xr.DataArray,
    path: Path,
    scene: str,
    log: LogFn,
) -> list[Path]:
    """Write a flood raster plus its connected-area polygons (when any survive)."""
    written = [_write_geotiff(data, path, log)]
    polygons = polygonize.write_flood_polygons(
        path, min_area_ha=cfg.gfm.min_area_ha, scene=scene, log=log
    )
    if polygons is not None:
        written.append(polygons)
    return written


def _write_geotiff(data: xr.DataArray, path: Path, log: LogFn) -> Path:
    data = data.rio.write_crs("EPSG:4326")
    data.rio.to_raster(path)
    log(f"wrote {path}")
    return path
