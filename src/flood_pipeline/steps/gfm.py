"""GFM flood-extent step: ensemble flood extent from the EODC STAC catalog.

One raster is written per acquisition timestamp (cube time-slice); FLEXTH then
consumes each per-scene mask. The whole-time maximum is always written too (as a
reference/overlay) and the temporal sum is optional (``gfm.aggregation: sum`` or
``both``).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import odc.stac # used to create a (time, y, x) cube from the GFM STAC items
import pandas as pd
import pystac
import pystac_client # open stac catalog and search for items
import rioxarray
import xarray as xr

from flood_pipeline import polygonize
from flood_pipeline.config import SCENE_STAMP_FORMAT, PipelineConfig, vector_path
from flood_pipeline.steps import LogFn, StepOutcome

GFM_NODATA = 255


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


def _has_flood(scene: xr.DataArray) -> bool:
    """True if the scene has any flood pixel (value > 0) over the AOI.

    Empty scenes (all nodata/NaN or all-zero) come from overpass frames that do
    not cover the AOI or saw no flood; FLEXTH would produce an empty depth map.
    """
    return bool((scene > 0).any())


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Fetch the GFM flood extent and write one raster per flooded scene + max."""
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
    log(f"loading {len(items)} items as a cube at {cfg.gfm.resolution} deg ...")

    flood = load_flood_cube(
        items, bbox, band=cfg.gfm.band, resolution=cfg.gfm.resolution
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    # Clear per-scene rasters (and their polygons) from a previous run so the
    # on-disk set matches this run — skipped/empty scenes must not linger and
    # re-feed FLEXTH.
    for stale in cfg.gfm_scene_paths():
        stale.unlink()
        vector_path(stale).unlink(missing_ok=True)

    outputs: list[Path] = []
    scene_count = 0
    skipped = 0
    for time_value in flood["time"].values:
        stamp = pd.Timestamp(time_value).strftime(SCENE_STAMP_FORMAT)
        scene = flood.sel(time=time_value, drop=True)
        if not _has_flood(scene):
            log(f"skipped {stamp}: no flood pixels over the AOI")
            skipped += 1
            continue
        scene_count += 1
        outputs.extend(_write_scene(cfg, scene, cfg.gfm_scene_path(stamp), stamp, log))

    log(f"wrote {scene_count} flooded scene(s), skipped {skipped} empty scene(s)")
    outputs.extend(
        _write_scene(cfg, flood.max(dim="time"), cfg.gfm_mask_path(), "max", log)
    )
    if cfg.gfm.aggregation in ("sum", "both"):
        # No polygons here: sum > 0 covers exactly the same pixels as max > 0.
        outputs.append(_write_geotiff(flood.sum(dim="time"), cfg.gfm_sum_path(), log))
    return StepOutcome(outputs=outputs)


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
