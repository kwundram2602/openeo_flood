"""GFM flood-extent step: ensemble flood extent from the EODC STAC catalog.

Port of the former ``xarray_pipelines/get_gfm_image.py``: AOI, dates and
resolution come from the config, and outputs land in the configured data dir.
The temporal maximum is always written because it is FLEXTH's input mask; the
temporal sum is optional (``gfm.aggregation: sum`` or ``both``).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import odc.stac
import pystac
import pystac_client
import rioxarray  # noqa: F401  registers the .rio accessor used in _write_geotiff
import xarray as xr

from flood_pipeline.config import PipelineConfig
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
    max_items: int,
) -> pystac.ItemCollection:
    """Search the STAC catalog for GFM items covering bbox and time range.

    Standalone so the dashboard scene browser can reuse it (public API,
    no authentication).
    """
    catalog = pystac_client.Client.open(stac_url)
    search = catalog.search(
        bbox=list(bbox),
        datetime=list(temporal_extent),
        collections=[collection],
        max_items=max_items,
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


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Fetch the GFM flood extent for the AOI and write max (and sum) rasters."""
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
        max_items=cfg.gfm.max_items,
    )
    if len(items) == 0:
        raise RuntimeError(
            f"no {cfg.gfm.collection} items found for bbox {bbox} in "
            f"{cfg.gfm.temporal_extent}; widen gfm.temporal_extent or check the AOI"
        )
    log(f"found {len(items)} items, loading cube at {cfg.gfm.resolution} deg ...")

    flood = load_flood_cube(
        items, bbox, band=cfg.gfm.band, resolution=cfg.gfm.resolution
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    outputs = [_write_geotiff(flood.max(dim="time"), cfg.gfm_mask_path(), log)]
    if cfg.gfm.aggregation in ("sum", "both"):
        outputs.append(_write_geotiff(flood.sum(dim="time"), cfg.gfm_sum_path(), log))
    return StepOutcome(outputs=outputs)


def _write_geotiff(data: xr.DataArray, path: Path, log: LogFn) -> Path:
    data = data.rio.write_crs("EPSG:4326")
    data.rio.to_raster(path)
    log(f"wrote {path}")
    return path
