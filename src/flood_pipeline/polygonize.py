"""Connected flood areas of a flood raster, as polygons.

The GFM step writes binary flood masks; this turns each one into the set of its
connected flood areas so they can be listed, measured and jumped to on the map.
Not a pipeline step (those expose ``run(cfg, log)``) — a plain helper that
:mod:`flood_pipeline.steps.gfm` calls for every raster it writes.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import shape

from flood_pipeline.config import FLOOD_AREA_LAYER, vector_path
from flood_pipeline.steps import LogFn

# Matches flexth's flood_processing.params.connectivity, so the raster and the
# vector view agree on what "connected" means.
CONNECTIVITY = 8
SQUARE_METRES_PER_HECTARE = 10_000.0


def flood_polygons(
    raster: Path, *, min_area_ha: float, scene: str
) -> gpd.GeoDataFrame:
    """Connected flood areas of ``raster``, largest first, in EPSG:4326.

    A pixel counts as flooded when it is valid (not nodata) and greater than
    zero. Areas smaller than ``min_area_ha`` are dropped (``0`` keeps all), then
    ``area_id`` is assigned 1..N by descending size, so the id is the size rank.
    """
    with rasterio.open(raster) as src:
        band = src.read(1, masked=True)
        flooded = np.ma.filled(band > 0, False).astype("uint8")
        shapes = rasterio.features.shapes(
            flooded,
            mask=flooded.astype(bool),
            transform=src.transform,
            connectivity=CONNECTIVITY,
        )
        geometries = [shape(geometry) for geometry, _value in shapes]
        crs = src.crs

    areas = gpd.GeoDataFrame(
        {"scene": [scene] * len(geometries)}, geometry=geometries, crs=crs
    )
    if areas.empty:
        # estimate_utm_crs needs a geometry to work from; skip straight to the
        # empty result with the right columns and CRS.
        return gpd.GeoDataFrame(
            {"area_id": [], "area_ha": [], "scene": []},
            geometry=[],
            crs="EPSG:4326",
        )

    # Areas are measured in a metric projection: degrees would make them depend
    # on latitude, and GFM rasters are EPSG:4326.
    metric = areas.to_crs(areas.estimate_utm_crs())
    areas["area_ha"] = metric.area / SQUARE_METRES_PER_HECTARE

    areas = areas[areas["area_ha"] >= min_area_ha]
    areas = areas.sort_values("area_ha", ascending=False).reset_index(drop=True)
    areas.insert(0, "area_id", np.arange(1, len(areas) + 1))
    return areas.to_crs(epsg=4326)[["area_id", "area_ha", "scene", "geometry"]]


def write_flood_polygons(
    raster: Path, *, min_area_ha: float, scene: str, log: LogFn
) -> Path | None:
    """Write the flood areas of ``raster`` beside it; return the path written.

    Returns ``None`` and writes nothing when no area survives the filter — an
    empty GeoPackage layer is more trouble to consume than an absent file.
    """
    areas = flood_polygons(raster, min_area_ha=min_area_ha, scene=scene)
    if areas.empty:
        log(f"no flood area >= {min_area_ha} ha in {raster.name}; no polygons written")
        return None

    target = vector_path(raster)
    areas.to_file(target, layer=FLOOD_AREA_LAYER, driver="GPKG")
    log(f"wrote {len(areas)} flood area(s) to {target}")
    return target
