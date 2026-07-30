"""OSM infrastructure step: roads and railways inside the AOI.

This step queries OpenStreetMap through OSMnx, stores AOI-wide road and railway
layers, then intersects them with each GFM flood extent to produce flooded
segments and km summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox

from flood_pipeline import polygonize
from flood_pipeline.app import ui
from flood_pipeline.config import PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

ROAD_LAYER = "roads"
RAILWAY_LAYER = "railways"
ROADS_FLOODED_LAYER = "roads_flooded"
RAILWAYS_FLOODED_LAYER = "railways_flooded"
METERS_PER_KM = 1000.0


def _aoi_geometry(cfg: PipelineConfig):
    aoi = ui.aoi_outline(cfg.aoi_abs_path).to_crs(epsg=4326)
    return aoi.geometry.unary_union


def _metric_crs(gdf: gpd.GeoDataFrame):
    try:
        return gdf.estimate_utm_crs()
    except Exception:
        return "EPSG:3857"


def _line_only(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if features.empty:
        return features
    return features[features.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()


def _roads_in_aoi(aoi_geometry) -> gpd.GeoDataFrame:
    graph = ox.graph_from_polygon(
        aoi_geometry,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    if edges.empty:
        return edges
    edges = edges.reset_index(drop=False)
    edges["feature_type"] = "road"
    return edges


def _railways_in_aoi(aoi_geometry) -> gpd.GeoDataFrame:
    features = ox.features_from_polygon(aoi_geometry, tags={"railway": True})
    features = _line_only(features)
    if features.empty:
        return features
    features = features.reset_index(drop=False)
    features["feature_type"] = "railway"
    return features


def _source_layers(cfg: PipelineConfig, aoi_geometry, log: LogFn) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[Path]]:
    roads_path = cfg.osm_source_roads_path()
    railways_path = cfg.osm_source_railways_path()
    outputs: list[Path] = []

    if roads_path.exists():
        roads = gpd.read_file(roads_path, layer=ROAD_LAYER)
        log(f"loaded cached roads: {roads_path}")
    else:
        log("querying OpenStreetMap roads via OSMnx")
        roads = _roads_in_aoi(aoi_geometry)
        roads_path.parent.mkdir(parents=True, exist_ok=True)
        if roads.empty:
            log("no road geometries returned for the AOI")
        else:
            roads.to_file(roads_path, layer=ROAD_LAYER, driver="GPKG")
            outputs.append(roads_path)
            log(f"wrote roads to {roads_path}")

    if railways_path.exists():
        railways = gpd.read_file(railways_path, layer=RAILWAY_LAYER)
        log(f"loaded cached railways: {railways_path}")
    else:
        log("querying OpenStreetMap railways via OSMnx")
        railways = _railways_in_aoi(aoi_geometry)
        railways_path.parent.mkdir(parents=True, exist_ok=True)
        if railways.empty:
            log("no railway geometries returned for the AOI")
        else:
            railways.to_file(railways_path, layer=RAILWAY_LAYER, driver="GPKG")
            outputs.append(railways_path)
            log(f"wrote railways to {railways_path}")

    return roads, railways, outputs


def _prepare_metric_layers(
    roads: gpd.GeoDataFrame, railways: gpd.GeoDataFrame, flood: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    metric_crs = _metric_crs(flood)

    roads_metric = roads.to_crs(metric_crs).copy() if not roads.empty else roads.copy()
    railways_metric = (
        railways.to_crs(metric_crs).copy() if not railways.empty else railways.copy()
    )
    flood_metric = flood.to_crs(metric_crs).copy()

    if not roads_metric.empty:
        roads_metric["length_m"] = roads_metric.geometry.length
    if not railways_metric.empty:
        railways_metric["length_m"] = railways_metric.geometry.length

    if roads_metric.empty:
        roads_flooded = roads_metric
    else:
        roads_flooded = gpd.overlay(
            roads_metric,
            flood_metric[["geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        if not roads_flooded.empty:
            roads_flooded["length_m"] = roads_flooded.geometry.length

    if railways_metric.empty:
        railways_flooded = railways_metric
    else:
        railways_flooded = gpd.overlay(
            railways_metric,
            flood_metric[["geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        if not railways_flooded.empty:
            railways_flooded["length_m"] = railways_flooded.geometry.length

    return roads_metric, railways_metric, roads_flooded, railways_flooded


def _write_layer(
    gdf: gpd.GeoDataFrame,
    path: Path,
    layer: str,
    log: LogFn,
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        log(f"no features for {layer}; skipped {path}")
        return None
    gdf.to_crs(epsg=4326).to_file(path, layer=layer, driver="GPKG")
    log(f"wrote {layer} to {path}")
    return path


def _write_summary(
    cfg: PipelineConfig,
    band: str,
    stamp: str,
    roads: gpd.GeoDataFrame,
    railways: gpd.GeoDataFrame,
    roads_flooded: gpd.GeoDataFrame,
    railways_flooded: gpd.GeoDataFrame,
    log: LogFn,
) -> Path:
    summary_path = cfg.osm_scene_summary_path(band, stamp)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "roads_total_km": float(roads.length_m.sum() / METERS_PER_KM) if not roads.empty else 0.0,
        "roads_flooded_km": float(roads_flooded.length_m.sum() / METERS_PER_KM)
        if not roads_flooded.empty
        else 0.0,
        "railways_total_km": float(railways.length_m.sum() / METERS_PER_KM)
        if not railways.empty
        else 0.0,
        "railways_flooded_km": float(railways_flooded.length_m.sum() / METERS_PER_KM)
        if not railways_flooded.empty
        else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote summary to {summary_path}")
    return summary_path


def _scene_flood_path(cfg: PipelineConfig, band: str, stamp: str) -> Path:
    return cfg.gfm_mask_path(band) if stamp == "max" else cfg.gfm_scene_path(band, stamp)


def _process_scene(
    cfg: PipelineConfig,
    band: str,
    stamp: str,
    roads: gpd.GeoDataFrame,
    railways: gpd.GeoDataFrame,
    log: LogFn,
) -> list[Path]:
    flood_path = _scene_flood_path(cfg, band, stamp)
    if not flood_path.exists():
        log(f"missing flood raster for {band}/{stamp}: {flood_path}")
        return []

    flood = polygonize.flood_polygons(flood_path, min_area_ha=0.0, scene=stamp)
    if flood.empty:
        log(f"no flooded pixels for {band}/{stamp}; skipped infrastructure intersection")
        return []

    roads_metric, railways_metric, roads_flooded, railways_flooded = _prepare_metric_layers(
        roads, railways, flood
    )
    if roads_metric.crs is None:
        roads_metric = roads_metric.set_crs("EPSG:4326")
    if railways_metric.crs is None:
        railways_metric = railways_metric.set_crs("EPSG:4326")
    if roads_flooded.crs is None and not roads_flooded.empty:
        roads_flooded = roads_flooded.set_crs("EPSG:4326")
    if railways_flooded.crs is None and not railways_flooded.empty:
        railways_flooded = railways_flooded.set_crs("EPSG:4326")

    scene_dir = cfg.osm_scene_dir(band, stamp)
    scene_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    roads_path = _write_layer(roads_metric, cfg.osm_scene_roads_path(band, stamp), ROAD_LAYER, log)
    if roads_path is not None:
        outputs.append(roads_path)
    railways_path = _write_layer(
        railways_metric, cfg.osm_scene_railways_path(band, stamp), RAILWAY_LAYER, log
    )
    if railways_path is not None:
        outputs.append(railways_path)
    flooded_roads_path = _write_layer(
        roads_flooded,
        cfg.osm_scene_flooded_roads_path(band, stamp),
        ROADS_FLOODED_LAYER,
        log,
    )
    if flooded_roads_path is not None:
        outputs.append(flooded_roads_path)
    flooded_railways_path = _write_layer(
        railways_flooded,
        cfg.osm_scene_flooded_railways_path(band, stamp),
        RAILWAYS_FLOODED_LAYER,
        log,
    )
    if flooded_railways_path is not None:
        outputs.append(flooded_railways_path)

    outputs.append(
        _write_summary(
            cfg,
            band,
            stamp,
            roads_metric,
            railways_metric,
            roads_flooded,
            railways_flooded,
            log,
        )
    )
    return outputs


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Extract roads and railways inside the AOI, then intersect them with flood extents."""
    aoi_geometry = _aoi_geometry(cfg)
    roads, railways, outputs = _source_layers(cfg, aoi_geometry, log)

    if roads.empty and railways.empty:
        raise RuntimeError(f"no OSM infrastructure found inside AOI {cfg.aoi_abs_path}")

    for key, _band_name in cfg.resolved_bands():
        log(f"== band {key}: processing OSM infrastructure impacts")
        outputs.extend(_process_band(cfg, key, roads, railways, log))

    return StepOutcome(outputs=outputs)


def _process_band(
    cfg: PipelineConfig,
    band: str,
    roads: gpd.GeoDataFrame,
    railways: gpd.GeoDataFrame,
    log: LogFn,
) -> list[Path]:
    outputs: list[Path] = []
    for stamp in [*cfg.gfm_scene_stamps(band), "max"]:
        outputs.extend(_process_scene(cfg, band, stamp, roads, railways, log))
    return outputs