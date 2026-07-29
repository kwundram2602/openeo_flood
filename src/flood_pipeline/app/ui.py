"""Shared dashboard helpers: config session state and raster overlays.

The dashboard edits the raw config dict (kept in ``st.session_state``) so the
YAML round-trips without schema friction; typed access for paths/validation
goes through :func:`pipeline_cfg`, which reads the file as saved on disk.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import matplotlib.colors
import matplotlib.patches
import numpy as np
import streamlit as st
import yaml

from flood_pipeline.cli import CONFIG_ENV_VAR
from flood_pipeline.config import FLOOD_AREA_LAYER, PipelineConfig, load_config

DEFAULT_CONFIG_NAME = "config.yaml"
PROJECTS_DIR_NAME = "projects"
OVERLAY_OPACITY = 0.85
CMAP_FLOOR = 0.25  # where the lowest displayed value lands in the colormap


@functools.cache
def display_colormap(cmap: str) -> matplotlib.colors.Colormap:
    """``cmap`` without its washed-out low end, for both overlay and legend.

    Sequential maps start at near-white ("Blues" at RGB 247, 251, 255), which
    is invisible over the basemap. That end is where the percentile stretch
    puts a lot of pixels: FLEXTH floors water depth at its minimum, so a large
    share of a WD raster carries exactly that one value and ``vmin`` lands on
    it. Starting a quarter into the ramp keeps the shallowest water readable.
    """
    base = matplotlib.colormaps[cmap]
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        f"{cmap}_from_{CMAP_FLOOR}", base(np.linspace(CMAP_FLOOR, 1.0, 256))
    )

# Exact GHSL settlement classification color mapping
GHSL_COLOR_DICT: dict[int, str] = {
    1: "#718c6c",   # open spaces, low vegetation surfaces
    2: "#8ad86b",   # open spaces, medium vegetation surfaces
    3: "#c1ffa1",   # open spaces, high vegetation surfaces
    4: "#01b7ff",   # open spaces, water surfaces
    5: "#ffd501",   # open spaces, road surfaces
    11: "#d28200",  # built spaces, residential, height <= 3m
    12: "#fe5900",  # built spaces, residential, 3m < height <= 6m
    13: "#ff0101",  # built spaces, residential, 6m < height <= 15m
    14: "#ce001b",  # built spaces, residential, 15m < height <= 30m
    15: "#7a000a",  # built spaces, residential, height > 30m
    21: "#ff9ff4",  # built spaces, non-residential, height <= 3m
    22: "#ff67e4",  # built spaces, non-residential, 3m < height <= 6m
    23: "#f701ff",  # built spaces, non-residential, 6m < height <= 15m
    24: "#a601ff",  # built spaces, non-residential, 15m < height <= 30m
    25: "#6e00fe",  # built spaces, non-residential, height > 30m
}

# GHSL+Depth+Damage raster band layout (matches ghsl_step._combine_ghsl_and_depth).
# Values are 0-indexed band_index for ui.raster_overlay()/sample_raster().
GHSL_DEPTH_BANDS: dict[str, int] = {
    "GHSL class": 0,       # band 1: GHSL characteristics, discrete classes
    "Relative damage": 2,  # band 3: JRC depth-damage fraction, 0-1
}


def discover_project_configs() -> list[Path]:
    """config.yaml files of projects under <cwd>/projects, sorted by name."""
    return sorted((Path.cwd() / PROJECTS_DIR_NAME).glob(f"*/{DEFAULT_CONFIG_NAME}"))


def default_config_path() -> Path | None:
    """The config to open on startup: env var, ./config.yaml, first project."""
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value)
    local = Path.cwd() / DEFAULT_CONFIG_NAME
    if local.is_file():
        return local
    discovered = discover_project_configs()
    if discovered:
        return discovered[0]
    return None


def create_project(name: str, parent: Path, template: dict | None) -> Path:
    """Create <parent>/<name>/config.yaml (from a template dict) and return it.

    The template is usually the currently loaded config, so a new project
    inherits GEE project, dates and FLEXTH parameters; its AOI starts empty
    (draw one on the AOI page).
    """
    import copy

    project_dir = parent / name
    config_path = project_dir / DEFAULT_CONFIG_NAME
    if config_path.exists():
        raise FileExistsError(f"project already exists: {config_path}")
    project_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = copy.deepcopy(template) if template else {}
    cfg_dict.setdefault("project", {})["name"] = name
    cfg_dict.setdefault("aoi", {})["path"] = "aoi_drawn.geojson"
    # Every new project gets its data folder up front.
    data_dir = cfg_dict.get("project", {}).get("data_dir", "flood_data")
    (project_dir / data_dir).mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def load_cfg(path: Path) -> None:
    """(Re)load the config file into session state."""
    st.session_state["cfg_dict"] = (
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    )
    st.session_state["cfg_path"] = path.resolve()


def get_cfg() -> tuple[dict, Path]:
    """The raw config dict and its path; loads the default on first access.

    Halts the page with a hint when no project is available yet.
    """
    if "cfg_path" not in st.session_state:
        default = default_config_path()
        if default is None or not default.is_file():
            st.warning("No project loaded — open or create one on the **Home** page.")
            st.stop()
        load_cfg(default)
    return st.session_state["cfg_dict"], st.session_state["cfg_path"]


def save_cfg() -> None:
    """Write the session config dict back to its YAML file."""
    cfg_dict, cfg_path = get_cfg()
    cfg_path.write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def pipeline_cfg() -> PipelineConfig:
    """Typed view of the config *as saved on disk* (for paths and validation)."""
    _, cfg_path = get_cfg()
    return load_config(cfg_path)


def configured_output_bands(cfg: PipelineConfig) -> list[str]:
    """Bands with outputs on disk that the *current* config actually produces.

    :meth:`PipelineConfig.gfm_output_bands` reports every band folder present,
    which still includes the leftovers of earlier runs with different ``gfm``
    settings — a different date range, and no OSM outputs at all (the osm step
    only writes for :meth:`PipelineConfig.resolved_bands`). The results page
    must neither offer nor load those, so filter them out here. Order follows
    ``resolved_bands`` so the picker opens on the run's primary band.
    """
    on_disk = set(cfg.gfm_output_bands())
    return [key for key, _band_name in cfg.resolved_bands() if key in on_disk]


def aoi_outline(aoi_path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(aoi_path).to_crs(epsg=4326)


def add_aoi_layer(parent: folium.Map | folium.FeatureGroup, aoi_path: Path) -> None:
    outline = aoi_outline(aoi_path)
    folium.GeoJson(
        outline.__geo_interface__,
        name="AOI",
        style_function=lambda _feat: {
            "color": "#d62728",
            "weight": 2,
            "fill": False,
            "dashArray": "6 4",
        },
    ).add_to(parent)


def flood_areas(vector: Path) -> gpd.GeoDataFrame | None:
    """Connected flood areas of a GFM raster, largest first; None if not written.

    The file is absent for data produced before the polygons existed, and for
    rasters whose areas all fell below ``gfm.min_area_ha``.
    """
    if not vector.exists():
        return None
    return gpd.read_file(vector, layer=FLOOD_AREA_LAYER)


def flood_area_label(area_id: int, area_ha: float) -> str:
    """Selectbox label for one flood area: ``#1 — 124.3 ha``."""
    return f"#{area_id} — {area_ha:.1f} ha"


def jump_bounds(geometry, *, pad_fraction: float = 0.25, min_pad: float = 0.001):
    """A folium ``fit_bounds`` box around a geometry, with breathing room.

    Padded by a fraction of the geometry's own size so large and small areas
    both land sensibly framed, but never by less than ``min_pad`` degrees —
    a single-pixel area would otherwise fit-bounds to a degenerate box and the
    map would slam to maximum zoom.
    """
    west, south, east, north = geometry.bounds
    pad_x = max((east - west) * pad_fraction, min_pad)
    pad_y = max((north - south) * pad_fraction, min_pad)
    return [[south - pad_y, west - pad_x], [north + pad_y, east + pad_x]]


def add_flood_area_layer(
    parent: folium.Map | folium.FeatureGroup,
    areas: gpd.GeoDataFrame,
    *,
    selected_id: int | None = None,
) -> None:
    """Draw the flood-area outlines, the selected one emphasized."""
    for row in areas.itertuples():
        chosen = selected_id is not None and row.area_id == selected_id
        folium.GeoJson(
            row.geometry.__geo_interface__,
            name=f"flood area {row.area_id}",
            # Bind `chosen` per iteration: the style function is called later,
            # when the loop variable would already point at the last row.
            style_function=lambda _feat, chosen=chosen: {
                "color": "#ff7f0e" if chosen else "#8c564b",
                "weight": 2 if chosen else 1,
                "fill": True,
                "fillColor": "#ff7f0e" if chosen else "#8c564b",
                "fillOpacity": 0.35,
            },
            tooltip=flood_area_label(row.area_id, row.area_ha),
        ).add_to(parent)


def format_share(fraction: float) -> str:
    """A fraction as a percentage that never rounds a non-zero share to "0.0%".

    Shares this small are the interesting case: a pre-flood acquisition can hold
    a few dozen flooded pixels in a raster of millions, and ``0.0%`` would
    report that as an empty scene. Below 0.1% the share therefore switches to
    two significant digits instead of one decimal place.
    """
    percent = fraction * 100.0
    if percent == 0.0:
        return "0%"
    return f"{percent:.1f}%" if percent >= 0.1 else f"{percent:.2g}%"


@dataclass
class RasterOverlay:
    """A display-ready raster: RGBA image, folium bounds and value stats."""

    rgba: np.ndarray
    bounds: list[list[float]]  # [[south, west], [north, east]] in EPSG:4326
    vmin: float
    vmax: float
    valid_min: float
    valid_mean: float
    valid_max: float
    valid_fraction: float


def raster_overlay(
    path: Path | str,
    *,
    cmap: str = "Blues",
    max_dim: int = 1500,
    mask_values: tuple[float, ...] = (),
    scale: float = 1.0,
    solid_color: str | None = None,
    band_index: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> RasterOverlay:
    """Load a raster as a colormapped RGBA overlay in EPSG:4326."""
    path = Path(path)
    fields = _build_overlay(
        path_str=str(path),
        _mtime=path.stat().st_mtime,
        cmap=cmap,
        max_dim=max_dim,
        mask_values=mask_values,
        scale=scale,
        solid_color=solid_color,
        band_index=band_index,
        vmin_override=vmin,
        vmax_override=vmax,
    )
    return RasterOverlay(**fields)


@st.cache_data(show_spinner="rendering raster overlay ...")
def _build_overlay(
    path_str: str,
    _mtime: float,  # cache key only: invalidates when the file changes
    cmap: str,
    max_dim: int,
    mask_values: tuple[float, ...],
    scale: float,
    solid_color: str | None = None,
    band_index: int = 0,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
) -> dict:
    """Compute the overlay fields as a plain dict."""
    import rioxarray

    data = rioxarray.open_rasterio(path_str, masked=True)
    if "band" in data.dims:
        band_count = data.sizes["band"]
        if band_index >= band_count:
            raise ValueError(
                f"requested band_index={band_index} but {path_str} only has "
                f"{band_count} band(s)"
            )
        data = data.isel(band=band_index)
    data = data.squeeze(drop=True)
    step_y = max(1, data.sizes["y"] // max_dim)
    step_x = max(1, data.sizes["x"] // max_dim)
    if step_y > 1 or step_x > 1:
        data = data.isel(y=slice(None, None, step_y), x=slice(None, None, step_x))
    data = data.rio.reproject("EPSG:4326")

    values = data.values.astype("float32")
    nodata = data.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        values = np.where(values == nodata, np.nan, values)
    for mask_value in mask_values:
        values = np.where(values == mask_value, np.nan, values)
    values *= scale

    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        raise ValueError(f"no valid pixels to display in {path_str}")
    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = float(vmin_override), float(vmax_override)
    else:
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if solid_color is not None:
        rgba = np.tile(matplotlib.colors.to_rgba(solid_color), values.shape + (1,))
    elif cmap == "GHSL":
        rgba = np.zeros(values.shape + (4,), dtype=np.float32)
        int_values = np.nan_to_num(values, nan=0).astype(int)
        known_class_mask = np.zeros(values.shape, dtype=bool)
        for class_id, hex_code in GHSL_COLOR_DICT.items():
            class_mask = (int_values == class_id) & finite_mask
            rgba[class_mask] = matplotlib.colors.to_rgba(hex_code)
            known_class_mask |= class_mask
        finite_mask = finite_mask & known_class_mask
    else:
        normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        rgba = matplotlib.colormaps[cmap](np.nan_to_num(normalized, nan=0.0))

    rgba[..., 3] = np.where(finite_mask, OVERLAY_OPACITY, 0.0)

    left, bottom, right, top = data.rio.bounds()
    return {
        "rgba": (rgba * 255).astype("uint8"),
        "bounds": [[float(bottom), float(left)], [float(top), float(right)]],
        "vmin": vmin,
        "vmax": vmax,
        "valid_min": float(finite.min()),
        "valid_mean": float(finite.mean()),
        "valid_max": float(finite.max()),
        "valid_fraction": float(finite.size / values.size),
    }


def sample_raster(
    path: Path,
    lat: float,
    lon: float,
    *,
    mask_values: tuple[float, ...] = (),
    scale: float = 1.0,
    band_index: int = 0,
) -> float | None:
    """Value of ``path`` at a WGS84 point, read at full resolution."""
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(path) as src:
        xs, ys = warp_transform(
            "EPSG:4326", src.crs or "EPSG:4326", [float(lon)], [float(lat)]
        )
        row, col = src.index(xs[0], ys[0])
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        window = rasterio.windows.Window(col, row, 1, 1)
        pixel = src.read(band_index + 1, window=window, masked=True).astype("float32")
        value = float(pixel.filled(np.nan)[0, 0])

    if value in mask_values:
        return float("nan")
    return value * scale


def colorbar_figure(vmin: float, vmax: float, cmap: str, label: str):
    """A slim horizontal colorbar to serve as the map legend."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 0.9))
    ax = fig.add_axes((0.05, 0.55, 0.9, 0.3))

    if cmap == "GHSL":
        patches = [
            matplotlib.patches.Patch(color=hex_code, label=f"Class {cid}")
            for cid, hex_code in GHSL_COLOR_DICT.items()
        ]
        fig.legend(handles=patches, loc="center", ncol=3, fontsize="x-small", frameon=False)
        ax.axis("off")
        return fig

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=display_colormap(cmap),
    )
    fig.colorbar(mappable, cax=ax, orientation="horizontal")
    ax.set_xlabel(label)
    return fig