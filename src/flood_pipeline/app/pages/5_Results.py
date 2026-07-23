"""Results page: per-scene water-depth / water-level rasters on a map.

One scene per GFM acquisition timestamp; step through scenes with the slider.
FLEXTH conventions: WD_*.tif is uint16 centimeters (shown x0.01 as meters),
WL_*.tif is float32 meters; 0 is nodata and 999 marks permanent water — both
are masked out.
"""

import folium
import streamlit as st
from streamlit_folium import st_folium

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError
from flood_pipeline.steps import flexth_step

st.set_page_config(page_title="Results", page_icon="🌊", layout="wide")
st.title("Results")

try:
    cfg = ui.pipeline_cfg()
except ConfigError as e:
    st.error(str(e))
    st.stop()

scenes = flexth_step.find_scene_outputs(cfg.output_dir)
if not scenes:
    st.info(
        f"No per-scene WD_/WL_ rasters under `{cfg.output_dir}` yet — run the "
        "pipeline first (Run page)."
    )
    st.stop()


def _label(stamp: str) -> str:
    """'2024-09-16_051230' -> '2024-09-16 05:12'."""
    date, _, clock = stamp.partition("_")
    return f"{date} {clock[:2]}:{clock[2:4]}" if clock else stamp


stamps = list(scenes.keys())  # find_scene_outputs returns them sorted
stamp = st.select_slider("Scene", options=stamps, format_func=_label)
kind = st.radio("Layer", ["Water depth", "Water level"], horizontal=True)

wanted = "WD" if kind == "Water depth" else "WL"
selected = scenes[stamp].get(wanted)
if selected is None:
    st.warning(f"No {wanted} raster for scene {_label(stamp)}.")
    st.stop()

if wanted == "WD":
    scale, mask_values, label = 0.01, (0.0, 999.0), "water depth [m]"
else:
    scale, mask_values, label = 1.0, (999.0,), "water level [m]"

show_gfm = st.checkbox("Overlay GFM flood mask (red)", value=False)
use_max = st.checkbox(
    "Use whole-time GFM max instead of this scene",
    value=False,
    disabled=not show_gfm,
)

try:
    overlay = ui.raster_overlay(
        selected, cmap="Blues", mask_values=mask_values, scale=scale
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("min", f"{overlay.valid_min:.2f} m")
col2.metric("mean", f"{overlay.valid_mean:.2f} m")
col3.metric("max", f"{overlay.valid_max:.2f} m")
col4.metric("valid pixels", f"{overlay.valid_fraction:.1%}")
st.caption("Statistics exclude nodata (0) and the permanent-water sentinel (999).")

fmap = folium.Map(tiles="OpenStreetMap")
folium.raster_layers.ImageOverlay(
    image=overlay.rgba,
    bounds=overlay.bounds,
    opacity=1.0,  # per-pixel alpha already encodes transparency
    name=selected.name,
).add_to(fmap)

if show_gfm:
    gfm_path = cfg.gfm_mask_path() if use_max else cfg.gfm_scene_path(stamp)
    gfm_name = "GFM flood max" if use_max else f"GFM flood {_label(stamp)}"
    if gfm_path.exists():
        gfm_overlay = ui.raster_overlay(
            gfm_path, cmap="Reds", mask_values=(0.0,), scale=1.0
        )
        folium.raster_layers.ImageOverlay(
            image=gfm_overlay.rgba,
            bounds=gfm_overlay.bounds,
            opacity=0.6,
            name=gfm_name,
        ).add_to(fmap)
    else:
        st.warning(f"GFM raster not found: {gfm_path}")

if cfg.aoi_abs_path.exists():
    ui.add_aoi_layer(fmap, cfg.aoi_abs_path)
folium.LayerControl().add_to(fmap)
fmap.fit_bounds(overlay.bounds)

st_folium(
    fmap, key="results_map", height=600, use_container_width=True, returned_objects=[]
)

st.pyplot(
    ui.colorbar_figure(overlay.vmin, overlay.vmax, "Blues", label),
    use_container_width=False,
)
st.caption(
    "Color range is the 2nd-98th percentile of valid pixels; the map is a "
    "downsampled preview — use QGIS on the GeoTIFF for full resolution."
)
