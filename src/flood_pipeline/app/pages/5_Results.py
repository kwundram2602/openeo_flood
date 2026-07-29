"""Results page: per-scene water-depth / water-level / GHSL-depth rasters on a map.

One scene per GFM acquisition timestamp; step through scenes with the slider.
The connected flood areas of the scene's GFM mask are drawn as outlines and
listed largest-first, so the map can be zoomed straight to one of them.
FLEXTH conventions: WD_*.tif is uint16 centimeters (shown x0.01 as meters),
WL_*.tif is float32 meters; 0 is nodata and 999 marks permanent water — both
are masked out.
"""

import math

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError, vector_path
from flood_pipeline.steps import flexth_step

st.set_page_config(page_title="Results", page_icon="🌊", layout="wide")
st.title("Results")

WHOLE_AOI = "— whole AOI —"  # the flood-area selectbox entry that resets the view

try:
    cfg = ui.pipeline_cfg()
except ConfigError as e:
    st.error(str(e))
    st.stop()

bands = [key for key, _name in cfg.resolved_bands()]
if not bands:
    st.info(
        f"No per-scene WD_/WL_ rasters under `{cfg.output_dir}` yet — run the "
        "pipeline first (Run page)."
    )
    st.stop()
band = (
    bands[0]
    if len(bands) == 1
    else st.selectbox(
        "Algorithm band", bands, help="GFM flood-detection algorithm to display."
    )
)

scenes = flexth_step.find_scene_outputs(cfg.scene_output_root(band))
if not scenes:
    st.info(f"No scenes for band `{band}` yet — run the pipeline (Run page).")
    st.stop()


def _label(stamp: str) -> str:
    """'2024-09-16_051230' -> '2024-09-16 05:12'."""
    date, _, clock = stamp.partition("_")
    return f"{date} {clock[:2]}:{clock[2:4]}" if clock else stamp


def _ensure_3d_rgba(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 RGBA with shape (H, W, 4) for Folium/Branca."""
    arr = np.asarray(arr)

    # 1. Squeeze away any leading/extra single dimensions (e.g. (1, 1, H, W) -> (H, W))
    arr = np.squeeze(arr)

    if arr.ndim not in (2, 3):
        raise ValueError(
            f"_ensure_3d_rgba: expected 2D or 3D input after squeeze, got "
            f"shape {arr.shape} (ndim={arr.ndim}) — check how overlay.rgba is built."
        )

    # 2. If 2D (grayscale or indexed raster), convert to 4-channel RGBA
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr, np.full_like(arr, 255)])

    # 3. If channels-first (C, H, W) with C in {3, 4}, move channels to the end
    elif arr.shape[2] not in (3, 4) and arr.shape[0] in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # 4. If 3D with 3 channels (RGB), add Alpha channel to make it 4 (RGBA)
    if arr.ndim == 3 and arr.shape[2] == 3:
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=2)

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(
            f"_ensure_3d_rgba: could not coerce array to (H, W, 4), final shape "
            f"was {arr.shape} — check how overlay.rgba is built."
        )

    # Ensure contiguous uint8 array format
    return np.ascontiguousarray(arr, dtype=np.uint8)


stamps = list(scenes.keys())  # find_scene_outputs returns them sorted
stamp = st.select_slider("Scene", options=stamps, format_func=_label)
layer_options = ["Water depth", "Water level"]
if cfg.ghsl.enabled:
    layer_options.append("GHSL + Depth")

kind = st.radio("Layer", layer_options, horizontal=True)

if kind == "Water depth":
    wanted = "WD"
    selected = scenes[stamp].get(wanted)
    scale, mask_values, label = 0.01, (0.0, 999.0), "water depth [m]"
    cmap = "Blues"
    band_index = 0
elif kind == "Water level":
    wanted = "WL"
    selected = scenes[stamp].get(wanted)
    scale, mask_values, label = 1.0, (999.0,), "water level [m]"
    cmap = "Blues"
    band_index = 0
else:  # GHSL + Depth
    selected = cfg.scene_ghsl_depth_path(stamp)
    ghsl_view = st.radio(
        "GHSL view",
        list(ui.GHSL_DEPTH_BANDS),
        horizontal=True,
        help="GHSL class: settlement type per pixel, from GHSL's built_characteristics "
        "band. Relative damage: JRC/Huizinga depth-damage fraction (0-1) for this "
        "AOI's continent, at this scene's flood depth.",
    )
    band_index = ui.GHSL_DEPTH_BANDS[ghsl_view]
    if ghsl_view == "GHSL class":
        scale, mask_values, label = 1.0, (0.0,), "GHSL settlement class"
        cmap = "GHSL"
    else:
        scale, mask_values, label = 1.0, (0.0,), "relative flood damage [fraction]"
        cmap = "YlOrRd"

if selected is None or not selected.exists():
    st.warning(f"No raster found for layer '{kind}' in scene {_label(stamp)}.")
    st.stop()

show_gfm = st.checkbox("Overlay GFM flood mask (red)", value=False)
use_max = st.checkbox(
    "Use whole-time GFM max instead of this scene",
    value=False,
    disabled=not show_gfm,
)
show_reference_water = st.checkbox(
    "Overlay GFM Reference Water Mask (dark blue)", value=False
)
show_exclusion = st.checkbox(
    "Overlay GFM exclusion mask (grey) — pixels GFM could not evaluate",
    value=False,
)
show_fill = st.checkbox(
    "Show interpolation-added flood (orange) — pixels FLEXTH added beyond GFM",
    value=False,
)
show_likelihood = st.checkbox(
    "Overlay likelihood (viridis) — raw GFM probability, likelihood bands only",
    value=False,
)
likelihood_range = st.slider(
    "Likelihood color range [%]",
    min_value=0,
    max_value=100,
    value=(10, 100),
    disabled=not show_likelihood,
    help="Narrow the range (e.g. 10–50) to bring out low probabilities.",
)

gfm_path = cfg.gfm_mask_path(band) if use_max else cfg.gfm_scene_path(band, stamp)
areas = ui.flood_areas(vector_path(gfm_path))
selected_area_id = None
choice = WHOLE_AOI
if areas is None or areas.empty:
    st.caption(
        "No flood-area polygons for this raster — re-run the GFM step to create them."
    )
else:
    labels = {
        ui.flood_area_label(row.area_id, row.area_ha): row.area_id
        for row in areas.itertuples()
    }
    choice = st.selectbox(
        "Flood area",
        [WHOLE_AOI, *labels],
        help="Zoom the map to one connected flood area; areas are ranked by size.",
    )
    selected_area_id = labels.get(choice)

try:
    overlay = ui.raster_overlay(
        selected, cmap=cmap, mask_values=mask_values, scale=scale, band_index=band_index
    )
except ValueError:
    st.info(
        f"Scene {_label(stamp)} has no valid pixels for {kind.lower()} — the "
        "extent was empty over the AOI for this acquisition."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("min", f"{overlay.valid_min:.2f}")
col2.metric("mean", f"{overlay.valid_mean:.2f}")
col3.metric("max", f"{overlay.valid_max:.2f}")
col4.metric("valid pixels", f"{overlay.valid_fraction:.1%}")
st.caption("Statistics exclude nodata values.")

# Base map holds only the tiles and its fit_bounds, so its rendered string stays
# identical from rerun to rerun. streamlit-folium then never reloads it and the
# browser keeps whatever pan/zoom the user set — stepping the slider only swaps
# the feature group below. Frame on a constant "home" extent (the AOI, else the
# first overlay) so the fixed initial view is stable across reruns too.
if "results_home_bounds" not in st.session_state:
    if cfg.aoi_abs_path.exists():
        minx, miny, maxx, maxy = ui.aoi_outline(cfg.aoi_abs_path).total_bounds
        st.session_state["results_home_bounds"] = [[miny, minx], [maxy, maxx]]
    else:
        st.session_state["results_home_bounds"] = overlay.bounds

# The view only moves when the user actively picks something in the flood-area
# box; it is remembered so every other rerun leaves fit_bounds — and with it the
# map's identity to streamlit-folium — untouched. Stepping the scene slider
# swaps the polygon set, so Streamlit drops the selection on its own; reading
# that reset as "zoom back out" would throw away the view just jumped to. Hence
# the comparison against the raster the previous selection belonged to.
st.session_state.setdefault("results_view_bounds", st.session_state["results_home_bounds"])
selection = (str(gfm_path), choice)
previous = st.session_state.get("results_area_selection")
st.session_state["results_area_selection"] = selection
if previous is not None and previous[0] == selection[0] and previous[1] != choice:
    if selected_area_id is None:
        st.session_state["results_view_bounds"] = st.session_state["results_home_bounds"]
    else:
        geometry = areas.loc[areas["area_id"] == selected_area_id, "geometry"].iloc[0]
        st.session_state["results_view_bounds"] = ui.jump_bounds(geometry)

fmap = folium.Map(tiles="OpenStreetMap")
fmap.fit_bounds(st.session_state["results_view_bounds"])

# Everything that changes with the slider/toggles goes into the feature group.
overlays = folium.FeatureGroup(name="overlays")
folium.raster_layers.ImageOverlay(
    image=_ensure_3d_rgba(overlay.rgba),
    bounds=overlay.bounds,
    opacity=1.0,  # per-pixel alpha already encodes transparency
    name=selected.name,
).add_to(overlays)

if show_gfm:
    gfm_name = "GFM flood max" if use_max else f"GFM flood {_label(stamp)}"
    if gfm_path.exists():
        # Solid red: the mask is binary, so colormapping its single remaining
        # value would render it at the near-white low end of "Reds".
        gfm_overlay = ui.raster_overlay(
            gfm_path, mask_values=(0.0,), scale=1.0, solid_color="#e31a1c"
        )
        folium.raster_layers.ImageOverlay(
            image=_ensure_3d_rgba(gfm_overlay.rgba),
            bounds=gfm_overlay.bounds,
            opacity=0.75,
            name=gfm_name,
        ).add_to(overlays)
    else:
        st.warning(f"GFM raster not found: {gfm_path}")

if show_reference_water:
    reference_path = cfg.gfm_reference_water_path(band)
    if reference_path.exists():
        # Solid dark blue: the reference mask is binary permanent water, so
        # colormapping its single value would wash out like the flood mask.
        reference_overlay = ui.raster_overlay(
            reference_path, mask_values=(0.0,), scale=1.0, solid_color="#08306b"
        )
        folium.raster_layers.ImageOverlay(
            image=reference_overlay.rgba,
            bounds=reference_overlay.bounds,
            opacity=0.75,
            name="GFM reference water",
        ).add_to(overlays)
    else:
        st.caption(
            "No GFM reference water mask yet — re-run the GFM step to create it."
        )

if show_exclusion:
    # Always this scene's own mask, never the max toggle: the exclusion is
    # pass-geometry specific, so it belongs to the acquisition being shown.
    exclusion_path = cfg.gfm_exclusion_path(band, stamp)
    if exclusion_path.exists():
        try:
            # Solid grey: binary mask of pixels GFM could not evaluate.
            exclusion_overlay = ui.raster_overlay(
                exclusion_path, mask_values=(0.0,), scale=1.0, solid_color="#606060"
            )
        except ValueError:
            st.caption(
                f"GFM excluded nothing over the AOI for scene {_label(stamp)} — "
                "this pass evaluated the whole area."
            )
        else:
            folium.raster_layers.ImageOverlay(
                image=exclusion_overlay.rgba,
                bounds=exclusion_overlay.bounds,
                opacity=0.6,
                name=f"GFM exclusion {_label(stamp)}",
            ).add_to(overlays)
    else:
        st.caption(
            "No GFM exclusion mask for this scene — re-run the GFM step to create it."
        )

if show_fill:
    fill_path = cfg.scene_fill_path(band, stamp)
    if fill_path.exists():
        try:
            # Solid orange: binary mask of what FLEXTH added beyond the GFM extent.
            fill_overlay = ui.raster_overlay(
                fill_path, mask_values=(0.0,), scale=1.0, solid_color="#ff7f00"
            )
        except ValueError:
            st.caption(f"No interpolation-added pixels for scene {_label(stamp)}.")
        else:
            folium.raster_layers.ImageOverlay(
                image=fill_overlay.rgba,
                bounds=fill_overlay.bounds,
                opacity=0.75,
                name=f"interpolation-added {_label(stamp)}",
            ).add_to(overlays)
    else:
        st.caption(
            "No interpolation-added raster for this scene — re-run FLEXTH (needs a "
            "WD output; enable flexth.fill_excluded for the urban fill)."
        )

if show_likelihood:
    likelihood_path = cfg.gfm_likelihood_path(band, stamp)
    if likelihood_path.exists():
        try:
            likelihood_overlay = ui.raster_overlay(
                likelihood_path,
                cmap="viridis",
                vmin=float(likelihood_range[0]),
                vmax=float(likelihood_range[1]),
            )
        except ValueError:
            st.caption(f"No likelihood values for scene {_label(stamp)}.")
        else:
            folium.raster_layers.ImageOverlay(
                image=likelihood_overlay.rgba,
                bounds=likelihood_overlay.bounds,
                opacity=0.75,
                name=f"likelihood {_label(stamp)}",
            ).add_to(overlays)
    else:
        st.caption(
            f"Band `{band}` has no likelihood raster — select a *_likelihood band."
        )

if cfg.aoi_abs_path.exists():
    ui.add_aoi_layer(overlays, cfg.aoi_abs_path)

# The probe point survives slider/toggle reruns, so the same spot can be read
# across scenes and layers. It is drawn from session state (not straight from
# the click) because the marker has to be in the feature group *before*
# st_folium renders it.
probe = st.session_state.get("results_probe")
if probe is not None:
    folium.CircleMarker(
        location=list(probe),
        radius=5,
        color="#111111",
        weight=2,
        fill=True,
        fill_color="#ffffff",
        fill_opacity=1.0,
        tooltip="probe",
    ).add_to(overlays)

map_state = st_folium(
    fmap,
    key="results_map",
    height=600,
    use_container_width=True,
    feature_group_to_add=overlays,
    returned_objects=["last_clicked"],
)

# st_folium keeps reporting the same last_clicked on every rerun, so remember
# which click was already consumed — otherwise clearing the probe would
# immediately re-create it from the stale click.
clicked = (map_state or {}).get("last_clicked")
if clicked is not None:
    point = (float(clicked["lat"]), float(clicked["lng"]))
    if point != st.session_state.get("results_probe_click"):
        st.session_state["results_probe_click"] = point
        st.session_state["results_probe"] = point
        st.rerun()  # redraw so the marker lands on the freshly clicked point

if probe is None:
    st.caption("Click the map to read the value at a point.")
else:
    lat, lon = probe
    value = ui.sample_raster(
        selected, lat, lon, mask_values=mask_values, scale=scale, band_index=band_index
    )
    if value is None:
        reading, note = "—", "Point is outside the raster."
    elif math.isnan(value):
        reading, note = "NaN", "No value here — nodata or permanent water (999)."
    elif kind == "Water depth" or kind == "Water level":
        reading, note = f"{value:.2f} m", "Value read from the full-resolution GeoTIFF."
    elif band_index == ui.GHSL_DEPTH_BANDS["Relative damage"]:
        reading, note = f"{value:.2f}", "Relative damage fraction (0-1); value read from the full-resolution GeoTIFF."
    else:
        reading, note = f"{value:.0f}", "GHSL class code; value read from the full-resolution GeoTIFF."
    probe_col, coord_col = st.columns([1, 3])
    probe_col.metric(kind, reading)
    coord_col.caption(f"at {lat:.5f}, {lon:.5f} — {note}")
    likelihood_path = cfg.gfm_likelihood_path(band, stamp)
    if likelihood_path.exists():
        lk = ui.sample_raster(likelihood_path, lat, lon)
        if lk is None or math.isnan(lk):
            coord_col.caption("likelihood: —")
        else:
            coord_col.caption(f"likelihood: {lk:.0f} %")
    if coord_col.button("Clear probe"):
        del st.session_state["results_probe"]
        st.rerun()

st.pyplot(
    ui.colorbar_figure(overlay.vmin, overlay.vmax, cmap, label),
    use_container_width=False,
)
st.caption(
    "Color range is the 2nd-98th percentile of valid pixels; the map is a "
    "downsampled preview — use QGIS on the GeoTIFF for full resolution."
)