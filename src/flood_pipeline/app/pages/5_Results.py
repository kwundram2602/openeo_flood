"""Results page: per-scene water-depth / water-level rasters on a map.

One scene per GFM acquisition timestamp; step through scenes with the slider.
The connected flood areas of the scene's GFM mask are drawn as outlines and
listed largest-first, so the map can be zoomed straight to one of them.
FLEXTH conventions: WD_*.tif is uint16 centimeters (shown x0.01 as meters),
WL_*.tif is float32 meters; 0 is nodata and 999 marks permanent water — both
are masked out.
"""

import json
import math
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError, vector_path
from flood_pipeline.steps import flexth_step
from flood_pipeline.steps.population import calculate_exposure

st.set_page_config(page_title="Results", page_icon="🌊", layout="wide")
st.title("Results")

WHOLE_AOI = "— whole AOI —"  # the flood-area selectbox entry that resets the view
METERS_PER_KM = 1000.0


def _load_gdf(path: Path, layer: str) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    return gpd.read_file(path, layer=layer)


def _length_km(gdf: gpd.GeoDataFrame | None) -> float:
    if gdf is None or gdf.empty:
        return 0.0
    if "length_m" in gdf.columns:
        return float(gdf["length_m"].sum() / METERS_PER_KM)
    metric = gdf.to_crs(gdf.estimate_utm_crs() or "EPSG:3857")
    return float(metric.geometry.length.sum() / METERS_PER_KM)


def _summary_from_layers(
    roads: gpd.GeoDataFrame | None,
    railways: gpd.GeoDataFrame | None,
    flooded_roads: gpd.GeoDataFrame | None,
    flooded_railways: gpd.GeoDataFrame | None,
) -> dict[str, float]:
    return {
        "roads_total_km": _length_km(roads),
        "roads_flooded_km": _length_km(flooded_roads),
        "railways_total_km": _length_km(railways),
        "railways_flooded_km": _length_km(flooded_railways),
    }


def _load_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _add_line_layer(
    parent: folium.Map | folium.FeatureGroup,
    gdf: gpd.GeoDataFrame | None,
    *,
    name: str,
    color: str,
    weight: int,
    opacity: float,
) -> None:
    if gdf is None or gdf.empty:
        return
    folium.GeoJson(
        gdf.to_crs(epsg=4326).__geo_interface__,
        name=name,
        style_function=lambda _feature, color=color, weight=weight, opacity=opacity: {
            "color": color,
            "weight": weight,
            "opacity": opacity,
            "fill": False,
        },
    ).add_to(parent)

try:
    cfg = ui.pipeline_cfg()
except ConfigError as e:
    st.error(str(e))
    st.stop()

bands = cfg.gfm_output_bands()
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


stamps = list(scenes.keys())  # find_scene_outputs returns them sorted
if len(stamps) == 1:
    stamp = stamps[0]
    st.caption(f"Scene: {_label(stamp)}")
else:
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
population_path = cfg.population_path()
show_population = st.checkbox(
    "Overlay WorldPop population (purple)",
    value=False,
    disabled=not population_path.exists(),
)

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

infra_stamp = "max" if use_max else stamp
osm_roads_path = cfg.osm_scene_roads_path(band, infra_stamp)
osm_railways_path = cfg.osm_scene_railways_path(band, infra_stamp)
osm_flooded_roads_path = cfg.osm_scene_flooded_roads_path(band, infra_stamp)
osm_flooded_railways_path = cfg.osm_scene_flooded_railways_path(band, infra_stamp)
osm_summary_path = cfg.osm_scene_summary_path(band, infra_stamp)

show_roads = st.checkbox(
    "Overlay roads (grey)",
    value=False,
    disabled=not osm_roads_path.exists(),
)
show_railways = st.checkbox(
    "Overlay railways (teal)",
    value=False,
    disabled=not osm_railways_path.exists(),
)
show_flooded_infra = st.checkbox(
    "Overlay flooded infrastructure (red/orange)",
    value=True,
    disabled=not osm_flooded_roads_path.exists() and not osm_flooded_railways_path.exists(),
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
        selected, cmap="Blues", mask_values=mask_values, scale=scale
    )
except ValueError:
    st.info(
        f"Scene {_label(stamp)} has no {kind.lower()} to show — the GFM flood "
        "extent was empty over the AOI for this acquisition."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("min", f"{overlay.valid_min:.2f} m")
col2.metric("mean", f"{overlay.valid_mean:.2f} m")
col3.metric("max", f"{overlay.valid_max:.2f} m")
col4.metric("valid pixels", ui.format_share(overlay.valid_fraction))
st.caption(
    "Statistics exclude nodata (0) and the permanent-water sentinel (999). "
    "A share far below 1% is a real reading, not an empty scene — acquisitions "
    "before or after the event carry only a few flooded pixels."
)

if population_path.exists() and wanted == "WD":
    try:
        exposure = calculate_exposure(population_path, selected)
        st.subheader("Population exposure")
        exp1, exp2, exp3 = st.columns(3)
        exp1.metric("Population in AOI", f"{exposure.total_population:,.0f}")
        exp2.metric("Estimated exposed population", f"{exposure.exposed_population:,.0f}")
        exp3.metric("Exposed share", f"{exposure.exposed_percent:.1f}%")
        st.caption(
            "WorldPop 2020 residential population × fractional modeled flood "
            "coverage per population cell. Values are estimates, not a live headcount."
        )
    except (OSError, ValueError) as e:
        st.warning(f"Could not calculate population exposure: {e}")
elif not population_path.exists():
    st.info("Run the population step to add WorldPop exposure statistics.")

roads_layer = _load_gdf(osm_roads_path, "roads") if show_roads else None
railways_layer = _load_gdf(osm_railways_path, "railways") if show_railways else None
flooded_roads_layer = (
    _load_gdf(osm_flooded_roads_path, "roads_flooded") if show_flooded_infra else None
)
flooded_railways_layer = (
    _load_gdf(osm_flooded_railways_path, "railways_flooded") if show_flooded_infra else None
)

osm_summary = _load_summary(osm_summary_path)
if osm_summary is None and any(
    layer is not None for layer in [roads_layer, railways_layer, flooded_roads_layer, flooded_railways_layer]
):
    osm_summary = _summary_from_layers(
        roads_layer,
        railways_layer,
        flooded_roads_layer,
        flooded_railways_layer,
    )

if osm_summary is not None:
    st.subheader("Infrastructure impact")
    infra1, infra2, infra3, infra4 = st.columns(4)
    infra1.metric("Roads total", f"{osm_summary['roads_total_km']:.1f} km")
    infra2.metric("Roads flooded", f"{osm_summary['roads_flooded_km']:.1f} km")
    infra3.metric("Railways total", f"{osm_summary['railways_total_km']:.1f} km")
    infra4.metric("Railways flooded", f"{osm_summary['railways_flooded_km']:.1f} km")
    st.caption(
        "Road and railway lengths are measured in a metric projection and the flooded share is the portion intersecting the GFM flood extent for the selected scene or the whole-time max."
    )
elif show_roads or show_railways or show_flooded_infra:
    st.info("Run the OSM step to add infrastructure layers and flooded km metrics.")
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
    image=overlay.rgba,
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
            image=gfm_overlay.rgba,
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

if show_population and population_path.exists():
    population_overlay = ui.raster_overlay(
        population_path, cmap="Purples", mask_values=(0.0,), scale=1.0
    )
    folium.raster_layers.ImageOverlay(
        image=population_overlay.rgba,
        bounds=population_overlay.bounds,
        opacity=0.65,
        name="WorldPop 2020 population",
    ).add_to(fmap)

if show_roads:
    _add_line_layer(
        overlays,
        roads_layer,
        name="roads",
        color="#666666",
        weight=2,
        opacity=0.7,
    )

if show_railways:
    _add_line_layer(
        overlays,
        railways_layer,
        name="railways",
        color="#1b9e77",
        weight=2,
        opacity=0.75,
    )

if show_flooded_infra:
    _add_line_layer(
        overlays,
        flooded_roads_layer,
        name="flooded roads",
        color="#e31a1c",
        weight=4,
        opacity=0.95,
    )
    _add_line_layer(
        overlays,
        flooded_railways_layer,
        name="flooded railways",
        color="#ff7f00",
        weight=4,
        opacity=0.95,
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
    value = ui.sample_raster(selected, lat, lon, mask_values=mask_values, scale=scale)
    if value is None:
        reading, note = "—", "Point is outside the raster."
    elif math.isnan(value):
        reading, note = "NaN", "No value here — nodata or permanent water (999)."
    else:
        reading, note = f"{value:.2f} m", "Value read from the full-resolution GeoTIFF."
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
    ui.colorbar_figure(overlay.vmin, overlay.vmax, "Blues", label),
    width="content",
)
st.caption(
    "Color range is the 2nd-98th percentile of valid pixels; the map is a "
    "downsampled preview — use QGIS on the GeoTIFF for full resolution."
)
