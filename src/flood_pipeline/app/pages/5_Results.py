"""Results page: water-depth / water-level rasters on an interactive map.

FLEXTH conventions: WD_*.tif is uint16 in centimeters (displayed x0.01 as
meters), WL_*.tif is float32 meters; 0 is nodata and 999 marks permanent
water — both are masked out.
"""

import folium
import streamlit as st
from streamlit_folium import st_folium

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError
from flood_pipeline.steps import flexth_step
from flood_pipeline.steps.population import calculate_exposure

st.set_page_config(page_title="Results", page_icon="🌊", layout="wide")
st.title("Results")

try:
    cfg = ui.pipeline_cfg()
except ConfigError as e:
    st.error(str(e))
    st.stop()

outputs = flexth_step.find_outputs(cfg.output_dir)
if not outputs:
    st.info(
        f"No WD_/WL_ rasters in `{cfg.output_dir}` yet — run the pipeline first "
        "(Run page)."
    )
    st.stop()

selected = st.selectbox("Raster", outputs, format_func=lambda p: p.name)
is_depth = selected.name.startswith("WD_")
if is_depth:
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

try:
    overlay = ui.raster_overlay(selected, cmap="Blues", mask_values=mask_values, scale=scale)
except ValueError as e:
    st.error(str(e))
    st.info(
        "This output contains no displayable water values. Check that "
        "`gfm_flood_max.tif` contains nonzero flood pixels before running FLEXTH."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("min", f"{overlay.valid_min:.2f} m")
col2.metric("mean", f"{overlay.valid_mean:.2f} m")
col3.metric("max", f"{overlay.valid_max:.2f} m")
col4.metric("valid pixels", f"{overlay.valid_fraction:.1%}")
st.caption("Statistics exclude nodata (0) and the permanent-water sentinel (999).")

if population_path.exists() and is_depth:
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

fmap = folium.Map(tiles="OpenStreetMap")
folium.raster_layers.ImageOverlay(
    image=overlay.rgba,
    bounds=overlay.bounds,
    opacity=1.0,  # per-pixel alpha already encodes transparency
    name=selected.name,
).add_to(fmap)

if show_gfm and cfg.gfm_mask_path().exists():
    gfm_overlay = ui.raster_overlay(
        cfg.gfm_mask_path(), cmap="Reds", mask_values=(0.0,), scale=1.0
    )
    folium.raster_layers.ImageOverlay(
        image=gfm_overlay.rgba,
        bounds=gfm_overlay.bounds,
        opacity=0.6,
        name="GFM flood max",
    ).add_to(fmap)
elif show_gfm:
    st.warning(f"GFM mask not found: {cfg.gfm_mask_path()}")

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

if cfg.aoi_abs_path.exists():
    ui.add_aoi_layer(fmap, cfg.aoi_abs_path)
folium.LayerControl().add_to(fmap)
fmap.fit_bounds(overlay.bounds)

st_folium(fmap, key="results_map", height=600, width="stretch", returned_objects=[])

st.pyplot(
    ui.colorbar_figure(overlay.vmin, overlay.vmax, "Blues", label),
    width="content",
)
st.caption(
    "Color range is the 2nd-98th percentile of valid pixels; the map is a "
    "downsampled preview — use QGIS on the GeoTIFF for full resolution."
)
