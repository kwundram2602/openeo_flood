"""Config editor: one form per section, saved straight back to the YAML file."""

import datetime

import streamlit as st

from flood_pipeline.app import ui
from flood_pipeline.config import (
    DEM_DELIVERY_MODES,
    GFM_AGGREGATIONS,
    GFM_SELECTABLE_BANDS,
)

st.set_page_config(page_title="Config", page_icon="⚙️", layout="wide")
st.title("Configuration")

cfg_dict, cfg_path = ui.get_cfg()
st.caption(f"Editing `{cfg_path}` — saving a section rewrites the file (comments are dropped).")


def _saved(section: str) -> None:
    ui.save_cfg()
    st.toast(f"Saved {section} to {cfg_path.name}", icon="💾")


# --- project + AOI -----------------------------------------------------------
project = cfg_dict.setdefault("project", {})
aoi = cfg_dict.setdefault("aoi", {})
with st.form("project_form"):
    st.subheader("Project")
    st.caption(
        f"Project folder: `{cfg_path.parent}` — all paths below are relative to it. "
        "Use the Home page to create or open another project."
    )
    name = st.text_input("Project name", value=project.get("name", "flood_run"))
    gee_project = st.text_input(
        "Google Earth Engine project", value=project.get("gee_project", "")
    )
    col1, col2, col3 = st.columns(3)
    data_dir = col1.text_input("Data dir", value=project.get("data_dir", "flood_data"))
    work_dir = col2.text_input("Work dir", value=project.get("work_dir", "preprocessed"))
    output_dir = col3.text_input("Output dir", value=project.get("output_dir", "wl_wd_out"))
    aoi_path = st.text_input("AOI file", value=aoi.get("path", "aoi_drawn.geojson"))
    if st.form_submit_button("Save project & AOI"):
        project.update(
            name=name, gee_project=gee_project,
            data_dir=data_dir, work_dir=work_dir, output_dir=output_dir,
        )
        aoi["path"] = aoi_path
        _saved("project")

# --- dem ----------------------------------------------------------------------
dem = cfg_dict.setdefault("dem", {})
with st.form("dem_form"):
    st.subheader("DEM (FABDEM via Google Earth Engine)")
    dem_enabled = st.checkbox("Enabled", value=dem.get("enabled", True))
    col1, col2, col3 = st.columns(3)
    delivery = col1.selectbox(
        "Delivery",
        DEM_DELIVERY_MODES,
        index=DEM_DELIVERY_MODES.index(dem.get("delivery", "local")),
        help="local = direct download; drive = Drive export, pipeline halts for manual download",
    )
    scale = col2.number_input("Scale [m]", value=int(dem.get("scale", 30)), min_value=1, step=10)
    dem_crs = col3.text_input("CRS", value=dem.get("crs", "EPSG:4326"))
    col4, col5 = st.columns(2)
    dem_out = col4.text_input("Output name", value=dem.get("out_name", "fabdem.tif"))
    overwrite = col5.checkbox(
        "Overwrite existing", value=dem.get("overwrite", False),
        help="off = skip the download when the output file already exists",
    )
    col6, col7 = st.columns(2)
    drive_folder = col6.text_input("Drive folder", value=dem.get("drive_folder", "fabdem_exports"))
    drive_prefix = col7.text_input("Drive prefix", value=dem.get("drive_prefix", "fabdem"))
    if st.form_submit_button("Save dem"):
        dem.update(
            enabled=dem_enabled, delivery=delivery, scale=int(scale), crs=dem_crs,
            out_name=dem_out, overwrite=overwrite,
            drive_folder=drive_folder, drive_prefix=drive_prefix,
        )
        _saved("dem")

# --- gfm ----------------------------------------------------------------------
gfm = cfg_dict.setdefault("gfm", {})
with st.form("gfm_form"):
    st.subheader("Flood extent (GFM via EODC STAC)")
    gfm_enabled = st.checkbox("Enabled", value=gfm.get("enabled", True))
    stac_url = st.text_input("STAC URL", value=gfm.get("stac_url", "https://stac.eodc.eu/api/v1"))
    col1, col2 = st.columns(2)
    collection = col1.text_input("Collection", value=gfm.get("collection", "GFM"))
    band_options = list(GFM_SELECTABLE_BANDS)
    current_band = gfm.get("band", "ensemble_flood_extent")
    if current_band not in band_options:
        band_options = [current_band, *band_options]  # keep a custom value
    band = col2.selectbox(
        "Band", band_options, index=band_options.index(current_band),
        help="*_flood_extent are binary; ensemble_likelihood is thresholded",
    )
    extent = gfm.get("temporal_extent", ["2024-09-15", "2024-09-20"])
    dates = st.date_input(
        "Temporal extent",
        value=(
            datetime.date.fromisoformat(str(extent[0])),
            datetime.date.fromisoformat(str(extent[1])),
        ),
        help="Pick the start and end date; the GFM Scenes page can set this from a search",
    )
    col3, col4, col5, col6 = st.columns(4)
    resolution = col3.number_input(
        "Resolution [deg]", value=float(gfm.get("resolution", 0.0003)),
        min_value=0.00001, step=0.0001, format="%.5f",
    )
    max_items = col4.number_input(
        "Max items", value=int(gfm.get("max_items", 0)), min_value=0,
        help="safety cap on the number of scenes loaded; 0 = keep every scene in the window",
    )
    aggregation = col5.selectbox(
        "Aggregation",
        GFM_AGGREGATIONS,
        index=GFM_AGGREGATIONS.index(gfm.get("aggregation", "max")),
        help="one raster is written per scene; the whole-time max is always written, sum/both add the sum raster",
    )
    min_area_ha = col6.number_input(
        "Min flood area [ha]", value=float(gfm.get("min_area_ha", 1.0)),
        min_value=0.0, step=0.5,
        help="smallest connected flood area kept as a polygon; 0 = keep every speck",
    )
    likelihood_threshold = st.number_input(
        "Likelihood threshold [%]",
        value=int(gfm.get("likelihood_threshold", 25)),
        min_value=1, max_value=100, step=5,
        help="only used when the band is a *_likelihood band",
    )
    compare_algorithms = st.checkbox(
        "Compare algorithms (ensemble, dlr, tuw, list)",
        value=gfm.get("compare_algorithms", False),
        help="Run flood extent + water depth for all four GFM algorithms instead of "
        "the single Band above (4× the FLEXTH runs); the Results page then shows a "
        "band dropdown",
    )
    if st.form_submit_button("Save gfm"):
        if len(dates) != 2:
            st.error("Pick both a start and an end date.")
        else:
            gfm.update(
                enabled=gfm_enabled, stac_url=stac_url, collection=collection, band=band,
                temporal_extent=[dates[0].isoformat(), dates[1].isoformat()],
                resolution=float(resolution), max_items=int(max_items),
                aggregation=aggregation, min_area_ha=float(min_area_ha),
                compare_algorithms=compare_algorithms,
                likelihood_threshold=int(likelihood_threshold),
            )
            _saved("gfm")

# --- flexth ---------------------------------------------------------------
flexth = cfg_dict.setdefault("flexth", {})
resample = flexth.setdefault("resample", {})
prepare_dtm = flexth.setdefault("prepare_dtm", {})
flood_processing = flexth.setdefault("flood_processing", {})
params = flood_processing.setdefault("params", {})

with st.form("flexth_form"):
    st.subheader("FLEXTH (water depth)")
    flexth_enabled = st.checkbox("Enabled", value=flexth.get("enabled", True))
    fill_excluded = st.checkbox(
        "Fill excluded/urban areas (feed FLEXTH the exclusion + permanent-water masks)",
        value=flexth.get("fill_excluded", False),
        help="FLEXTH interpolates a water surface into the urban zones GFM could "
        "not observe; the Results page can overlay what it added",
    )

    st.markdown("**Resample** — GFM mask → metric master grid (`flood.tif`)")
    col1, col2, col3, col4 = st.columns(4)
    resample_enabled = col1.checkbox("resample enabled", value=resample.get("enabled", True))
    resample_crs = col2.text_input("Target CRS", value=resample.get("crs", "EPSG:32633"))
    resample_res = resample.get("resolution", [30, 30])
    res_x = col3.number_input("Resolution x [m]", value=float(resample_res[0]), min_value=1.0)
    res_y = col4.number_input("Resolution y [m]", value=float(resample_res[1]), min_value=1.0)
    col5, col6 = st.columns(2)
    resample_alg_options = ("near", "mode", "bilinear")
    current_alg = resample.get("resample_alg", "near")
    resample_alg = col5.selectbox(
        "Resample algorithm", resample_alg_options,
        index=resample_alg_options.index(current_alg) if current_alg in resample_alg_options else 0,
    )
    resample_compression = col6.text_input("Compression", value=resample.get("compression", "LZW"))

    st.markdown("**Prepare DTM** — DEM → `flood.tif` grid (`dtm.tif`)")
    col1, col2, col3 = st.columns(3)
    prepare_enabled = col1.checkbox("prepare_dtm enabled", value=prepare_dtm.get("enabled", True))
    method_options = ("rasterio_gdal", "gdal_only")
    current_method = prepare_dtm.get("method", "rasterio_gdal")
    prepare_method = col2.selectbox(
        "Method", method_options,
        index=method_options.index(current_method) if current_method in method_options else 0,
    )
    continuous_input = col3.checkbox(
        "Continuous input (DTM → bilinear)", value=prepare_dtm.get("continuous_input", True)
    )
    prepare_compression = st.text_input(
        "Compression ", value=prepare_dtm.get("compression", "ZSTD")
    )

    st.markdown("**Flood processing**")
    col1, col2, col3 = st.columns(3)
    processing_enabled = col1.checkbox(
        "flood_processing enabled", value=flood_processing.get("enabled", True)
    )
    output_map_options = ("WL_WD", "WD", "WL")
    current_map = flood_processing.get("output_map", "WL_WD")
    output_map = col2.selectbox(
        "Output map", output_map_options,
        index=output_map_options.index(current_map) if current_map in output_map_options else 0,
    )
    method_a_b = ("method_A", "method_B")
    current_wl = flood_processing.get("wl_estimation_method", "method_A")
    wl_method = col3.selectbox(
        "WL estimation method", method_a_b,
        index=method_a_b.index(current_wl) if current_wl in method_a_b else 0,
    )

    st.markdown("**Algorithm parameters** (see FLEXTH docs)")
    new_params: dict[str, float | int] = {}
    param_columns = st.columns(3)
    for position, (key, value) in enumerate(sorted(params.items())):
        column = param_columns[position % 3]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            column.text_input(key, value=str(value), disabled=True)
            new_params[key] = value
        elif isinstance(value, int):
            new_params[key] = int(column.number_input(key, value=int(value), step=1))
        else:
            new_params[key] = float(column.number_input(key, value=float(value), format="%.4f"))

    if st.form_submit_button("Save flexth"):
        flexth["enabled"] = flexth_enabled
        flexth["fill_excluded"] = fill_excluded
        resample.update(
            enabled=resample_enabled, crs=resample_crs,
            resolution=[int(res_x), int(res_y)],
            resample_alg=resample_alg, compression=resample_compression,
        )
        prepare_dtm.update(
            enabled=prepare_enabled, method=prepare_method,
            continuous_input=continuous_input, compression=prepare_compression,
        )
        flood_processing.update(
            enabled=processing_enabled, output_map=output_map, wl_estimation_method=wl_method
        )
        params.update(new_params)
        _saved("flexth")
