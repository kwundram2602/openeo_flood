"""AOI page: view the current AOI and draw a new one on the map.

A drawn polygon/rectangle is saved as ``aoi_drawn.geojson`` inside the project
folder and becomes ``aoi.path`` (GeoJSON instead of GPKG: human-readable and
no sqlite write locks on Windows).
"""

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from flood_pipeline.app import ui

DRAWN_AOI_NAME = "aoi_drawn.geojson"

st.set_page_config(page_title="AOI", page_icon="🗺️", layout="wide")
st.title("Area of interest")

cfg_dict, cfg_path = ui.get_cfg()
aoi_section = cfg_dict.setdefault("aoi", {})
current_aoi = cfg_path.parent / aoi_section.get("path", "aoi.gpkg")

if current_aoi.exists():
    outline = ui.aoi_outline(current_aoi)
    west, south, east, north = outline.total_bounds
    center = [(south + north) / 2, (west + east) / 2]
    st.caption(f"Current AOI: `{aoi_section.get('path')}` (red dashed outline)")
else:
    outline = None
    center = [48.3, 15.3]  # Lower Austria; matches the project's default AOI
    st.warning(f"Current AOI file not found: {current_aoi} — draw one below.")

fmap = folium.Map(location=center, zoom_start=8, tiles="OpenStreetMap")
if outline is not None:
    ui.add_aoi_layer(fmap, current_aoi)
Draw(
    export=False,
    draw_options={
        "polyline": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": True,
        "rectangle": True,
    },
    edit_options={"edit": False},
).add_to(fmap)

map_state = st_folium(
    fmap,
    key="aoi_map",
    height=550,
    width="stretch",
    returned_objects=["last_active_drawing"],
)

drawing = (map_state or {}).get("last_active_drawing")
if not drawing:
    st.info("Draw a rectangle or polygon on the map, then save it as the new AOI.")
else:
    geometry = gpd.GeoDataFrame.from_features([drawing], crs="EPSG:4326")
    west, south, east, north = (round(float(b), 5) for b in geometry.total_bounds)
    st.markdown(f"**Drawn geometry bounds:** {west}, {south} → {east}, {north}")
    if st.button("Save drawing as AOI", type="primary"):
        target = cfg_path.parent / DRAWN_AOI_NAME
        geometry.to_file(target, driver="GeoJSON")
        aoi_section["path"] = DRAWN_AOI_NAME
        ui.save_cfg()
        st.success(f"Saved {target} and set aoi.path — future runs use it.")
        st.rerun()
