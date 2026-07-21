"""GFM scene browser: search the STAC catalog and set the temporal extent.

Uses the same search function as the gfm pipeline step; the EODC STAC API is
public, so this works before any Earth Engine authentication.
"""

import datetime

import pandas as pd
import streamlit as st

from flood_pipeline.app import ui
from flood_pipeline.steps import gfm

SEARCH_MAX_ITEMS = 100

st.set_page_config(page_title="GFM Scenes", page_icon="🛰️", layout="wide")
st.title("GFM scene browser")

cfg_dict, cfg_path = ui.get_cfg()
gfm_section = cfg_dict.setdefault("gfm", {})
extent = gfm_section.get("temporal_extent", ["2024-09-15", "2024-09-20"])
configured_start = datetime.date.fromisoformat(str(extent[0]))
configured_end = datetime.date.fromisoformat(str(extent[1]))

st.caption(
    f"Configured extent: {configured_start} → {configured_end}. Search a wider "
    "window, inspect the scenes, and write a new extent back to the config."
)

col1, col2, col3 = st.columns([1, 1, 1])
search_start = col1.date_input("Search from", value=configured_start - datetime.timedelta(days=30))
search_end = col2.date_input("Search to", value=configured_end + datetime.timedelta(days=30))
col3.write("")  # vertical alignment for the button
search_clicked = col3.button("Search scenes", type="primary")

if search_clicked:
    try:
        cfg = ui.pipeline_cfg()
        bbox = gfm.aoi_bbox_4326(cfg.aoi_abs_path)
        items = gfm.search_gfm_items(
            bbox,
            [search_start.isoformat(), search_end.isoformat()],
            stac_url=cfg.gfm.stac_url,
            collection=cfg.gfm.collection,
            max_items=SEARCH_MAX_ITEMS,
        )
    except Exception as e:  # deliberate: network/AOI errors become page messages
        st.error(f"Search failed: {e}")
        st.stop()
    st.session_state["gfm_scenes"] = [
        {
            "datetime": item.datetime.isoformat(sep=" ", timespec="minutes") if item.datetime else "",
            "date": item.datetime.date().isoformat() if item.datetime else "",
            "id": item.id,
        }
        for item in items
    ]

scenes: list[dict] = st.session_state.get("gfm_scenes", [])
if not scenes:
    st.info("No search results yet — pick a window and press **Search scenes**.")
    st.stop()

st.markdown(f"**{len(scenes)} scene(s) found.** Select rows to derive a date range (none = all).")
frame = pd.DataFrame(scenes)
selection = st.dataframe(
    frame,
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="multi-row",
)

selected_rows = selection.selection.rows if selection and selection.selection else []
chosen = frame.iloc[selected_rows] if selected_rows else frame
chosen_dates = sorted(d for d in chosen["date"] if d)
if not chosen_dates:
    st.warning("The chosen scenes have no datetimes; cannot derive an extent.")
    st.stop()

new_start, new_end = chosen_dates[0], chosen_dates[-1]
st.markdown(f"**Derived extent:** {new_start} → {new_end}")
if st.button("Set as gfm.temporal_extent", type="primary"):
    gfm_section["temporal_extent"] = [new_start, new_end]
    ui.save_cfg()
    st.success(f"temporal_extent = [{new_start}, {new_end}] saved to {cfg_path.name}.")
