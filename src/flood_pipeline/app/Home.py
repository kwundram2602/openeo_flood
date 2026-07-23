"""Dashboard home: open or create a project, show validation status and summary.

A project is a folder holding its own config.yaml plus all data; projects are
discovered under ./projects/ but any config path can be opened.
"""

from pathlib import Path

import streamlit as st
import yaml

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError, load_config, validate
from flood_pipeline.steps import flexth_step

st.set_page_config(page_title="Flood Pipeline", page_icon="🌊", layout="wide")
st.title("Flood Monitor & Water Depth Pipeline")
st.caption(
    "FABDEM DEM (Google Earth Engine) + GFM flood extent (EODC STAC) + "
    "FLEXTH water depth — one config per project, one pipeline."
)

discovered = ui.discover_project_configs()

if "cfg_path" not in st.session_state:
    default = ui.default_config_path()
    if default is not None and default.is_file():
        ui.load_cfg(default)

current_path: Path | None = st.session_state.get("cfg_path")

open_column, create_column = st.columns(2)

with open_column:
    st.subheader("Open project")
    if discovered:
        index = discovered.index(current_path) if current_path in discovered else 0
        chosen = st.selectbox(
            "Projects in ./projects/",
            discovered,
            index=index,
            format_func=lambda p: p.parent.name,
        )
        if st.button("Open selected"):
            ui.load_cfg(chosen)
            st.rerun()
    else:
        st.info("No projects found under ./projects/ — create one on the right.")
    with st.expander("Open a config by path"):
        path_text = st.text_input("Config file", value=str(current_path or ""))
        if st.button("Open path"):
            config_path = Path(path_text)
            if not config_path.is_file():
                st.error(f"Not a file: {config_path}")
            else:
                try:
                    ui.load_cfg(config_path)
                    st.rerun()
                except yaml.YAMLError as e:
                    st.error(f"Not valid YAML: {e}")

with create_column:
    st.subheader("Create new project")
    with st.form("new_project"):
        new_name = st.text_input("Project name", placeholder="e.g. poland_2024")
        parent_text = st.text_input(
            "Parent folder",
            value=ui.PROJECTS_DIR_NAME,
            help="Relative to the current working directory, or absolute — "
            "projects can live anywhere.",
        )
        st.caption(
            "The new project copies the currently loaded config (GEE project, "
            "dates, FLEXTH parameters); draw its AOI on the AOI page afterwards."
        )
        if st.form_submit_button("Create & open"):
            if not new_name.strip():
                st.error("Give the project a name.")
            else:
                template = st.session_state.get("cfg_dict")
                try:
                    created = ui.create_project(
                        new_name.strip(), Path(parent_text), template
                    )
                except FileExistsError as e:
                    st.error(str(e))
                else:
                    ui.load_cfg(created)
                    st.rerun()

if "cfg_path" not in st.session_state:
    st.stop()

cfg_dict, cfg_path = ui.get_cfg()
st.success(f"Project: **{cfg_path.parent.name}** — `{cfg_path}`")

try:
    cfg = load_config(cfg_path)
except ConfigError as e:
    st.error(str(e))
    st.stop()

errors = validate(cfg)
if errors:
    st.warning("**Config problems:**\n" + "\n".join(f"- {e}" for e in errors))
else:
    st.info("Config is valid — head to **Run** to start the pipeline.")

left, middle, right = st.columns(3)
with left:
    st.subheader("Inputs")
    st.markdown(
        f"**AOI:** `{cfg.aoi_path}`\n\n"
        f"**GFM dates:** {cfg.gfm.temporal_extent[0]} → {cfg.gfm.temporal_extent[1]}\n\n"
        f"**DEM delivery:** {cfg.dem.delivery}"
    )
with middle:
    st.subheader("Steps enabled")
    st.markdown(
        f"- dem: {'✅' if cfg.dem.enabled else '⏸️'}\n"
        f"- gfm: {'✅' if cfg.gfm.enabled else '⏸️'}\n"
        f"- flexth: {'✅' if cfg.flexth.get('enabled', True) else '⏸️'}"
    )
with right:
    st.subheader("Data status")
    dem_state = "✅" if cfg.dem_path().exists() else "—"
    gfm_state = "✅" if cfg.gfm_mask_path().exists() else "—"
    st.markdown(
        f"- DEM `{cfg.dem.out_name}`: {dem_state}\n"
        f"- GFM `{cfg.gfm_mask_path().name}`: {gfm_state}\n"
        f"- Water-depth scenes: {len(flexth_step.find_scene_outputs(cfg.output_dir))}"
    )

st.divider()
st.markdown(
    "**Pages** — *Config*: edit all parameters · *AOI*: draw a new area on the map · "
    "*GFM Scenes*: browse available flood scenes and set the date range · "
    "*Run*: execute the pipeline with live logs · *Results*: water-depth map."
)
