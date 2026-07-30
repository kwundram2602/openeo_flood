"""Run page: pick steps, stream the pipeline's live output split per step.

The pipeline runs as a subprocess of the same CLI the terminal uses; its
``##[step:...]`` marker lines route the output into one status block per step.
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

import streamlit as st

from flood_pipeline.app import ui
from flood_pipeline.config import ConfigError, validate
from flood_pipeline.runner import STEP_ORDER

STEP_MARKER = re.compile(r"^##\[step:(?P<step>\w+)\] (?P<event>.+)$")
PIPELINE_MARKER = "##[pipeline]"
LOG_TAIL_LINES = 40

st.set_page_config(page_title="Run", page_icon="🚀", layout="wide")
st.title("Run pipeline")

cfg_dict, cfg_path = ui.get_cfg()

try:
    cfg = ui.pipeline_cfg()
    errors = validate(cfg)
except ConfigError as e:
    st.error(str(e))
    st.stop()

if errors:
    st.error("**Fix the config before running:**\n" + "\n".join(f"- {e}" for e in errors))

enabled_by_step = {
    "dem": cfg.dem.enabled,
    "gfm": cfg.gfm.enabled,
    "osm": cfg.osm.enabled,
    "flexth": bool(cfg.flexth.get("enabled", True)),
    "population": cfg.population.enabled,
    "ghsl": cfg.ghsl.enabled,
}
st.markdown("**Steps to run** (steps disabled in the config are skipped either way):")
columns = st.columns(len(STEP_ORDER))
selected_steps = [
    step
    for step, column in zip(STEP_ORDER, columns)
    if column.checkbox(step, value=enabled_by_step.get(step, True))
]

# GEE is required if running either the dem step (with local delivery) or the ghsl/population step
needs_gee = (
    ("dem" in selected_steps and enabled_by_step["dem"] and cfg.dem.delivery == "local")
    or ("population" in selected_steps and enabled_by_step["population"])
    or ("ghsl" in selected_steps and enabled_by_step["ghsl"])
)

if needs_gee:
    with st.expander("Google Earth Engine status (needed for GEE-based steps)"):
        st.caption(
            "The dem/ghsl steps need cached GEE credentials; the dashboard cannot run "
            "the interactive login itself."
        )
        if st.button("Check GEE authentication"):
            import flood_pipeline.gee as gee_module  # lazy direct import

            error_message = gee_module.try_init_gee(cfg.project.gee_project)
            if error_message:
                st.error(error_message)
            else:
                st.success("Earth Engine ready.")

run_clicked = st.button(
    "Run pipeline",
    type="primary",
    disabled=bool(errors) or not selected_steps,
)

if run_clicked:
    command = [
        sys.executable, "-m", "flood_pipeline.cli",
        "run", str(cfg_path), "--steps", ",".join(selected_steps),
    ]
    st.code(" ".join(command), language="text")

    statuses: dict[str, object] = {}
    placeholders: dict[str, object] = {}
    lines_by_step: dict[str, list[str]] = defaultdict(list)
    stray_lines: list[str] = []
    pipeline_result = ""
    current_step: str | None = None

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    ) as process:
        assert process.stdout is not None  # PIPE guarantees a stream
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            marker = STEP_MARKER.match(line)
            if marker:
                step, event = marker["step"], marker["event"]
                if event == "start":
                    statuses[step] = st.status(f"**{step}** — running", expanded=True)
                    placeholders[step] = statuses[step].empty()
                    current_step = step
                elif event == "done":
                    statuses[step].update(label=f"**{step}** — done", state="complete", expanded=False)
                    current_step = None
                elif event.startswith("failed"):
                    lines_by_step[step].append(event)
                    placeholders[step].code("\n".join(lines_by_step[step][-LOG_TAIL_LINES:]))
                    statuses[step].update(label=f"**{step}** — {event}", state="error", expanded=True)
                    current_step = None
                else:  # skipped (disabled in config)
                    st.info(f"{step}: {event}")
            elif line.startswith(PIPELINE_MARKER):
                pipeline_result = line.removeprefix(PIPELINE_MARKER).strip(" :")
            elif current_step is not None:
                lines_by_step[current_step].append(line)
                placeholders[current_step].code(
                    "\n".join(lines_by_step[current_step][-LOG_TAIL_LINES:])
                )
            else:
                stray_lines.append(line)

    if stray_lines:
        st.code("\n".join(stray_lines[-LOG_TAIL_LINES:]))

    if process.returncode == 0 and pipeline_result.startswith("finished"):
        st.success("Pipeline finished — see the **Results** page.")
    elif process.returncode == 0 and pipeline_result.startswith("halted"):
        st.warning(pipeline_result)
    else:
        st.error(pipeline_result or f"Pipeline failed (exit code {process.returncode}).")