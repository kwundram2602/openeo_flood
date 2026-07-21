"""FLEXTH water-depth step: generate FLEXTH's config and run its pipeline CLI.

FLEXTH (installed as a git dependency) has its own YAML schema and a
``pipeline`` subcommand chaining merge -> resample -> prepare-dtm -> run. This
module maps the unified config onto that schema and drives it as a subprocess
so its output can be streamed line by line.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from flood_pipeline.config import PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

FLEXTH_CONFIG_NAME = "flexth_config.yaml"


def build_flexth_config(cfg: PipelineConfig) -> dict:
    """Build a dict in FLEXTH's own config schema from the unified config.

    The ``io`` section wires the dem/gfm step outputs into FLEXTH (which
    derives ``work_dir/flood.tif`` and ``work_dir/dtm.tif`` from it), ``merge``
    is always disabled (single-scene pipeline), and the ``flexth:`` subtree of
    the unified config is passed through verbatim.
    """
    passthrough = {
        key: copy.deepcopy(value)
        for key, value in cfg.flexth.items()
        if key != "enabled"
    }
    return {
        "io": {
            "dtm": str(cfg.dem_path()),
            "gfm": str(cfg.gfm_mask_path()),
            "work_dir": str(cfg.work_dir),
            "output_dir": str(cfg.output_dir),
        },
        "merge": {"enabled": False},
        **passthrough,
    }


def write_flexth_config(flexth_config: dict, work_dir: Path) -> Path:
    """Persist the generated FLEXTH config into the work dir and return its path.

    Kept on disk (not a temp file) so a run can be debugged or repeated with
    ``uv run flexth pipeline <path>`` directly.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / FLEXTH_CONFIG_NAME
    path.write_text(
        yaml.safe_dump(flexth_config, sort_keys=False), encoding="utf-8"
    )
    return path


def find_outputs(output_dir: Path) -> list[Path]:
    """WD_/WL_ rasters in ``output_dir``, newest first.

    FLEXTH derives output filenames from its algorithm parameters, so globbing
    beats reconstructing the exact names.
    """
    if not output_dir.exists():
        return []
    rasters = [*output_dir.glob("WD_*.tif"), *output_dir.glob("WL_*.tif")]
    return sorted(rasters, key=lambda p: p.stat().st_mtime, reverse=True)


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Generate the FLEXTH config and run ``flexth pipeline`` as a subprocess."""
    missing = [p for p in (cfg.dem_path(), cfg.gfm_mask_path()) if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"FLEXTH inputs missing: {missing_list} (run the dem/gfm steps first)"
        )

    config_path = write_flexth_config(build_flexth_config(cfg), cfg.work_dir)
    log(f"generated FLEXTH config: {config_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = set(find_outputs(cfg.output_dir))

    command = [sys.executable, "-m", "flexth.cli", "pipeline", str(config_path)]
    log("running: " + " ".join(command))
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    ) as process:
        assert process.stdout is not None  # PIPE guarantees a stream
        for line in process.stdout:
            log(line.rstrip())
    if process.returncode != 0:
        raise RuntimeError(
            f"flexth pipeline failed with exit code {process.returncode}"
        )

    # Prefer rasters created by this run; fall back to all (an identical
    # parameter set overwrites the same filenames).
    new_outputs = [p for p in find_outputs(cfg.output_dir) if p not in existing_outputs]
    outputs = new_outputs or find_outputs(cfg.output_dir)
    for path in outputs:
        log(f"output: {path}")
    return StepOutcome(outputs=outputs)
