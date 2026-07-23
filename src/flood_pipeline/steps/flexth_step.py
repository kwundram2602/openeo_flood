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

from flood_pipeline.config import GFM_SCENE_STAMP_RE, PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

FLEXTH_CONFIG_NAME = "flexth_config.yaml"


def build_flexth_config(
    cfg: PipelineConfig,
    *,
    gfm_path: Path,
    work_dir: Path,
    output_dir: Path,
) -> dict:
    """Build a FLEXTH-schema config for one scene from the unified config.

    ``gfm_path`` is that scene's flood mask; ``work_dir``/``output_dir`` are the
    scene's own dirs so FLEXTH's derived flood.tif/dtm.tif never collide between
    scenes. ``merge`` is always disabled and the ``flexth:`` subtree of the
    unified config is passed through verbatim.
    """
    passthrough = {
        key: copy.deepcopy(value)
        for key, value in cfg.flexth.items()
        if key != "enabled"
    }
    return {
        "io": {
            "dtm": str(cfg.dem_path()),
            "gfm": str(gfm_path),
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
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


def find_scene_outputs(output_dir: Path) -> dict[str, dict[str, Path]]:
    """Map each scene subfolder (its stamp) to its WD_/WL_ rasters.

    ``{stamp: {"WD": path, "WL": path}}``, sorted by stamp. Missing WD or WL
    keys are simply absent for that scene.
    """
    if not output_dir.exists():
        return {}
    scenes: dict[str, dict[str, Path]] = {}
    for sub in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        entry: dict[str, Path] = {}
        wd = sorted(sub.glob("WD_*.tif"))
        wl = sorted(sub.glob("WL_*.tif"))
        if wd:
            entry["WD"] = wd[0]
        if wl:
            entry["WL"] = wl[0]
        if entry:
            scenes[sub.name] = entry
    return scenes


def _stamp_output(path: Path, stamp: str) -> Path:
    """Rename WD_/WL_ output to embed the stamp: WD_<stamp>_<rest>.tif."""
    token, _, rest = path.name.partition("_")  # "WD", "", "method_A_...tif"
    target = path.with_name(f"{token}_{stamp}_{rest}")
    path.replace(target)
    return target


def _scene_stamp(scene_path: Path) -> str:
    """Extract the timestamp from a per-scene GFM raster filename."""
    match = GFM_SCENE_STAMP_RE.search(scene_path.name)
    assert match is not None  # gfm_scene_paths only returns matching names
    return match.group(1)


def _run_flexth(config_path: Path, log: LogFn) -> None:
    """Run ``flexth pipeline <config>`` as a subprocess, streaming its output."""
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


def run(cfg: PipelineConfig, log: LogFn = print) -> StepOutcome:
    """Run FLEXTH once per GFM scene into per-timestamp subfolders."""
    if not cfg.dem_path().exists():
        raise FileNotFoundError(
            f"FLEXTH input missing: {cfg.dem_path()} (run the dem step first)"
        )
    scenes = cfg.gfm_scene_paths()
    if not scenes:
        raise FileNotFoundError(
            f"no GFM scene rasters in {cfg.data_dir} (run the gfm step first)"
        )

    outputs: list[Path] = []
    for scene_path in scenes:
        stamp = _scene_stamp(scene_path)
        work_dir = cfg.scene_work_dir(stamp)
        output_dir = cfg.scene_output_dir(stamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = set(find_outputs(output_dir))

        config_path = write_flexth_config(
            build_flexth_config(
                cfg, gfm_path=scene_path, work_dir=work_dir, output_dir=output_dir
            ),
            work_dir,
        )
        log(f"##[scene:{stamp}] generated FLEXTH config: {config_path}")
        _run_flexth(config_path, log)

        # Prefer rasters this run created; fall back to all (identical params
        # overwrite the same filenames).
        fresh = [p for p in find_outputs(output_dir) if p not in existing]
        produced = fresh or find_outputs(output_dir)
        for raster in produced:
            stamped = _stamp_output(raster, stamp)
            outputs.append(stamped)
            log(f"##[scene:{stamp}] output: {stamped}")

    return StepOutcome(outputs=outputs)
