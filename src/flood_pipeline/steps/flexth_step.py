"""FLEXTH water-depth step: generate FLEXTH's config and run its pipeline CLI.

FLEXTH (installed as a git dependency) has its own YAML schema and a
``pipeline`` subcommand chaining merge -> resample -> prepare-dtm -> run. This
module maps the unified config onto that schema and drives it as a subprocess
so its output can be streamed line by line.

FLEXTH is driven once per GFM scene, but prepare-dtm only runs for the first of
them: the warped DEM is grid-identical for every date, so it is cached in the
shared work dir and linked into the later scenes' work dirs.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from flexth.config import DTM_FILENAME

from flood_pipeline.config import SCENE_DIR_RE, PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

FLEXTH_CONFIG_NAME = "flexth_config.yaml"


def prepared_dtm_path(cfg: PipelineConfig) -> Path:
    """The DEM warped onto the FLEXTH grid, shared by every scene.

    All scenes come from one GFM cube (same bbox, CRS and resolution), so their
    resampled flood.tif grids are identical and the warped DEM is too. It is
    therefore prepared once per run and cached in the shared work dir instead of
    being re-warped for every date.
    """
    return cfg.work_dir / DTM_FILENAME


def build_flexth_config(
    cfg: PipelineConfig,
    *,
    gfm_path: Path,
    work_dir: Path,
    output_dir: Path,
    prepare_dtm: bool = True,
) -> dict:
    """Build a FLEXTH-schema config for one scene from the unified config.

    ``gfm_path`` is that scene's flood mask; ``work_dir``/``output_dir`` are the
    scene's own dirs so FLEXTH's derived flood.tif/dtm.tif never collide between
    scenes. ``merge`` is always disabled and the ``flexth:`` subtree of the
    unified config is passed through verbatim.

    ``prepare_dtm=False`` switches FLEXTH's prepare-dtm step off for this scene:
    the caller has already linked the shared prepared DTM into ``work_dir``. A
    user who disabled ``flexth.prepare_dtm`` keeps it disabled either way.
    """
    passthrough = {
        key: copy.deepcopy(value)
        for key, value in cfg.flexth.items()
        if key != "enabled"
    }
    prepare_section = passthrough.get("prepare_dtm") or {}
    prepare_section["enabled"] = prepare_dtm and prepare_section.get("enabled", True)
    passthrough["prepare_dtm"] = prepare_section
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


def _clear_orphan_scene_dirs(
    cfg: PipelineConfig, band: str, stamps: set[str], log: LogFn
) -> None:
    """Delete a band's scene folders whose GFM scene this run no longer has.

    The gfm step drops the per-scene folders a re-run does not reproduce (a
    narrowed time window, a redrawn AOI, a scene that turned out empty), so
    without this their WD/WL folders outlive their input: the dashboard keeps
    listing the scene, on a grid that no longer matches, with no mask and no
    flood-area polygons behind it. Only folders named exactly like a stamp are
    touched — anything else in the band's work/output dir is left alone.
    """
    for parent in (cfg.scene_work_root(band), cfg.scene_output_root(band)):
        if not parent.exists():
            continue
        for scene_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
            if scene_dir.name in stamps or not SCENE_DIR_RE.match(scene_dir.name):
                continue
            shutil.rmtree(scene_dir)
            log(f"removed stale scene folder (no GFM scene): {scene_dir}")


def _link_or_copy(source: Path, target: Path) -> None:
    """Make ``target`` a second name for ``source``, hardlinking when possible.

    Scene work dirs live inside the shared work dir (same filesystem), so the
    hardlink normally succeeds and the prepared DTM exists only once on disk;
    the copy is the fallback for filesystems without hardlinks.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


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
    """Run FLEXTH once per GFM scene, for every resolved band.

    The DEM is warped onto the flood grid only once per run (grid-identical
    across bands and scenes); the result is cached as ``work_dir/dtm.tif`` and
    linked into each scene's work dir (see :func:`prepared_dtm_path`).
    """
    if not cfg.dem_path().exists():
        raise FileNotFoundError(
            f"FLEXTH input missing: {cfg.dem_path()} (run the dem step first)"
        )
    bands = cfg.resolved_bands()
    if not any(cfg.gfm_scene_stamps(key) for key, _ in bands):
        raise FileNotFoundError(
            f"no GFM scene folders under {cfg.data_dir} (run the gfm step first)"
        )

    # Warp the DEM once per run: the first scene prepares it, the rest reuse it.
    # Dropping a cached one keeps it in sync with the current AOI/resample grid.
    shared_dtm = prepared_dtm_path(cfg)
    shared_dtm.parent.mkdir(parents=True, exist_ok=True)
    shared_dtm.unlink(missing_ok=True)

    outputs: list[Path] = []
    for key, _band_name in bands:
        stamps = cfg.gfm_scene_stamps(key)
        _clear_orphan_scene_dirs(cfg, key, set(stamps), log)
        for stamp in stamps:
            outputs.extend(_run_scene(cfg, key, stamp, shared_dtm, log))
    return StepOutcome(outputs=outputs)


def _run_scene(
    cfg: PipelineConfig, band: str, stamp: str, shared_dtm: Path, log: LogFn
) -> list[Path]:
    """Run FLEXTH for one band/scene and return its stamped WD/WL outputs."""
    work_dir = cfg.scene_work_dir(band, stamp)
    output_dir = cfg.scene_output_dir(band, stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    existing = set(find_outputs(output_dir))

    scene_dtm = work_dir / DTM_FILENAME
    prepare_dtm = not shared_dtm.exists()
    if prepare_dtm:
        # Never warp into a leftover hardlink; that would edit other names.
        scene_dtm.unlink(missing_ok=True)
    else:
        _link_or_copy(shared_dtm, scene_dtm)
        log(f"##[scene:{band}/{stamp}] reusing prepared DTM: {shared_dtm}")

    config_path = write_flexth_config(
        build_flexth_config(
            cfg,
            gfm_path=cfg.gfm_scene_path(band, stamp),
            work_dir=work_dir,
            output_dir=output_dir,
            prepare_dtm=prepare_dtm,
        ),
        work_dir,
    )
    log(f"##[scene:{band}/{stamp}] generated FLEXTH config: {config_path}")
    _run_flexth(config_path, log)

    if prepare_dtm and scene_dtm.exists():
        _link_or_copy(scene_dtm, shared_dtm)
        log(f"##[scene:{band}/{stamp}] cached prepared DTM: {shared_dtm}")

    # Prefer rasters this run created; fall back to all (identical params
    # overwrite the same filenames).
    fresh = [p for p in find_outputs(output_dir) if p not in existing]
    produced = fresh or find_outputs(output_dir)
    stamped: list[Path] = []
    for raster in produced:
        renamed = _stamp_output(raster, stamp)
        stamped.append(renamed)
        log(f"##[scene:{band}/{stamp}] output: {renamed}")
    return stamped
