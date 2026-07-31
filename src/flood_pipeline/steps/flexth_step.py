"""generate FLEXTH's config and run its pipeline CLI.

resample -> prepare-dtm -> run. This
module runs it as a subprocess so its output can be streamed line by line.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from flexth import resample as flexth_resample
from flexth.config import DTM_FILENAME, ResampleConfig

from flood_pipeline.config import SCENE_DIR_RE, PipelineConfig
from flood_pipeline.steps import LogFn, StepOutcome

FLEXTH_CONFIG_NAME = "flexth_config.yaml"


def prepared_dtm_path(cfg: PipelineConfig) -> Path:
    """The DEM warped onto the FLEXTH grid, shared by every scene.
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

    ``gfm_path`` is that scene's flood mask.

    ``prepare_dtm=False`` switches FLEXTH's prepare-dtm step off for this scene:
    the caller has already linked the shared prepared DTM into ``work_dir``. A
    user who disabled ``flexth.prepare_dtm`` keeps it disabled either way.
    """
    passthrough = {
        key: copy.deepcopy(value)
        for key, value in cfg.flexth.items()
        if key not in ("enabled", "fill_excluded")
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
    token, _, rest = path.name.partition("_")
    target = path.with_name(f"{token}_{stamp}_{rest}")
    # Windows file-lock workaround: retry a few times with a tiny delay
    for attempt in range(5):
        try:
            path.replace(target)
            break
        except PermissionError as e:
            if attempt == 4:
                raise e
            time.sleep(0.5)
    return target


def _clear_orphan_scene_dirs(
    cfg: PipelineConfig, band: str, stamps: set[str], log: LogFn
) -> None:
    """Delete a band's scene folders whose GFM scene this run no longer has.

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
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _warp_to_grid(cfg: PipelineConfig, input_raster: Path, output_raster: Path) -> None:
    """Warp a categorical raster onto the FLEXTH master grid.

    Goes through FLEXTH's own ``resample`` function (the same one the
    pipeline itself uses to build ``flood.tif``), with the same CRS/
    resolution/algorithm from ``cfg.flexth.resample`` and nearest-neighbour
    resampling (the masks are categorical, not continuous).

    This intentionally does NOT warp onto ``cfg.dem_path()``'s raw grid via a
    hand-rolled ``rasterio.warp.reproject`` call. Doing so used to produce a
    grid with different pixel dimensions than the ``flood.tif`` FLEXTH itself
    resamples from the same GFM cube, because the DTM's native grid and
    FLEXTH's resample-step grid aren't guaranteed to agree. That mismatch
    surfaces deep inside FLEXTH's ``flood_processing.run`` as a numpy
    broadcast ``ValueError`` between the exclusion mask and the flood mask.
    Routing every mask through the exact function/config FLEXTH uses for
    ``flood.tif`` keeps them grid-identical by construction.
    """
    resample = cfg.flexth.get("resample", {})
    flexth_resample.run(
        ResampleConfig(
            input_raster=input_raster,
            output_raster=output_raster,
            crs=resample.get("crs", "EPSG:32633"),
            resolution=list(resample.get("resolution", [30, 30])),
            resample_alg="near",
            compression=resample.get("compression", "LZW"),
        )
    )


def _feed_masks(
        cfg: PipelineConfig, band: str, stamp: str, work_dir: Path, log: LogFn
) -> None:
    """Write exclusion.tif and permanent_water.tif into a scene work dir."""
    # 1. Handle permanent water
    perm_water_src = cfg.gfm_reference_water_path(band)
    if perm_water_src.exists():
        _warp_to_grid(cfg, perm_water_src, work_dir / "permanent_water.tif")
        log(f"##[scene:{band}/{stamp}] fill: wrote permanent_water.tif")
    else:
        log(f"##[scene:{band}/{stamp}] fill: permanent_water.tif absent, skipping")

    # 2. Handle GFM exclusion mask
    excl_src = cfg.gfm_exclusion_path(band, stamp)
    excl_out = work_dir / "exclusion.tif"

    if excl_src.exists():
        _warp_to_grid(cfg, excl_src, excl_out)
        log(f"##[scene:{band}/{stamp}] fill: wrote exclusion.tif")
    else:
        log(f"##[scene:{band}/{stamp}] fill: exclusion.tif absent from GFM")


def _write_fill_raster(
    cfg: PipelineConfig, band: str, stamp: str, wd_path: Path, log: LogFn
) -> Path | None:
    """Write a mask of pixels FLEXTH flooded beyond the raw GFM extent.

    ``added = wet & (raw GFM extent != flood)`` on the shared FLEXTH grid, where
    ``wet`` excludes nodata (0) and the permanent-water sentinel (999). Returns
    ``None`` (no-op) when the WD raster or the resampled flood.tif is absent.
    """
    flood_path = cfg.scene_work_dir(band, stamp) / "flood.tif"
    if not wd_path.exists() or not flood_path.exists():
        return None
    try:
        with rasterio.open(wd_path) as wd_src:
            wd = wd_src.read(1)
            profile = wd_src.profile
        with rasterio.open(flood_path) as flood_src:
            flood = flood_src.read(1)
    except rasterio.RasterioIOError:
        return None
    wet = (wd != 0) & (wd != 999)
    added = (wet & (flood != 1)).astype("uint8")
    out_path = cfg.scene_fill_path(band, stamp)
    profile.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(added, 1)
    log(f"##[scene:{band}/{stamp}] wrote {out_path.name}")
    return out_path


def _mask_swath_extrapolation(
    cfg: PipelineConfig, band: str, stamp: str, raster_path: Path, log: LogFn
) -> None:
    """Trim FLEXTH output to the valid data footprint of the raw satellite swath.

    FLEXTH's interpolation can blindly extrapolate water depth into areas that
    were completely unobserved by the satellite (nodata in the raw GFM input).
    This crops the result back to the observed swath.
    """
    gfm_path = cfg.gfm_scene_path(band, stamp)
    if not gfm_path.exists():
        return

    work_dir = cfg.scene_work_dir(band, stamp)
    swath_mask_path = work_dir / "swath_mask.tif"

    # 1. Warp the raw GFM raster onto the FLEXTH grid (same warp path as the
    #    other masks, so this lands on the same grid as raster_path).
    if not swath_mask_path.exists():
        _warp_to_grid(cfg, gfm_path, swath_mask_path)

    # 2. Mask the extrapolated output
    try:
        with rasterio.open(swath_mask_path) as src_swath:
            swath_data = src_swath.read(1)
            swath_nodata = src_swath.nodata

        # flexth_resample represents pixels it never touched as NaN in the
        # data itself; it does not tag a formal GDAL nodata value on the
        # file, so src.nodata is always None here and checking it alone
        # silently no-ops this entire function. Detect NaN directly instead,
        # falling back to a formal nodata tag if one is ever present (e.g.
        # if flexth_resample's behaviour differs for non-float/categorical
        # inputs).
        if np.issubdtype(swath_data.dtype, np.floating):
            outside_swath = np.isnan(swath_data)
        elif swath_nodata is not None:
            outside_swath = (swath_data == swath_nodata)
        else:
            log(
                f"##[scene:{band}/{stamp}] swath mask is non-float with no nodata "
                "value set; cannot identify unobserved pixels, skipping "
                "extrapolation crop"
            )
            return

        if not np.any(outside_swath):
            return

        with rasterio.open(raster_path) as src_raster:
            data = src_raster.read(1)
            profile = src_raster.profile

        out_nodata = profile.get("nodata", 0)

        # Apply mask to cut off bleeding triangulation artifacts
        data[outside_swath] = out_nodata

        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(data, 1)

        log(f"##[scene:{band}/{stamp}] cropped extrapolation outside satellite swath for {raster_path.name}")
    except Exception as e:
        log(f"##[scene:{band}/{stamp}] failed to apply swath mask to {raster_path.name}: {e}")


def _run_flexth(config_path: Path, log: LogFn) -> None:
    """Run ``flexth pipeline <config>`` as a subprocess, streaming its output."""
    command = [sys.executable, "-m", "flexth.cli", "pipeline", str(config_path)]
    log("running: " + " ".join(command))
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
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

    # --- CONFIG HASH & AUTOMATIC INVALIDATION CHECK ---
    config_dict = cfg.flexth.model_dump() if hasattr(cfg.flexth, "model_dump") else dict(cfg.flexth)
    config_string = json.dumps(config_dict, sort_keys=True, default=str)
    current_hash = hashlib.md5(config_string.encode()).hexdigest()

    hash_file_path = cfg.work_dir / ".flexth_config_hash.json"
    hash_file_path.parent.mkdir(parents=True, exist_ok=True)

    stale_detected = False
    if hash_file_path.exists():
        try:
            with open(hash_file_path, "r") as f:
                saved_data = json.load(f)
                if saved_data.get("hash") != current_hash:
                    stale_detected = True
        except Exception:
            stale_detected = True

    if stale_detected:
        log("[WARNING] Config changes detected in FLEXTH parameters! Clearing stale output and work directories...")
        for key, _band_name in bands:
            out_root = cfg.scene_output_root(key)
            if out_root.exists():
                shutil.rmtree(out_root)
                log(f"Removed stale output directory: {out_root}")
            work_root = cfg.scene_work_root(key)
            if work_root.exists():
                shutil.rmtree(work_root)
                log(f"Removed stale work directory: {work_root}")
        shared_dtm_target = prepared_dtm_path(cfg)
        shared_dtm_target.unlink(missing_ok=True)

    with open(hash_file_path, "w") as f:
        json.dump({"hash": current_hash}, f)
    # --------------------------------------------------

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
    if cfg.flexth.get("fill_excluded"):
        _feed_masks(cfg, band, stamp, work_dir, log)
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
        # --> MASK EXTRAPOLATION BEFORE RENAMING <--
        _mask_swath_extrapolation(cfg, band, stamp, raster, log)

        renamed = _stamp_output(raster, stamp)
        stamped.append(renamed)
        log(f"##[scene:{band}/{stamp}] output: {renamed}")

    wd = next((p for p in stamped if p.name.startswith("WD")), None)
    if wd is not None:
        fill = _write_fill_raster(cfg, band, stamp, wd, log)
        if fill is not None:
            stamped.append(fill)
    return stamped