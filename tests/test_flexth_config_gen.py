"""Contract tests: the generated FLEXTH config must satisfy FLEXTH's own loader.

These tests feed the YAML we generate back through flexth's config getters and
assert the derived paths (work_dir/flood.tif, work_dir/dtm.tif) and parameters
land where flood_processing expects them. They need no network and no rasters.
"""

import time
from pathlib import Path

import pytest
import yaml
from flexth import config as flexth_config

from flood_pipeline.config import load_config
from flood_pipeline.steps import flexth_step

FULL_CONFIG = {
    "project": {
        "name": "contract_test",
        "gee_project": "ee-test",
        "data_dir": "flood_data",
        "work_dir": "preprocessed",
        "output_dir": "wl_wd_out",
    },
    "aoi": {"path": "aoi.geojson"},
    "dem": {"out_name": "fabdem.tif"},
    "gfm": {"temporal_extent": ["2024-09-15", "2024-09-20"]},
    "flexth": {
        "enabled": True,
        "resample": {
            "enabled": True,
            "crs": "EPSG:32633",
            "resolution": [30, 30],
            "resample_alg": "near",
            "compression": "LZW",
        },
        "prepare_dtm": {
            "enabled": True,
            "method": "rasterio_gdal",
            "continuous_input": True,
            "compression": "ZSTD",
        },
        "flood_processing": {
            "enabled": True,
            "output_map": "WL_WD",
            "wl_estimation_method": "method_A",
            "params": {"threshold_slope": 0.07, "max_number_neighbors": 42},
        },
    },
}


@pytest.fixture
def cfg(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(FULL_CONFIG), encoding="utf-8")
    return load_config(path)


SCENE_STAMP = "2024-09-16_051230"
BAND = "ensemble"


@pytest.fixture
def scene(cfg):
    return {
        "gfm_path": cfg.gfm_scene_path(BAND, SCENE_STAMP),
        "work_dir": cfg.scene_work_dir(BAND, SCENE_STAMP),
        "output_dir": cfg.scene_output_dir(BAND, SCENE_STAMP),
    }


@pytest.fixture
def flexth_raw(cfg, scene) -> dict:
    """The generated per-scene FLEXTH config, written out and re-read by flexth."""
    generated = flexth_step.build_flexth_config(cfg, **scene)
    written = flexth_step.write_flexth_config(generated, scene["work_dir"])
    return flexth_config.load_config(written)


def test_io_section_wires_step_outputs(cfg, scene) -> None:
    generated = flexth_step.build_flexth_config(cfg, **scene)
    assert generated["io"]["dtm"] == str(cfg.dem_path())
    assert generated["io"]["gfm"] == str(cfg.gfm_scene_path(BAND, SCENE_STAMP))
    assert generated["io"]["work_dir"] == str(cfg.scene_work_dir(BAND, SCENE_STAMP))
    assert generated["io"]["output_dir"] == str(cfg.scene_output_dir(BAND, SCENE_STAMP))
    assert generated["merge"] == {"enabled": False}
    assert "enabled" not in generated  # unified-config marker must not leak


def test_passthrough_is_a_deep_copy(cfg, scene) -> None:
    generated = flexth_step.build_flexth_config(cfg, **scene)
    generated["resample"]["crs"] = "EPSG:9999"
    assert cfg.flexth["resample"]["crs"] == "EPSG:32633"


def test_resample_derives_flood_tif(cfg, flexth_raw: dict) -> None:
    resample = flexth_config.get_resample_config(flexth_raw)
    assert resample.enabled is True
    assert resample.input_raster == cfg.gfm_scene_path(BAND, SCENE_STAMP)
    assert resample.output_raster == cfg.scene_work_dir(BAND, SCENE_STAMP) / "flood.tif"
    assert resample.crs == "EPSG:32633"
    assert resample.resolution == [30, 30]
    assert resample.resample_alg == "near"


def test_prepare_dtm_derives_dtm_tif(cfg, flexth_raw: dict) -> None:
    prepare = flexth_config.get_prepare_dtm_config(flexth_raw)
    assert prepare.enabled is True
    assert prepare.input_raster == cfg.dem_path()
    assert prepare.output_raster == cfg.scene_work_dir(BAND, SCENE_STAMP) / "dtm.tif"
    assert prepare.flood_reference == cfg.scene_work_dir(BAND, SCENE_STAMP) / "flood.tif"
    assert prepare.continuous_input is True


def test_prepare_dtm_can_be_switched_off_per_scene(cfg, scene) -> None:
    generated = flexth_step.build_flexth_config(cfg, **scene, prepare_dtm=False)
    assert generated["prepare_dtm"]["enabled"] is False
    # the rest of the user's prepare_dtm settings survive
    assert generated["prepare_dtm"]["method"] == "rasterio_gdal"
    assert cfg.flexth["prepare_dtm"]["enabled"] is True  # source config untouched


def test_prepare_dtm_stays_off_when_the_user_disabled_it(cfg, scene) -> None:
    cfg.flexth["prepare_dtm"]["enabled"] = False
    generated = flexth_step.build_flexth_config(cfg, **scene, prepare_dtm=True)
    assert generated["prepare_dtm"]["enabled"] is False


def test_prepared_dtm_lives_in_the_shared_work_dir(cfg) -> None:
    assert flexth_step.prepared_dtm_path(cfg) == cfg.work_dir / "dtm.tif"


def test_flood_processing_dirs_and_params(cfg, flexth_raw: dict) -> None:
    processing = flexth_config.get_flood_processing_config(flexth_raw)
    assert processing.enabled is True
    assert processing.input_dir == cfg.scene_work_dir(BAND, SCENE_STAMP)
    assert processing.output_dir == cfg.scene_output_dir(BAND, SCENE_STAMP)
    assert processing.output_map == "WL_WD"
    assert processing.wl_estimation_method == "method_A"
    assert processing.params.threshold_slope == 0.07
    assert processing.params.max_number_neighbors == 42
    assert processing.params.connectivity == 8  # flexth default fills the gap


def test_merge_stays_disabled(flexth_raw: dict) -> None:
    assert flexth_config.get_merge_config(flexth_raw).enabled is False


def test_generated_config_written_into_work_dir(cfg, scene) -> None:
    written = flexth_step.write_flexth_config(
        flexth_step.build_flexth_config(cfg, **scene), scene["work_dir"]
    )
    assert written == scene["work_dir"] / flexth_step.FLEXTH_CONFIG_NAME
    assert written.exists()


def test_find_outputs_newest_first(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    older = out_dir / "WD_method_A_old.tif"
    newer = out_dir / "WL_method_A_new.tif"
    unrelated = out_dir / "notes.txt"
    older.write_bytes(b"")
    unrelated.write_text("")
    time.sleep(0.01)  # ensure distinct mtimes
    newer.write_bytes(b"")
    assert flexth_step.find_outputs(out_dir) == [newer, older]
    assert flexth_step.find_outputs(tmp_path / "absent") == []


def test_run_requires_dem(cfg) -> None:
    with pytest.raises(FileNotFoundError, match="run the dem step first"):
        flexth_step.run(cfg, log=lambda _line: None)


def test_run_requires_scenes(cfg) -> None:
    cfg.dem_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.dem_path().write_bytes(b"")  # DEM present, but no gfm scenes
    with pytest.raises(FileNotFoundError, match="run the gfm step first"):
        flexth_step.run(cfg, log=lambda _line: None)


def test_find_scene_outputs_maps_stamp_to_wd_wl(tmp_path: Path) -> None:
    out_dir = tmp_path / "wl_wd_out"
    scene = out_dir / "2024-09-16_051230"
    scene.mkdir(parents=True)
    wd = scene / "WD_2024-09-16_051230_method_A.tif"
    wl = scene / "WL_2024-09-16_051230_method_A.tif"
    wd.write_bytes(b"")
    wl.write_bytes(b"")

    result = flexth_step.find_scene_outputs(out_dir)
    assert result == {"2024-09-16_051230": {"WD": wd, "WL": wl}}
    assert flexth_step.find_scene_outputs(tmp_path / "absent") == {}


STAMPS = ["2024-09-16_051230", "2024-09-17_051230"]


@pytest.fixture
def project(cfg):
    """A project with a DEM and two GFM scene folders (ensemble band)."""
    cfg.dem_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.dem_path().write_bytes(b"dem")
    for stamp in STAMPS:
        cfg.gfm_scene_dir("ensemble", stamp).mkdir(parents=True, exist_ok=True)
        cfg.gfm_scene_path("ensemble", stamp).write_bytes(b"scene")
    return cfg


@pytest.fixture
def fake_flexth(monkeypatch):
    """Stand in for the FLEXTH subprocess; returns the configs it was given.

    It honours ``prepare_dtm.enabled`` by writing dtm.tif and always writes one
    WD raster, so the step sees an output for every scene.
    """
    seen: list[dict] = []

    def fake_run_flexth(config_path: Path, log) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        seen.append(raw)
        if raw["prepare_dtm"]["enabled"]:
            (Path(raw["io"]["work_dir"]) / "dtm.tif").write_bytes(b"prepared")
        output_dir = Path(raw["io"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "WD_method_A.tif").write_bytes(b"wd")

    monkeypatch.setattr(flexth_step, "_run_flexth", fake_run_flexth)
    return seen


@pytest.fixture
def two_scene_run(project, fake_flexth):
    """Run the step over two scenes with the fake FLEXTH in place."""
    outcome = flexth_step.run(project, log=lambda _line: None)
    return project, STAMPS, fake_flexth, outcome


ORPHAN_STAMP = "2024-08-01_120000"  # a scene from an earlier, wider run


def _orphan_dirs(cfg):
    """Scene folders left behind by a run whose GFM scene no longer exists."""
    dirs = [
        cfg.scene_work_dir("ensemble", ORPHAN_STAMP),
        cfg.scene_output_dir("ensemble", ORPHAN_STAMP),
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
        (path / "WD_old.tif").write_bytes(b"stale result")
    return dirs


def test_orphan_scene_folders_are_removed(project, fake_flexth) -> None:
    """Outputs must not outlive their GFM scene: the step clears stale folders.

    The gfm step deletes scene rasters that a re-run no longer produces; without
    this the WD/WL folders of those scenes linger and the dashboard keeps
    offering them, with no mask and no polygons behind them.
    """
    orphans = _orphan_dirs(project)
    flexth_step.run(project, log=lambda _line: None)
    assert [path.exists() for path in orphans] == [False, False]
    for stamp in STAMPS:  # the current run's folders stay
        assert project.scene_output_dir("ensemble", stamp).exists()
        assert project.scene_work_dir("ensemble", stamp).exists()


def test_removal_only_touches_scene_folders(project, fake_flexth) -> None:
    """Anything not named like a scene stamp is left alone."""
    keep_dir = project.output_dir / "notes"
    keep_dir.mkdir(parents=True)
    (keep_dir / "readme.txt").write_text("keep me", encoding="utf-8")
    loose_file = project.work_dir / "scratch.tif"
    loose_file.parent.mkdir(parents=True, exist_ok=True)
    loose_file.write_bytes(b"keep me too")

    flexth_step.run(project, log=lambda _line: None)
    assert (keep_dir / "readme.txt").read_text(encoding="utf-8") == "keep me"
    assert loose_file.exists()


def test_dtm_is_prepared_only_for_the_first_scene(two_scene_run) -> None:
    _cfg, _stamps, seen, _outcome = two_scene_run
    assert [raw["prepare_dtm"]["enabled"] for raw in seen] == [True, False]


def test_prepared_dtm_is_cached_and_linked_into_every_scene(two_scene_run) -> None:
    cfg, stamps, _seen, _outcome = two_scene_run
    shared = flexth_step.prepared_dtm_path(cfg)
    assert shared.read_bytes() == b"prepared"
    for stamp in stamps:
        scene_dtm = cfg.scene_work_dir("ensemble", stamp) / "dtm.tif"
        assert scene_dtm.read_bytes() == b"prepared"


def test_stale_prepared_dtm_is_rebuilt(two_scene_run, monkeypatch) -> None:
    """A DTM cached by an earlier run must not survive into the next one.

    It would otherwise be reused against a changed AOI or resample grid.
    """
    cfg, _stamps, _seen, _outcome = two_scene_run
    shared = flexth_step.prepared_dtm_path(cfg)
    shared.write_bytes(b"stale")

    seen: list[dict] = []

    def fake_run_flexth(config_path: Path, log) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        seen.append(raw)
        if raw["prepare_dtm"]["enabled"]:
            (Path(raw["io"]["work_dir"]) / "dtm.tif").write_bytes(b"rebuilt")

    monkeypatch.setattr(flexth_step, "_run_flexth", fake_run_flexth)
    flexth_step.run(cfg, log=lambda _line: None)
    assert seen[0]["prepare_dtm"]["enabled"] is True
    assert shared.read_bytes() == b"rebuilt"


def test_stamp_output_inserts_stamp_after_token(tmp_path: Path) -> None:
    original = tmp_path / "WD_method_A_params.tif"
    original.write_bytes(b"")
    renamed = flexth_step._stamp_output(original, "2024-09-16_051230")
    assert renamed.name == "WD_2024-09-16_051230_method_A_params.tif"
    assert renamed.exists()
    assert not original.exists()


def test_fill_excluded_dropped_from_flexth_config(cfg, scene) -> None:
    cfg.flexth["fill_excluded"] = True
    generated = flexth_step.build_flexth_config(cfg, **scene)
    assert "fill_excluded" not in generated


def test_masks_fed_to_flexth_when_flag_on(project, fake_flexth, monkeypatch) -> None:
    project.flexth["fill_excluded"] = True
    for stamp in STAMPS:
        project.gfm_exclusion_path("ensemble", stamp).write_bytes(b"excl")
    project.gfm_reference_water_path("ensemble").parent.mkdir(parents=True, exist_ok=True)
    project.gfm_reference_water_path("ensemble").write_bytes(b"ref")

    warped: list[tuple[str, str]] = []

    def fake_warp(cfg, input_raster, output_raster):
        Path(output_raster).parent.mkdir(parents=True, exist_ok=True)
        Path(output_raster).write_bytes(b"warped")
        warped.append((Path(input_raster).name, Path(output_raster).name))

    monkeypatch.setattr(flexth_step, "_warp_to_grid", fake_warp)
    flexth_step.run(project, log=lambda _line: None)

    outputs = {name for _src, name in warped}
    assert "exclusion.tif" in outputs
    assert "permanent_water.tif" in outputs
    for stamp in STAMPS:
        assert (project.scene_work_dir("ensemble", stamp) / "exclusion.tif").exists()


def _tiny_raster(path: Path, values) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.array(values, dtype="uint16")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint16", crs="EPSG:32633",
        transform=from_origin(500_000.0, 5_000_000.0, 30.0, 30.0),
    ) as dst:
        dst.write(arr, 1)


def test_write_fill_raster_marks_added_pixels(cfg) -> None:
    import rasterio

    stamp = SCENE_STAMP
    work = cfg.scene_work_dir(BAND, stamp)
    _tiny_raster(work / "flood.tif", [[1, 0], [0, 0]])  # raw GFM: one flooded cell
    wd = cfg.scene_output_dir(BAND, stamp) / "WD_x.tif"
    _tiny_raster(wd, [[1, 5], [0, 999]])  # FLEXTH wet: (0,0),(0,1); 999=perm water

    out = flexth_step._write_fill_raster(cfg, BAND, stamp, wd, log=lambda _l: None)
    assert out == cfg.scene_fill_path(BAND, stamp)
    with rasterio.open(out) as src:
        added = src.read(1)
    assert added.tolist() == [[0, 1], [0, 0]]


def test_write_fill_raster_skips_without_flood(cfg) -> None:
    stamp = SCENE_STAMP
    wd = cfg.scene_output_dir(BAND, stamp) / "WD_x.tif"
    _tiny_raster(wd, [[1, 0]])  # no flood.tif in the work dir
    assert flexth_step._write_fill_raster(cfg, BAND, stamp, wd, log=lambda _l: None) is None
