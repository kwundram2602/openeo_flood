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
    "gfm": {"out_name": "gfm_flood_max.tif"},
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


@pytest.fixture
def flexth_raw(cfg, tmp_path: Path) -> dict:
    """The generated FLEXTH config, written out and re-read by flexth itself."""
    generated = flexth_step.build_flexth_config(cfg)
    written = flexth_step.write_flexth_config(generated, cfg.work_dir)
    return flexth_config.load_config(written)


def test_io_section_wires_step_outputs(cfg) -> None:
    generated = flexth_step.build_flexth_config(cfg)
    assert generated["io"]["dtm"] == str(cfg.data_dir / "fabdem.tif")
    assert generated["io"]["gfm"] == str(cfg.data_dir / "gfm_flood_max.tif")
    assert generated["merge"] == {"enabled": False}
    assert "enabled" not in generated  # unified-config marker must not leak


def test_passthrough_is_a_deep_copy(cfg) -> None:
    generated = flexth_step.build_flexth_config(cfg)
    generated["resample"]["crs"] = "EPSG:9999"
    assert cfg.flexth["resample"]["crs"] == "EPSG:32633"


def test_resample_derives_flood_tif(cfg, flexth_raw: dict) -> None:
    resample = flexth_config.get_resample_config(flexth_raw)
    assert resample.enabled is True
    assert resample.input_raster == cfg.gfm_mask_path()
    assert resample.output_raster == cfg.work_dir / "flood.tif"
    assert resample.crs == "EPSG:32633"
    assert resample.resolution == [30, 30]
    assert resample.resample_alg == "near"


def test_prepare_dtm_derives_dtm_tif(cfg, flexth_raw: dict) -> None:
    prepare = flexth_config.get_prepare_dtm_config(flexth_raw)
    assert prepare.enabled is True
    assert prepare.input_raster == cfg.dem_path()
    assert prepare.output_raster == cfg.work_dir / "dtm.tif"
    assert prepare.flood_reference == cfg.work_dir / "flood.tif"
    assert prepare.continuous_input is True


def test_flood_processing_dirs_and_params(cfg, flexth_raw: dict) -> None:
    processing = flexth_config.get_flood_processing_config(flexth_raw)
    assert processing.enabled is True
    assert processing.input_dir == cfg.work_dir
    assert processing.output_dir == cfg.output_dir
    assert processing.output_map == "WL_WD"
    assert processing.wl_estimation_method == "method_A"
    assert processing.params.threshold_slope == 0.07
    assert processing.params.max_number_neighbors == 42
    assert processing.params.connectivity == 8  # flexth default fills the gap


def test_merge_stays_disabled(flexth_raw: dict) -> None:
    assert flexth_config.get_merge_config(flexth_raw).enabled is False


def test_generated_config_written_into_work_dir(cfg) -> None:
    written = flexth_step.write_flexth_config(
        flexth_step.build_flexth_config(cfg), cfg.work_dir
    )
    assert written == cfg.work_dir / flexth_step.FLEXTH_CONFIG_NAME
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


def test_run_requires_step_inputs(cfg) -> None:
    with pytest.raises(FileNotFoundError, match="run the dem/gfm steps first"):
        flexth_step.run(cfg, log=lambda _line: None)
