"""Tests for flood_pipeline.config: loading, round-trip, path resolution, validation."""

from pathlib import Path

import pytest
import yaml

from flood_pipeline.config import ConfigError, load_config, save_config, to_dict, validate

MINIMAL_CONFIG = {
    "project": {"name": "test_run", "gee_project": "ee-test"},
    "aoi": {"path": "aoi.geojson"},
    "dem": {"delivery": "local"},
    "gfm": {"temporal_extent": ["2024-09-15", "2024-09-20"]},
    "flexth": {
        "enabled": True,
        "resample": {"enabled": True, "crs": "EPSG:32633"},
        "flood_processing": {"enabled": True, "params": {"threshold_slope": 0.05}},
    },
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """A loadable config file with an existing (empty) AOI file next to it."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(MINIMAL_CONFIG), encoding="utf-8")
    (tmp_path / "aoi.geojson").write_text("{}", encoding="utf-8")
    return path


def test_load_fills_defaults(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.project.name == "test_run"
    assert cfg.project.data_dir == "flood_data"  # default
    assert cfg.dem.scale == 30  # default
    assert cfg.gfm.aggregation == "max"  # default
    assert cfg.flexth["resample"]["crs"] == "EPSG:32633"


def test_everything_resolves_inside_the_project_folder(config_file: Path) -> None:
    """The project folder is the config file's directory; all paths live in it."""
    cfg = load_config(config_file)
    project_dir = config_file.parent
    assert cfg.project_dir == project_dir
    assert cfg.data_dir == project_dir / "flood_data"
    assert cfg.dem_path() == project_dir / "flood_data" / "fabdem.tif"
    assert cfg.gfm_mask_path().name == "gfm_flood_max.tif"
    assert cfg.aoi_abs_path == project_dir / "aoi.geojson"


def test_absolute_paths_kept(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    absolute = tmp_path / "elsewhere" / "data"
    cfg.project.data_dir = str(absolute)
    assert cfg.data_dir == absolute


def test_gfm_scene_path_and_paths(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.gfm_mask_path().name == "gfm_flood_max.tif"
    assert cfg.gfm_sum_path().name == "gfm_flood_sum.tif"
    assert cfg.gfm_scene_path("2024-09-16_051230").name == "gfm_flood_2024-09-16_051230.tif"

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "gfm_flood_2024-09-18_052000.tif").write_bytes(b"")
    (cfg.data_dir / "gfm_flood_2024-09-16_051230.tif").write_bytes(b"")
    cfg.gfm_mask_path().write_bytes(b"")   # must be excluded
    cfg.gfm_sum_path().write_bytes(b"")    # must be excluded
    (cfg.data_dir / "unrelated.tif").write_bytes(b"")

    names = [p.name for p in cfg.gfm_scene_paths()]
    assert names == [
        "gfm_flood_2024-09-16_051230.tif",
        "gfm_flood_2024-09-18_052000.tif",
    ]


def test_scene_dirs(config_file: Path) -> None:
    cfg = load_config(config_file)
    stamp = "2024-09-16_051230"
    assert cfg.scene_work_dir(stamp) == cfg.work_dir / stamp
    assert cfg.scene_output_dir(stamp) == cfg.output_dir / stamp


def test_round_trip_preserves_content(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    copy_path = tmp_path / "copy.yaml"
    save_config(cfg, copy_path)
    reloaded = load_config(copy_path)
    assert to_dict(reloaded) == to_dict(cfg)


def test_save_defaults_to_source_path(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.project.name = "renamed"
    save_config(cfg)
    assert load_config(config_file).project.name == "renamed"


def test_unquoted_yaml_dates_are_coerced(tmp_path: Path) -> None:
    raw = dict(MINIMAL_CONFIG)
    path = tmp_path / "config.yaml"
    # Unquoted dates: yaml.safe_load turns these into datetime.date objects.
    path.write_text(
        yaml.safe_dump(raw).replace(
            "'2024-09-15'", "2024-09-15"
        ).replace("'2024-09-20'", "2024-09-20"),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.gfm.temporal_extent == ["2024-09-15", "2024-09-20"]


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_unknown_section_key_raises(tmp_path: Path) -> None:
    raw = {"dem": {"delviery": "local"}}  # typo
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="delviery"):
        load_config(path)


def test_validate_passes_on_good_config(config_file: Path) -> None:
    assert validate(load_config(config_file)) == []


def test_validate_missing_aoi(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.aoi_path = "missing.gpkg"
    assert any("AOI file not found" in e for e in validate(cfg))


def test_validate_bad_delivery(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.dem.delivery = "carrier_pigeon"
    assert any("dem.delivery" in e for e in validate(cfg))


def test_validate_missing_gee_project(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.project.gee_project = ""
    assert any("gee_project" in e for e in validate(cfg))
    cfg.dem.enabled = False  # not needed when the dem step is off
    assert not any("gee_project" in e for e in validate(cfg))


def test_validate_bad_aggregation(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.aggregation = "median"
    assert any("gfm.aggregation" in e for e in validate(cfg))


@pytest.mark.parametrize(
    "extent",
    [["2024-09-15"], ["not-a-date", "2024-09-20"], ["2024-09-20", "2024-09-15"]],
)
def test_validate_bad_temporal_extent(config_file: Path, extent: list[str]) -> None:
    cfg = load_config(config_file)
    cfg.gfm.temporal_extent = extent
    assert any("temporal_extent" in e for e in validate(cfg))


def test_validate_unknown_flexth_key(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.flexth["io"] = {"dtm": "x.tif"}  # io is generated, not user-set
    assert any("flexth" in e for e in validate(cfg))


def test_tracked_demo_project_config_is_valid_apart_from_data_files() -> None:
    """The tracked demo project config must load cleanly; only absent data may fail."""
    repo_config = (
        Path(__file__).parent.parent / "projects" / "austria_demo" / "config.yaml"
    )
    cfg = load_config(repo_config)
    errors = validate(cfg)
    assert all("AOI file" in e for e in errors)
