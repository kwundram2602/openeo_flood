"""Tests for flood_pipeline.config: loading, round-trip, path resolution, validation."""

from pathlib import Path

import pytest
import yaml

from flood_pipeline.config import (
    GFM_ALGORITHM_BANDS,
    ConfigError,
    band_key,
    load_config,
    save_config,
    to_dict,
    validate,
    vector_path,
)

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


def test_band_key_strips_known_suffixes() -> None:
    assert band_key("ensemble_flood_extent") == "ensemble"
    assert band_key("dlr_flood_extent") == "dlr"
    assert band_key("ensemble_water_extent") == "ensemble"
    assert band_key("weird/name") == "weird_name"  # sanitized fallback


def test_resolved_bands_single_by_default(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.gfm.compare_algorithms is False
    assert cfg.resolved_bands() == [("ensemble", "ensemble_flood_extent")]


def test_resolved_bands_all_algorithms_when_comparing(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.compare_algorithms = True
    assert cfg.resolved_bands() == list(GFM_ALGORITHM_BANDS.items())
    assert [k for k, _ in cfg.resolved_bands()] == ["ensemble", "dlr", "tuw", "list"]


def test_compare_algorithms_round_trips(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.compare_algorithms = True
    copy_path = tmp_path / "copy.yaml"
    save_config(cfg, copy_path)
    assert load_config(copy_path).gfm.compare_algorithms is True


def test_load_fills_defaults(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.project.name == "test_run"
    assert cfg.project.data_dir == "flood_data"  # default
    assert cfg.dem.scale == 30  # default
    assert cfg.gfm.aggregation == "max"  # default
    assert cfg.gfm.min_area_ha == 1.0  # default
    assert cfg.flexth["resample"]["crs"] == "EPSG:32633"


def test_everything_resolves_inside_the_project_folder(config_file: Path) -> None:
    """The project folder is the config file's directory; all paths live in it."""
    cfg = load_config(config_file)
    project_dir = config_file.parent
    assert cfg.project_dir == project_dir
    assert cfg.data_dir == project_dir / "flood_data"
    assert cfg.dem_path() == project_dir / "flood_data" / "fabdem.tif"
    assert (
        cfg.gfm_mask_path("ensemble")
        == project_dir / "flood_data" / "ensemble" / "gfm_flood_max.tif"
    )
    assert cfg.aoi_abs_path == project_dir / "aoi.geojson"


def test_absolute_paths_kept(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    absolute = tmp_path / "elsewhere" / "data"
    cfg.project.data_dir = str(absolute)
    assert cfg.data_dir == absolute


def test_gfm_band_and_scene_paths(config_file: Path) -> None:
    cfg = load_config(config_file)
    stamp = "2024-09-16_051230"
    band_dir = cfg.data_dir / "dlr"
    assert cfg.gfm_band_dir("dlr") == band_dir
    assert cfg.gfm_scene_dir("dlr", stamp) == band_dir / stamp
    assert cfg.gfm_scene_path("dlr", stamp) == band_dir / stamp / "gfm_flood.tif"
    assert cfg.gfm_exclusion_path("dlr", stamp) == band_dir / stamp / "gfm_exclusion.tif"
    assert cfg.gfm_mask_path("dlr") == band_dir / "gfm_flood_max.tif"
    assert cfg.gfm_sum_path("dlr") == band_dir / "gfm_flood_sum.tif"
    assert cfg.gfm_reference_water_path("dlr") == band_dir / "gfm_flood_reference_water.tif"


def test_gfm_scene_stamps_discovers_scene_folders(config_file: Path) -> None:
    cfg = load_config(config_file)
    band_dir = cfg.gfm_band_dir("ensemble")
    (band_dir / "2024-09-18_052000").mkdir(parents=True)
    (band_dir / "2024-09-16_051230").mkdir(parents=True)
    (band_dir / "not_a_scene").mkdir(parents=True)  # ignored
    (band_dir / "gfm_flood_max.tif").write_bytes(b"")  # a file, ignored
    assert cfg.gfm_scene_stamps("ensemble") == [
        "2024-09-16_051230",
        "2024-09-18_052000",
    ]
    assert cfg.gfm_scene_stamps("dlr") == []  # absent band dir -> empty


def test_vector_path_swaps_the_raster_suffix(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert vector_path(cfg.gfm_mask_path("ensemble")).name == "gfm_flood_max.gpkg"
    assert (
        vector_path(cfg.gfm_scene_path("ensemble", "2024-09-16_051230")).name
        == "gfm_flood.gpkg"
    )


def test_scene_dirs(config_file: Path) -> None:
    cfg = load_config(config_file)
    stamp = "2024-09-16_051230"
    assert cfg.scene_work_root("tuw") == cfg.work_dir / "tuw"
    assert cfg.scene_output_root("tuw") == cfg.output_dir / "tuw"
    assert cfg.scene_work_dir("tuw", stamp) == cfg.work_dir / "tuw" / stamp
    assert cfg.scene_output_dir("tuw", stamp) == cfg.output_dir / "tuw" / stamp


def test_is_likelihood_band() -> None:
    from flood_pipeline.config import GFM_SELECTABLE_BANDS, is_likelihood_band
    assert is_likelihood_band("ensemble_likelihood") is True
    assert is_likelihood_band("ensemble_flood_extent") is False
    assert "ensemble_likelihood" in GFM_SELECTABLE_BANDS
    assert "ensemble_flood_extent" in GFM_SELECTABLE_BANDS


def test_likelihood_threshold_default_and_round_trip(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.gfm.likelihood_threshold == 25
    cfg.gfm.band = "ensemble_likelihood"
    cfg.gfm.likelihood_threshold = 40
    copy_path = tmp_path / "copy.yaml"
    save_config(cfg, copy_path)
    reloaded = load_config(copy_path)
    assert reloaded.gfm.band == "ensemble_likelihood"
    assert reloaded.gfm.likelihood_threshold == 40


def test_resolved_bands_for_likelihood(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.band = "ensemble_likelihood"
    assert cfg.resolved_bands() == [("ensemble_likelihood", "ensemble_likelihood")]


def test_validate_likelihood_threshold_range(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.band = "ensemble_likelihood"
    cfg.gfm.likelihood_threshold = 0
    assert any("likelihood_threshold" in e for e in validate(cfg))
    cfg.gfm.likelihood_threshold = 25
    assert not any("likelihood_threshold" in e for e in validate(cfg))
    cfg.gfm.band = "ensemble_flood_extent"  # range unchecked for a binary band
    cfg.gfm.likelihood_threshold = 0
    assert not any("likelihood_threshold" in e for e in validate(cfg))


def test_gfm_likelihood_path(config_file: Path) -> None:
    cfg = load_config(config_file)
    stamp = "2024-09-16_051230"
    assert (
        cfg.gfm_likelihood_path("ensemble_likelihood", stamp)
        == cfg.data_dir / "ensemble_likelihood" / stamp / "gfm_likelihood.tif"
    )


def test_scene_fill_path(config_file: Path) -> None:
    cfg = load_config(config_file)
    stamp = "2024-09-16_051230"
    assert (
        cfg.scene_fill_path("dlr", stamp)
        == cfg.output_dir / "dlr" / stamp / "interpolated_fill.tif"
    )


def test_validate_accepts_fill_excluded(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.flexth["fill_excluded"] = True
    assert validate(cfg) == []


def test_gfm_output_bands_lists_bands_with_scenes(config_file: Path) -> None:
    cfg = load_config(config_file)
    (cfg.output_dir / "ensemble" / "2024-09-16_051230").mkdir(parents=True)
    (cfg.output_dir / "dlr" / "2024-09-16_051230").mkdir(parents=True)
    (cfg.output_dir / "empty_band").mkdir(parents=True)  # no scene subdir -> skip
    assert cfg.gfm_output_bands() == ["dlr", "ensemble"]


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


def test_validate_negative_min_area(config_file: Path) -> None:
    cfg = load_config(config_file)
    cfg.gfm.min_area_ha = -1.0
    assert any("gfm.min_area_ha" in e for e in validate(cfg))
    cfg.gfm.min_area_ha = 0.0  # 0 is legal: it means "no filter"
    assert not any("gfm.min_area_ha" in e for e in validate(cfg))


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
