"""Execute every dashboard page headless via Streamlit's AppTest.

This catches import errors, widget misuse and config-handling bugs without a
browser. Pages behave read-only here: no buttons are clicked, so no pipeline
runs and no network calls happen (the scenes page only searches on click).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from flood_pipeline.cli import CONFIG_ENV_VAR
from flood_pipeline.config import load_config
from flood_pipeline.steps import flexth_step

REPO_ROOT = Path(__file__).parent.parent
APP_DIR = REPO_ROOT / "src" / "flood_pipeline" / "app"
DEMO_CONFIG = REPO_ROOT / "projects" / "austria_demo" / "config.yaml"
PAGE_TIMEOUT_SECONDS = 120

PAGES = [
    APP_DIR / "Home.py",
    APP_DIR / "pages" / "1_Config.py",
    APP_DIR / "pages" / "2_AOI.py",
    APP_DIR / "pages" / "3_GFM_Scenes.py",
    APP_DIR / "pages" / "4_Run.py",
]


def _run_page(page: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(DEMO_CONFIG))
    test = AppTest.from_file(str(page), default_timeout=PAGE_TIMEOUT_SECONDS)
    test.run()
    return test


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_exception(page: Path, monkeypatch) -> None:
    result = _run_page(page, monkeypatch)
    assert not result.exception, f"{page.name} raised: {result.exception}"


def _demo_bands():
    return load_config(DEMO_CONFIG).gfm_output_bands()


needs_outputs = pytest.mark.skipif(
    not _demo_bands(),
    reason="needs per-scene FLEXTH outputs (run the pipeline first)",
)


@needs_outputs
def test_results_page_renders_with_outputs(monkeypatch) -> None:
    result = _run_page(APP_DIR / "pages" / "5_Results.py", monkeypatch)
    assert not result.exception, f"5_Results.py raised: {result.exception}"


@needs_outputs
def test_results_page_jumps_to_a_flood_area(monkeypatch) -> None:
    """Selecting an area must re-render cleanly; it re-frames the base map."""
    from flood_pipeline.config import vector_path

    cfg = load_config(DEMO_CONFIG)
    band = cfg.gfm_output_bands()[0]
    scenes = flexth_step.find_scene_outputs(cfg.scene_output_root(band))
    first_stamp = next(iter(scenes))
    if not vector_path(cfg.gfm_scene_path(band, first_stamp)).exists():
        pytest.skip("needs GFM flood-area polygons (re-run the gfm step)")

    result = _run_page(APP_DIR / "pages" / "5_Results.py", monkeypatch)
    picker = next(s for s in result.selectbox if s.label == "Flood area")
    assert len(picker.options) > 1  # whole-AOI entry plus at least one area

    result = picker.select(picker.options[1]).run()
    assert not result.exception, f"5_Results.py raised: {result.exception}"


@needs_outputs
def test_switching_scenes_keeps_the_map_where_the_user_left_it(monkeypatch) -> None:
    """Stepping the scene slider must not undo a jump.

    The slider changes which polygons the picker offers, so Streamlit resets the
    selection — that must not be mistaken for the user asking to zoom back out.
    """
    from flood_pipeline.config import vector_path

    cfg = load_config(DEMO_CONFIG)
    band = cfg.gfm_output_bands()[0]
    stamps = list(flexth_step.find_scene_outputs(cfg.scene_output_root(band)))
    if len(stamps) < 2 or not all(
        vector_path(cfg.gfm_scene_path(band, s)).exists() for s in stamps
    ):
        pytest.skip("needs two scenes with GFM flood-area polygons")

    result = _run_page(APP_DIR / "pages" / "5_Results.py", monkeypatch)
    picker = next(s for s in result.selectbox if s.label == "Flood area")

    home = result.session_state["results_home_bounds"]
    selected_option = None

    for opt in picker.options[1:]:
        res = picker.select(opt).run()
        try:
            bounds = res.session_state["results_view_bounds"]
        except KeyError:
            bounds = None

        if bounds and bounds != home:
            selected_option = opt
            result = res
            break

    if selected_option is None:
        pytest.skip("No flood area polygon with bounds distinct from home bounds")

    jumped = result.session_state["results_view_bounds"]
    assert jumped != result.session_state["results_home_bounds"]

    # SelectSlider.options are the *formatted* labels while .value is the raw
    # stamp, so step by index rather than comparing the two.
    slider = next(s for s in result.select_slider if s.label == "Scene")
    before = slider.value
    other = 1 if before == stamps[0] else 0
    result = slider.set_value(slider.options[other]).run()
    assert not result.exception, f"5_Results.py raised: {result.exception}"

    moved_on = next(s for s in result.select_slider if s.label == "Scene")
    assert moved_on.value != before, "the scene slider did not actually step"
    picker = next(s for s in result.selectbox if s.label == "Flood area")
    assert picker.value == "— whole AOI —"  # Streamlit drops the stale selection

    assert result.session_state["results_view_bounds"] == jumped

def test_results_page_renders_when_no_scenes(tmp_path: Path, monkeypatch) -> None:
    """With no per-scene outputs the page shows an info notice, not an error."""
    import shutil

    project = tmp_path / "empty_project"
    project.mkdir()
    src_cfg = DEMO_CONFIG.read_text(encoding="utf-8")
    (project / "config.yaml").write_text(src_cfg, encoding="utf-8")
    shutil.copy(DEMO_CONFIG.parent / "aoi_drawn.geojson", project / "aoi_drawn.geojson")

    monkeypatch.setenv(CONFIG_ENV_VAR, str(project / "config.yaml"))
    test = AppTest.from_file(
        str(APP_DIR / "pages" / "5_Results.py"), default_timeout=PAGE_TIMEOUT_SECONDS
    )
    test.run()
    assert not test.exception
    assert any("run the" in str(msg.value).lower() for msg in test.info)


def test_gfm_output_bands_from_disk(tmp_path: Path) -> None:
    import shutil

    from flood_pipeline.config import load_config as _load

    project = tmp_path / "p"
    project.mkdir()
    (project / "config.yaml").write_text(
        DEMO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copy(DEMO_CONFIG.parent / "aoi_drawn.geojson", project / "aoi_drawn.geojson")
    cfg = _load(project / "config.yaml")
    (cfg.output_dir / "ensemble" / "2024-09-16_051230").mkdir(parents=True)
    (cfg.output_dir / "dlr" / "2024-09-16_051230").mkdir(parents=True)
    assert cfg.gfm_output_bands() == ["dlr", "ensemble"]


def _project_with_band_dirs(tmp_path: Path, *bands: str):
    """A copy of the demo project with per-band FLEXTH output folders on disk."""
    import shutil

    from flood_pipeline.config import load_config as _load

    project = tmp_path / "p"
    project.mkdir()
    (project / "config.yaml").write_text(
        DEMO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    shutil.copy(DEMO_CONFIG.parent / "aoi_drawn.geojson", project / "aoi_drawn.geojson")
    cfg = _load(project / "config.yaml")
    for band in bands:
        (cfg.output_dir / band / "2024-09-16_051230").mkdir(parents=True)
    return cfg


def test_band_picker_hides_folders_from_earlier_runs(tmp_path: Path) -> None:
    """Only the configured band is offered, so nothing stale can be loaded.

    Switching gfm.compare_algorithms true->false leaves the old per-algorithm
    folders on disk. They cover a different date range and have no OSM outputs
    (the osm step only writes for cfg.resolved_bands()).
    """
    from flood_pipeline.app import ui

    cfg = _project_with_band_dirs(
        tmp_path, "dlr", "ensemble", "ensemble_likelihood", "list", "tuw"
    )
    cfg.gfm.compare_algorithms = False
    cfg.gfm.band = "ensemble_likelihood"
    assert cfg.gfm_output_bands() == ["dlr", "ensemble", "ensemble_likelihood", "list", "tuw"]
    assert ui.configured_output_bands(cfg) == ["ensemble_likelihood"]


def test_band_picker_lists_every_algorithm_when_comparing(tmp_path: Path) -> None:
    """compare_algorithms offers the four algorithm bands, still not the leftovers."""
    from flood_pipeline.app import ui

    cfg = _project_with_band_dirs(
        tmp_path, "dlr", "ensemble", "ensemble_likelihood", "list", "tuw"
    )
    cfg.gfm.compare_algorithms = True
    assert ui.configured_output_bands(cfg) == ["ensemble", "dlr", "tuw", "list"]


def test_band_picker_omits_a_configured_band_without_outputs(tmp_path: Path) -> None:
    """A configured band that never ran must not be offered as selectable."""
    from flood_pipeline.app import ui

    cfg = _project_with_band_dirs(tmp_path, "dlr")
    cfg.gfm.compare_algorithms = False
    cfg.gfm.band = "ensemble_likelihood"
    assert ui.configured_output_bands(cfg) == []


def test_flood_area_label_shows_rank_and_size() -> None:
    from flood_pipeline.app import ui

    assert ui.flood_area_label(1, 124.34) == "#1 — 124.3 ha"
    assert ui.flood_area_label(12, 0.09) == "#12 — 0.1 ha"


def test_jump_bounds_pad_around_the_geometry() -> None:
    from shapely.geometry import box

    from flood_pipeline.app import ui

    (south, west), (north, east) = ui.jump_bounds(box(10.0, 47.0, 11.0, 48.0))
    assert west < 10.0 and south < 47.0
    assert east > 11.0 and north > 48.0
    assert (east - west) == pytest.approx(1.5, rel=1e-6)  # 1 degree + 25% each side


def test_jump_bounds_stay_usable_for_a_tiny_area() -> None:
    """A single-pixel area must not collapse the map to a degenerate box."""
    from shapely.geometry import box

    from flood_pipeline.app import ui

    (south, west), (north, east) = ui.jump_bounds(box(10.0, 47.0, 10.0001, 47.0001))
    assert east - west >= 0.002
    assert north - south >= 0.002


def test_flood_areas_is_none_when_the_file_is_missing(tmp_path: Path) -> None:
    from flood_pipeline.app import ui

    assert ui.flood_areas(tmp_path / "gfm_flood_max.gpkg") is None


def test_flood_areas_reads_the_polygons_largest_first(tmp_path: Path) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from flood_pipeline import polygonize
    from flood_pipeline.app import ui
    from flood_pipeline.config import vector_path

    mask = np.zeros((12, 12), dtype="uint8")
    mask[1:5, 1:5] = 1
    mask[7:9, 1:3] = 1
    raster = tmp_path / "gfm_flood_max.tif"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500_000.0, 5_000_000.0, 30.0, 30.0),
    ) as dst:
        dst.write(mask, 1)
    polygonize.write_flood_polygons(
        raster, min_area_ha=0.0, scene="max", log=lambda _line: None
    )

    areas = ui.flood_areas(vector_path(raster))

    assert list(areas["area_id"]) == [1, 2]
    assert areas["area_ha"].iloc[0] > areas["area_ha"].iloc[1]


def test_flood_area_layer_emphasizes_the_selected_area() -> None:
    import folium
    import geopandas as gpd
    from shapely.geometry import box

    from flood_pipeline.app import ui

    areas = gpd.GeoDataFrame(
        {"area_id": [1, 2], "area_ha": [12.0, 3.0]},
        geometry=[box(10.0, 47.0, 10.1, 47.1), box(11.0, 48.0, 11.05, 48.05)],
        crs="EPSG:4326",
    )
    group = folium.FeatureGroup(name="test")

    ui.add_flood_area_layer(group, areas, selected_id=2)

    layers = list(group._children.values())
    assert len(layers) == 2
    weights = [layer.style_function({})["weight"] for layer in layers]
    assert weights[1] > weights[0]  # the selected area is drawn heavier


def test_create_project_from_template(tmp_path: Path) -> None:
    from flood_pipeline.app import ui

    template = {
        "project": {"name": "old", "gee_project": "ee-test"},
        "aoi": {"path": "somewhere.gpkg"},
        "gfm": {"temporal_extent": ["2024-09-21", "2024-09-22"]},
    }
    created = ui.create_project("new_region", tmp_path / "runs", template)
    assert created == tmp_path / "runs" / "new_region" / "config.yaml"
    assert (tmp_path / "runs" / "new_region" / "flood_data").is_dir()

    cfg = load_config(created)
    assert cfg.project.name == "new_region"
    assert cfg.project.gee_project == "ee-test"  # inherited from the template
    assert cfg.aoi_path == "aoi_drawn.geojson"  # AOI starts fresh
    assert cfg.gfm.temporal_extent == ["2024-09-21", "2024-09-22"]
    assert template["project"]["name"] == "old"  # template not mutated

    with pytest.raises(FileExistsError):
        ui.create_project("new_region", tmp_path / "runs", template)


def _write_road_segments(path: Path, count: int) -> "gpd.GeoDataFrame":
    """A road layer shaped like OSMnx's: many short segments, many tag columns."""
    import geopandas as gpd
    from shapely.geometry import LineString

    step = 0.001
    segments = [
        LineString([(10.0 + i * step, 47.0), (10.0 + (i + 1) * step, 47.0)])
        for i in range(count)
    ]
    roads = gpd.GeoDataFrame(
        {
            "osmid": list(range(count)),
            "highway": ["residential"] * count,
            "name": ["Some long street name"] * count,
            "length_m": [step * 80_000] * count,
        },
        geometry=segments,
        crs="EPSG:4326",
    )
    roads.to_file(path, layer="roads", driver="GPKG")
    return roads


def test_line_overlay_geojson_is_one_stripped_feature(tmp_path: Path) -> None:
    """The map layer must not carry 78k tagged segments — that crashes the app."""
    import json

    from flood_pipeline.app import ui

    path = tmp_path / "roads.gpkg"
    roads = _write_road_segments(path, 500)

    overlay = ui.line_overlay_geojson(path, "roads")

    assert overlay["type"] == "FeatureCollection"
    assert len(overlay["features"]) == 1
    assert overlay["features"][0]["properties"] == {}
    raw_size = len(json.dumps(roads.__geo_interface__))
    assert len(json.dumps(overlay)) < raw_size / 5


def test_line_overlay_geojson_is_none_without_a_file(tmp_path: Path) -> None:
    from flood_pipeline.app import ui

    assert ui.line_overlay_geojson(tmp_path / "missing.gpkg", "roads") is None


def test_line_overlay_geojson_is_none_for_an_empty_layer(tmp_path: Path) -> None:
    import geopandas as gpd

    from flood_pipeline.app import ui

    path = tmp_path / "empty.gpkg"
    gpd.GeoDataFrame({"osmid": []}, geometry=[], crs="EPSG:4326").to_file(
        path, layer="roads", driver="GPKG"
    )

    assert ui.line_overlay_geojson(path, "roads") is None
