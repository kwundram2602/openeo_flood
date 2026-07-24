"""Colormapping of raster overlays.

FLEXTH floors water depth at its minimum, so a large share of a WD raster sits
on exactly one value that the 2nd-percentile stretch then picks as ``vmin``.
The overlay must still paint those pixels visibly: an untruncated ``Blues``
starts at RGB(247, 251, 255) and they vanish into the basemap.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from flood_pipeline.app import ui

# A color vanishes into the basemap when even its darkest channel sits near
# 255: Blues starts at RGB(247, 251, 255). Its 25% point, RGB(198, 219, 239),
# still reads as blue because red and green drop away — so measure that channel.
MAX_DARKEST_CHANNEL = 220


@pytest.fixture
def depth_raster(tmp_path: Path) -> Path:
    """A WD-like raster: two thirds at the 0.10 m floor, the rest deeper."""
    values = np.full((12, 12), np.nan, dtype="float32")
    values[0:8, :] = 0.10  # the FLEXTH minimum-depth spike
    values[8:12, :] = np.linspace(0.2, 1.5, 4 * 12).reshape(4, 12)
    path = tmp_path / "WD_test.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=12,
        width=12,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        nodata=np.nan,
        transform=from_origin(500_000.0, 5_000_000.0, 30.0, 30.0),
    ) as dst:
        dst.write(values, 1)
    return path


def test_floor_pixels_are_not_painted_white(depth_raster: Path) -> None:
    """Valid pixels at ``vmin`` stay distinguishable from the basemap."""
    overlay = ui.raster_overlay(depth_raster, cmap="Blues", mask_values=(0.0, 999.0))

    opaque = overlay.rgba[..., 3] > 0
    assert opaque.any(), "expected valid pixels in the overlay"
    palest = overlay.rgba[..., :3][opaque].min(axis=-1)
    assert palest.max() <= MAX_DARKEST_CHANNEL


def test_colorbar_matches_the_overlay_colors(depth_raster: Path) -> None:
    """The legend uses the same truncated colormap, so it does not lie."""
    overlay = ui.raster_overlay(depth_raster, cmap="Blues", mask_values=(0.0, 999.0))
    figure = ui.colorbar_figure(overlay.vmin, overlay.vmax, "Blues", "water depth [m]")

    # Sample the rendered bar rather than its artists: that is what the user
    # compares against the map. The bar axes spans (0.05, 0.55, 0.9, 0.3) in
    # figure coordinates, which have their origin at the bottom left.
    figure.canvas.draw()
    canvas = np.asarray(figure.canvas.buffer_rgba())
    height, width = canvas.shape[:2]
    row = int((1.0 - 0.70) * height)
    vmin_end = canvas[row, int(0.07 * width), :3]
    vmax_end = canvas[row, int(0.90 * width), :3]

    assert vmin_end.min() <= MAX_DARKEST_CHANNEL
    assert vmax_end.max() < vmin_end.min(), "bar must still run light to dark"


def test_share_keeps_a_digit_for_tiny_shares() -> None:
    """A share that is small but not zero must never be shown as "0.0%".

    A pre-flood scene can carry a handful of flooded pixels; rounding those to
    0.0% reads as "nothing here" when the map plainly shows something.
    """
    assert ui.format_share(15 / 3_343_257) == "0.00045%"
    assert ui.format_share(0.0) == "0%"
    assert ui.format_share(0.049372) == "4.9%"
    assert ui.format_share(1.0) == "100.0%"


def test_solid_color_overlays_are_untouched(depth_raster: Path) -> None:
    """Binary masks bypass colormapping, so truncation must not affect them."""
    overlay = ui.raster_overlay(depth_raster, mask_values=(0.0,), solid_color="#e31a1c")

    opaque = overlay.rgba[..., 3] > 0
    assert np.array_equal(overlay.rgba[..., :3][opaque][0], np.array([227, 26, 28]))


@pytest.fixture
def reference_water_raster(tmp_path: Path) -> Path:
    """A GFM-style reference mask: 1 = permanent water, 0 = land."""
    values = np.zeros((8, 8), dtype="float32")
    values[2:5, 2:5] = 1.0
    path = tmp_path / "gfm_flood_reference_water.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    ) as dst:
        dst.write(values, 1)
    return path


def test_reference_water_renders_dark_blue(reference_water_raster: Path) -> None:
    """Water pixels get the requested dark blue; land (0) stays transparent."""
    overlay = ui.raster_overlay(
        reference_water_raster, mask_values=(0.0,), solid_color="#08306b"
    )

    opaque = overlay.rgba[..., 3] > 0
    assert opaque.sum() == 9, "only the nine water pixels should be painted"
    assert np.array_equal(overlay.rgba[..., :3][opaque][0], np.array([8, 48, 107]))
