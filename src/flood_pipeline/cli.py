"""Command-line interface for the flood pipeline.

Commands:
    run        Run the pipeline (or a subset of steps) for a config file.
    scenes     List available GFM scenes for the AOI (no auth needed).
    auth       One-time interactive Google Earth Engine login.
    dashboard  Launch the Streamlit dashboard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click

from flood_pipeline import runner
from flood_pipeline.config import ConfigError, PipelineConfig, load_config

CONFIG_ENV_VAR = "FLOOD_PIPELINE_CONFIG"


def _load_or_fail(config_path: str) -> PipelineConfig:
    try:
        return load_config(config_path)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e


@click.group()
def cli():
    """Unified flood pipeline: FABDEM DEM -> GFM flood extent -> FLEXTH water depth -> GHSL settlement."""


@cli.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--steps",
    "steps_csv",
    default=",".join(runner.STEP_ORDER),
    show_default=True,
    help="Comma-separated subset of: " + ", ".join(runner.STEP_ORDER),
)
def run_cmd(config_path: str, steps_csv: str) -> None:
    """Run the pipeline for CONFIG_PATH."""
    cfg = _load_or_fail(config_path)
    try:
        steps = runner.normalize_steps(steps_csv.split(","))
    except runner.UnknownStepError as e:
        raise click.ClickException(str(e)) from e
    sys.exit(runner.run_pipeline(cfg, steps))


@cli.command("scenes")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--start", default=None, help="Search start date (default: gfm.temporal_extent[0]).")
@click.option("--end", default=None, help="Search end date (default: gfm.temporal_extent[1]).")
@click.option("--max-items", default=100, show_default=True)
def scenes_cmd(config_path: str, start: str | None, end: str | None, max_items: int) -> None:
    """List available GFM scenes for the AOI (public STAC, no auth)."""
    from flood_pipeline.steps import gfm  # lazy: pulls in odc/geopandas

    cfg = _load_or_fail(config_path)
    extent = [start or cfg.gfm.temporal_extent[0], end or cfg.gfm.temporal_extent[1]]
    bbox = gfm.aoi_bbox_4326(cfg.aoi_abs_path)
    items = gfm.search_gfm_items(
        bbox,
        extent,
        stac_url=cfg.gfm.stac_url,
        collection=cfg.gfm.collection,
    )
    # The search itself is uncapped (it returns newest-first, so capping there
    # would drop the oldest scenes silently); apply --max-items here instead.
    items = gfm._cap_items(items, max_items, click.echo)
    bounds = tuple(round(float(value), 5) for value in bbox)
    if len(items) == 0:
        click.echo(f"No {cfg.gfm.collection} items for bbox {bounds} in {extent}.")
        return
    click.echo(f"{len(items)} {cfg.gfm.collection} item(s) for bbox {bounds} in {extent}:")
    for item in items:
        timestamp = f"{item.datetime:%Y-%m-%d %H:%M}" if item.datetime else "(no datetime)"
        click.echo(f"  {timestamp}  {item.id}")


@cli.command("auth")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def auth_cmd(config_path: str) -> None:
    """Authenticate with Google Earth Engine (needed once per machine)."""
    from flood_pipeline import gee  # lazy: pulls in ee

    cfg = _load_or_fail(config_path)
    if not cfg.project.gee_project:
        raise click.ClickException("project.gee_project is not set in the config")
    gee.init_gee(cfg.project.gee_project)
    click.echo(f"Earth Engine ready (project {cfg.project.gee_project}).")


@cli.command("dashboard")
@click.argument("config_path", required=False, type=click.Path(exists=True, dir_okay=False))
def dashboard_cmd(config_path: str | None) -> None:
    """Launch the Streamlit dashboard (optionally preloading CONFIG_PATH)."""
    home = Path(__file__).parent / "app" / "Home.py"
    environment = dict(os.environ)
    if config_path:
        environment[CONFIG_ENV_VAR] = str(Path(config_path).resolve())
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(home)], env=environment
        )
    )


if __name__ == "__main__":
    cli()