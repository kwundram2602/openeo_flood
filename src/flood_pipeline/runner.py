"""Pipeline orchestration: run the dem/gfm/flexth steps in order.

The runner prints ``##[step:<name>] start|done|failed|skipped`` and
``##[pipeline] ...`` marker lines through the supplied logger. The dashboard's
Run page splits the live output on these markers, so change them carefully.
"""

from __future__ import annotations

from collections.abc import Sequence

from flood_pipeline.config import PipelineConfig, validate
from flood_pipeline.steps import LogFn, StepOutcome

STEP_ORDER = ("dem", "gfm", "osm", "flexth", "population")


class UnknownStepError(ValueError):
    """Raised when a requested step name is not one of STEP_ORDER."""


def normalize_steps(steps: Sequence[str]) -> list[str]:
    """Deduplicate and order the requested steps canonically.

    Raises:
        UnknownStepError: For names outside STEP_ORDER.
    """
    requested = {step.strip().lower() for step in steps if step.strip()}
    unknown = requested - set(STEP_ORDER)
    if unknown:
        raise UnknownStepError(
            f"unknown step(s) {sorted(unknown)}; valid steps: {', '.join(STEP_ORDER)}"
        )
    return [step for step in STEP_ORDER if step in requested]


def _step_runner(step: str):
    """Import the step module on first use; ee/geemap/odc imports are slow."""
    if step == "dem":
        from flood_pipeline.steps import dem

        return dem.run
    if step == "gfm":
        from flood_pipeline.steps import gfm

        return gfm.run
    if step == "osm":
        from flood_pipeline.steps import osm

        return osm.run
    if step == "flexth":
        from flood_pipeline.steps import flexth_step

        return flexth_step.run
    from flood_pipeline.steps import population

    return population.run


def _step_enabled(cfg: PipelineConfig, step: str) -> bool:
    if step == "dem":
        return cfg.dem.enabled
    if step == "gfm":
        return cfg.gfm.enabled
    if step == "osm":
        return cfg.osm.enabled
    if step == "flexth":
        return bool(cfg.flexth.get("enabled", True))
    return cfg.population.enabled


def run_pipeline(
    cfg: PipelineConfig, steps: Sequence[str], log: LogFn = print
) -> int:
    """Run the requested steps in canonical order; return a process exit code.

    Steps disabled in the config are skipped even when requested. A step
    returning ``halt=True`` (DEM Drive export) stops the pipeline cleanly.
    """
    errors = validate(cfg)
    if errors:
        log("##[pipeline] invalid config:")
        for error in errors:
            log(f"  - {error}")
        return 1

    for step in normalize_steps(steps):
        if not _step_enabled(cfg, step):
            log(f"##[step:{step}] skipped (disabled in config)")
            continue
        log(f"##[step:{step}] start")
        try:
            outcome: StepOutcome = _step_runner(step)(cfg, log)
        except Exception as e:  # deliberate: any step failure ends the pipeline
            log(f"##[step:{step}] failed: {e}")
            return 1
        log(f"##[step:{step}] done")
        if outcome.halt:
            log(f"##[pipeline] halted: {outcome.message}")
            return 0

    log("##[pipeline] finished")
    return 0
