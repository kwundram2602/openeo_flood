"""Pipeline steps: DEM, flood extent, water depth and population exposure.

Each step module exposes ``run(cfg, log) -> StepOutcome``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


# Steps report progress through a plain line logger so the CLI can print and
# the dashboard can stream the same output.
LogFn = Callable[[str], None]


@dataclass
class StepOutcome:
    """What a step hands back to the runner.

    ``halt=True`` stops the remaining steps without failing the pipeline
    (used by the DEM Drive-export mode, which needs a manual download).
    """

    halt: bool = False
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
