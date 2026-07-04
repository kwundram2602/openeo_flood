"""Google Earth Engine session handling.

``ee.Authenticate()`` caches credentials in the user profile
(~/.config/earthengine), so one interactive authentication from the CLI also
serves later dashboard runs.
"""

from __future__ import annotations

import ee


def init_gee(project: str) -> None:
    """Initialize the EE session, running the interactive auth flow if needed.

    Meant for CLI use: on a machine without cached credentials this opens the
    browser-based Earth Engine login once.
    """
    try:
        ee.Initialize(project=project)
    except Exception:  # ee raises assorted ee/google.auth types; retry after auth
        ee.Authenticate()
        ee.Initialize(project=project)


def try_init_gee(project: str) -> str | None:
    """Initialize non-interactively; return an error message instead of prompting.

    Meant for the dashboard, which must never block on a browser login.
    Returns None on success.
    """
    try:
        ee.Initialize(project=project)
    except Exception as e:  # deliberate: surface any auth/config failure as text
        return (
            f"Google Earth Engine is not ready for project {project!r} ({e}). "
            "Run `uv run flood-pipeline auth config.yaml` once in a terminal."
        )
    return None
