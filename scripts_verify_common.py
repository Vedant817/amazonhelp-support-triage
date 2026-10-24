"""Shared helpers for the live-instance verification scripts.

`scripts_smoke.py`, `scripts_ui_check.py` and `scripts_security_check.py` all
talk to a running server rather than importing the app, because they check the
HTTP contract, the rendered dashboard and the middleware stack — none of which
pytest exercises when it drives the app in-process. That means they are the first
thing a new contributor runs, and their first experience used to be an httpx
traceback when no server was listening.

These helpers exist so all three fail with the same actionable sentence instead.
"""

from __future__ import annotations

import sys
from typing import Optional

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Health is the cheapest endpoint that still proves the pipeline finished warming.
REACHABILITY_PATH = "/api/v1/health"


def base_url_from_argv() -> str:
    """Read the optional base URL positional argument."""
    return sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL


def require_server(base_url: str) -> Optional[str]:
    """Return None when a server is reachable, else a human-readable reason.

    Distinguishing "nothing is listening" from "the server answered oddly" matters
    because the remedies differ: the first means start the app, the second means
    read the response.
    """
    try:
        response = httpx.get(f"{base_url}{REACHABILITY_PATH}", timeout=10.0)
    except httpx.ConnectError:
        return (
            f"No server is listening on {base_url}.\n"
            "Start one in another terminal first:\n"
            "    python main.py serve\n"
            "or point this script at a running instance:\n"
            f"    python {sys.argv[0]} http://host:port"
        )
    except httpx.TimeoutException:
        return (
            f"{base_url} accepted the connection but did not answer within 10s.\n"
            "If the pipeline is still warming up, wait for /api/v1/health to return "
            "200 before re-running."
        )
    if response.status_code >= 500:
        return (
            f"{base_url}{REACHABILITY_PATH} returned {response.status_code}.\n"
            "The service is up but not serving yet; check its logs, then re-run."
        )
    return None


def exit_if_unreachable(base_url: str) -> None:
    """Abort with exit code 2 and an actionable message when the server is down."""
    problem = require_server(base_url)
    if problem is not None:
        print(problem, file=sys.stderr)
        raise SystemExit(2)
