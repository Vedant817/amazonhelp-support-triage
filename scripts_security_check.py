"""Live security verification against a running server.

Complements tests/test_security.py: pytest drives the app in-process, so it
cannot observe what the middleware stack actually emits over a socket, which is
what a browser or an attacker sees. Requires a server:

    python main.py serve                       # in another terminal
    python scripts_security_check.py [base_url]

Exits 0 when every check passes, 1 on any failure, 2 when no server is reachable.
This script previously only printed what it observed, which meant a broken
security posture produced output that looked identical to a healthy one — a check
that cannot fail is documentation, not verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts_verify_common import base_url_from_argv, exit_if_unreachable

BASE = base_url_from_argv()
FAILURES: list[str] = []

EXPECTED_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
}

# 131072 bytes exceeds the 64 KiB body cap enforced by the middleware.
OVERSIZE_BYTES = 131072


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    exit_if_unreachable(BASE)
    client = httpx.Client(base_url=BASE, timeout=30.0)

    # ---------------------------------------------------------------- dashboard
    dashboard = client.get("/")
    check("dashboard returns 200", dashboard.status_code == 200, f"status={dashboard.status_code}")
    headers = {key.lower(): value for key, value in dashboard.headers.items()}

    for name, expected in EXPECTED_HEADERS.items():
        actual = headers.get(name, "")
        check(f"{name} is {expected}", actual == expected, f"got {actual!r}")

    csp = headers.get("content-security-policy", "")
    check("content-security-policy present", bool(csp), "missing")
    check("CSP forbids remote frames",
          "frame-ancestors 'none'" in csp or "default-src 'none'" in csp, f"csp={csp[:80]!r}")
    check("permissions-policy present", "permissions-policy" in headers, "missing")

    # ---------------------------------------------------------------------- API
    health = client.get("/api/v1/health")
    check("health is 200 or 503 (warming)", health.status_code in (200, 503),
          f"status={health.status_code}")
    api_headers = {key.lower(): value for key, value in health.headers.items()}
    check("API responses are not cached", api_headers.get("cache-control") == "no-store",
          f"cache-control={api_headers.get('cache-control')!r}")
    check("no wildcard CORS by default", "access-control-allow-origin" not in api_headers,
          f"got {api_headers.get('access-control-allow-origin')!r}")

    # ---------------------------------------------------------------- rejection
    invalid_session = client.post("/api/v1/triage", json={"text": "hello", "session_id": "bad space!"})
    check("unsafe session_id rejected", invalid_session.status_code == 422,
          f"status={invalid_session.status_code}")

    oversize = client.post(
        "/api/v1/triage",
        content=b"x" * OVERSIZE_BYTES,
        headers={"Content-Type": "application/json", "Content-Length": str(OVERSIZE_BYTES)},
    )
    check("oversized body rejected", oversize.status_code == 413, f"status={oversize.status_code}")

    # ------------------------------------------------------------ origin policy
    evil = client.get("/api/v1/health", headers={"Origin": "http://evil.example"})
    check("evil origin gets no CORS grant",
          "access-control-allow-origin" not in {k.lower() for k in evil.headers},
          f"got {evil.headers.get('access-control-allow-origin')!r}")
    check("loopback host accepted",
          client.get("/api/v1/health", headers={"Host": "localhost"}).status_code in (200, 503),
          "localhost Host header rejected")

    client.close()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("All security checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
