"""Security primitives for the local support API.

This module centralises the defensive layer that sits in front of the model:

* :class:`TokenBucketRateLimiter` — bounded, thread-safe request throttling so a
  runaway local process (or a malicious page abusing the local port) cannot
  exhaust CPU.
* :func:`resolve_allowed_origins` — same-origin by default; an explicit
  comma-separated allowlist can be provided through ``SUPPORT_ALLOWED_ORIGINS``.
* :func:`apply_security_headers` — conservative response headers (CSP, frame
  protection, referrer policy) for both the dashboard and JSON responses.

All configuration is read from the environment with safe defaults, so the
secure behaviour is what you get when you do nothing.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Tuple


#: Hard ceiling for a single request body. The largest legitimate payload is a
#: 1000-character JSON message, so 64 KiB is already generous.
MAX_BODY_BYTES = int(os.getenv("SUPPORT_MAX_BODY_BYTES", str(64 * 1024)))

#: Requests allowed per minute per client, and the burst capacity above the
#: steady-state refill.
RATE_LIMIT_PER_MINUTE = int(os.getenv("SUPPORT_RATE_LIMIT_PER_MINUTE", "120"))
RATE_LIMIT_BURST = int(os.getenv("SUPPORT_RATE_LIMIT_BURST", "30"))

#: Maximum number of distinct clients tracked before the oldest idle buckets are
#: evicted, so the limiter itself cannot be turned into a memory-exhaustion bug.
RATE_LIMIT_MAX_CLIENTS = int(os.getenv("SUPPORT_RATE_LIMIT_MAX_CLIENTS", "4096"))

#: Content Security Policy for the self-contained dashboard. Everything is
#: inline (no external assets), so only ``'self'`` connections are allowed and
#: all remote script/style/frame origins are denied.
DASHBOARD_CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)

#: Static, cacheable security headers applied to every response.
STATIC_SECURITY_HEADERS: Dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": DASHBOARD_CSP,
}


def resolve_allowed_origins() -> List[str]:
    """Return the explicit CORS allowlist (empty means same-origin only).

    ``SUPPORT_ALLOWED_ORIGINS`` accepts a comma-separated list of exact origins,
    for example ``http://127.0.0.1:5173,http://localhost:5173`` for a separate
    front-end dev server. ``*`` is deliberately rejected: it would let any
    website in the browser reach this local API.
    """
    raw = os.getenv("SUPPORT_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    origins = []
    for item in raw.split(","):
        origin = item.strip().rstrip("/")
        if not origin or origin == "*":
            continue
        origins.append(origin)
    return origins


class TokenBucketRateLimiter:
    """Thread-safe per-client token bucket with bounded memory.

    The bucket refills continuously at ``per_minute / 60`` tokens per second and
    starts with ``burst`` tokens. When the number of tracked clients exceeds
    ``max_clients``, the least-recently-used buckets are dropped.
    """

    def __init__(
        self,
        per_minute: int = RATE_LIMIT_PER_MINUTE,
        burst: int = RATE_LIMIT_BURST,
        max_clients: int = RATE_LIMIT_MAX_CLIENTS,
    ) -> None:
        self.per_minute = max(1, int(per_minute))
        self.burst = max(1, int(burst))
        self.max_clients = max(1, int(max_clients))
        self._refill_per_second = self.per_minute / 60.0
        self._lock = threading.Lock()
        # client key -> (tokens, last_refill_monotonic)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def allow(self, client_key: str) -> bool:
        """Consume one token for ``client_key``; return False when exhausted."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(client_key, (float(self.burst), now))
            elapsed = max(0.0, now - last)
            tokens = min(float(self.burst), tokens + elapsed * self._refill_per_second)
            if tokens < 1.0:
                self._buckets[client_key] = (tokens, now)
                return False
            self._buckets[client_key] = (tokens - 1.0, now)
            if len(self._buckets) > self.max_clients:
                self._evict_lru_locked()
            return True

    def _evict_lru_locked(self) -> None:
        """Drop the oldest half of the tracked buckets (caller holds the lock)."""
        ordered = sorted(self._buckets.items(), key=lambda item: item[1][1])
        for key, _ in ordered[: max(1, len(ordered) // 2)]:
            self._buckets.pop(key, None)

    def tracked_clients(self) -> int:
        with self._lock:
            return len(self._buckets)


def apply_security_headers(response, *, api_route: bool) -> None:
    """Attach the static security headers, and no-store for API responses."""
    for name, value in STATIC_SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if api_route:
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Pragma", "no-cache")


def client_identifier(request) -> str:
    """Best-effort stable client key.

    Only the socket peer is used. ``X-Forwarded-For`` is intentionally ignored
    because trusting it without a proxy allowlist would let a caller forge
    arbitrary identities and bypass throttling.
    """
    client = getattr(request, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    return str(host)
