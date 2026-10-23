"""
Security regression tests: throttling, headers, input validation, cache trust.

These tests pin the defensive behaviour added to the local API so a future
refactor cannot silently reopen a vulnerability.
"""

import os
import time
import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.security import (
    TokenBucketRateLimiter,
    resolve_allowed_origins,
    client_identifier,
    apply_security_headers,
    STATIC_SECURITY_HEADERS,
)
from src.schemas import TriageRequest
from src.model_factory import _cache_is_trusted, _write_cache_with_manifest, _file_sha256


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TestTokenBucketRateLimiter:
    def test_burst_then_block(self):
        limiter = TokenBucketRateLimiter(per_minute=60, burst=3)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False

    def test_refill_over_time(self):
        limiter = TokenBucketRateLimiter(per_minute=6000, burst=1)  # 100 tokens/s
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False
        time.sleep(0.05)
        assert limiter.allow("a") is True

    def test_clients_are_isolated(self):
        limiter = TokenBucketRateLimiter(per_minute=60, burst=1)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True  # independent bucket
        assert limiter.allow("a") is False

    def test_memory_is_bounded(self):
        limiter = TokenBucketRateLimiter(per_minute=60, burst=1, max_clients=10)
        for i in range(200):
            limiter.allow(f"client-{i}")
        assert limiter.tracked_clients() <= 10

    def test_thread_safety(self):
        limiter = TokenBucketRateLimiter(per_minute=60, burst=1000)
        results = []
        lock = threading.Lock()

        def worker():
            local = [limiter.allow("shared") for _ in range(50)]
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 8 workers x 50 calls = 400 requests, all within the 1000-token burst.
        assert len(results) == 400
        assert sum(1 for r in results if r) == 400


# ---------------------------------------------------------------------------
# Origin allowlist
# ---------------------------------------------------------------------------

class TestOriginAllowlist:
    def test_default_is_same_origin_only(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SUPPORT_ALLOWED_ORIGINS", None)
            assert resolve_allowed_origins() == []

    def test_parses_comma_list(self):
        with patch.dict(os.environ, {"SUPPORT_ALLOWED_ORIGINS": "http://a.test, http://b.test/"}):
            assert resolve_allowed_origins() == ["http://a.test", "http://b.test"]

    def test_rejects_wildcard(self):
        with patch.dict(os.environ, {"SUPPORT_ALLOWED_ORIGINS": "*"}):
            assert resolve_allowed_origins() == []

    def test_wildcard_mixed_with_real_origin_is_dropped(self):
        with patch.dict(os.environ, {"SUPPORT_ALLOWED_ORIGINS": "*,http://ok.test"}):
            assert resolve_allowed_origins() == ["http://ok.test"]


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host, headers=None):
        self.client = _FakeClient(host) if host is not None else None
        self.headers = headers or {}


class TestClientIdentifier:
    def test_uses_socket_peer(self):
        req = _FakeRequest("127.0.0.1")
        assert client_identifier(req) == "127.0.0.1"

    def test_forwarded_header_cannot_spoof_identity(self):
        """A forged X-Forwarded-For must not change the bucket key."""
        req = _FakeRequest("10.0.0.5", {"X-Forwarded-For": "1.2.3.4"})
        assert client_identifier(req) == "10.0.0.5"

    def test_missing_client_is_unknown(self):
        assert client_identifier(_FakeRequest(None)) == "unknown"


# ---------------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    class _Response:
        def __init__(self):
            self.headers = {}

    def test_static_headers_applied(self):
        resp = self._Response()
        apply_security_headers(resp, api_route=False)
        for name, value in STATIC_SECURITY_HEADERS.items():
            assert resp.headers[name] == value

    def test_no_store_only_for_api(self):
        api = self._Response()
        apply_security_headers(api, api_route=True)
        assert api.headers["Cache-Control"] == "no-store"

        page = self._Response()
        apply_security_headers(page, api_route=False)
        assert "Cache-Control" not in page.headers

    def test_csp_blocks_remote_origins(self):
        csp = STATIC_SECURITY_HEADERS["Content-Security-Policy"]
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "http://" not in csp and "https://" not in csp


# ---------------------------------------------------------------------------
# Request schema validation
# ---------------------------------------------------------------------------

class TestSessionIdValidation:
    @pytest.mark.parametrize("value", ["abc", "session-1", "a.b_c:d", "A" * 80])
    def test_accepts_safe_ids(self, value):
        req = TriageRequest(text="hello", session_id=value)
        assert req.session_id == value

    @pytest.mark.parametrize(
        "value",
        [
            "A" * 81,               # too long
            "has space",             # space not allowed
            "newline\ninjection",    # log injection
            "<script>alert(1)</script>",  # markup
            "semi;colon",            # separator
            "../../etc/passwd",      # path traversal
        ],
    )
    def test_rejects_unsafe_ids(self, value):
        with pytest.raises(Exception):
            TriageRequest(text="hello", session_id=value)

    def test_empty_text_rejected(self):
        with pytest.raises(Exception):
            TriageRequest(text="")

    def test_overlong_text_rejected(self):
        with pytest.raises(Exception):
            TriageRequest(text="x" * 1001)


# ---------------------------------------------------------------------------
# Cache trust
# ---------------------------------------------------------------------------

class TestCacheTrust:
    def test_missing_file_is_untrusted(self, tmp_path):
        assert _cache_is_trusted(tmp_path / "nope.joblib") is False

    def test_file_without_manifest_is_untrusted(self, tmp_path):
        target = tmp_path / "cache.joblib"
        target.write_bytes(b"payload")
        assert _cache_is_trusted(target) is False

    def test_valid_manifest_is_trusted(self, tmp_path):
        target = tmp_path / "cache.joblib"
        target.write_bytes(b"payload")
        digest = _file_sha256(target)
        (tmp_path / "cache.joblib.sha256").write_text(f"{digest}  cache.joblib\n")
        assert _cache_is_trusted(target) is True

    def test_tampered_payload_is_untrusted(self, tmp_path):
        target = tmp_path / "cache.joblib"
        target.write_bytes(b"payload")
        digest = _file_sha256(target)
        (tmp_path / "cache.joblib.sha256").write_text(f"{digest}  cache.joblib\n")
        target.write_bytes(b"malicious-payload")  # tamper after manifest
        assert _cache_is_trusted(target) is False

    def test_tampered_manifest_is_untrusted(self, tmp_path):
        target = tmp_path / "cache.joblib"
        target.write_bytes(b"payload")
        (tmp_path / "cache.joblib.sha256").write_text("0" * 64 + "  cache.joblib\n")
        assert _cache_is_trusted(target) is False

    def test_kill_switch_disables_cache(self, tmp_path):
        target = tmp_path / "cache.joblib"
        target.write_bytes(b"payload")
        digest = _file_sha256(target)
        (tmp_path / "cache.joblib.sha256").write_text(f"{digest}  cache.joblib\n")
        with patch.dict(os.environ, {"SUPPORT_DISABLE_PIPELINE_CACHE": "1"}):
            assert _cache_is_trusted(target) is False

    def test_write_helper_creates_verifiable_manifest(self, tmp_path):
        target = tmp_path / "sub" / "cache.joblib"
        _write_cache_with_manifest({"example": 1}, target)
        assert target.is_file()
        assert (tmp_path / "sub" / "cache.joblib.sha256").is_file()
        assert _cache_is_trusted(target) is True


# ---------------------------------------------------------------------------
# API middleware integration
# ---------------------------------------------------------------------------

class TestApiMiddleware:
    """Middleware-level checks that do not require the trained model.

    ``TestClient`` is intentionally *not* used as a context manager: that would
    run the application lifespan and train the pipeline on first call. Requests
    still exercise the full middleware stack without it.
    """

    @pytest.fixture()
    def client(self):
        from src import api as api_module
        return TestClient(api_module.app)

    def test_security_headers_on_dashboard(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Content-Security-Policy" in response.headers

    def test_api_responses_are_not_cached(self, client):
        response = client.get("/api/v1/health")
        # 503 while warming up is fine; headers must still be present.
        assert response.status_code in (200, 503)
        assert response.headers["Cache-Control"] == "no-store"

    def test_oversized_body_rejected(self, client):
        response = client.post(
            "/api/v1/triage",
            content=b"x" * (128 * 1024),
            headers={"Content-Type": "application/json", "Content-Length": str(128 * 1024)},
        )
        assert response.status_code == 413

    def test_invalid_session_id_rejected(self, client):
        response = client.post(
            "/api/v1/triage",
            json={"text": "hello", "session_id": "bad space!"},
        )
        assert response.status_code == 422

    def test_rate_limit_returns_429(self, client):
        from src import api as api_module
        original = api_module._rate_limiter
        api_module._rate_limiter = TokenBucketRateLimiter(per_minute=60, burst=2)
        try:
            codes = [
                client.get("/api/v1/health").status_code
                for _ in range(5)
            ]
            assert 429 in codes
        finally:
            api_module._rate_limiter = original

    def test_cors_headers_absent_by_default(self, client):
        """Without an allowlist, no Access-Control-Allow-Origin may be emitted."""
        response = client.get("/api/v1/health", headers={"Origin": "http://evil.test"})
        assert "access-control-allow-origin" not in {k.lower() for k in response.headers}

    def test_untrusted_host_header_rejected(self, client):
        """DNS-rebinding guard: a foreign Host header is refused."""
        response = client.get("/api/v1/health", headers={"Host": "evil.example.com"})
        assert response.status_code == 400

    def test_localhost_host_header_accepted(self, client):
        for host in ("127.0.0.1", "localhost", "127.0.0.1:8080", "localhost:8080"):
            response = client.get("/api/v1/health", headers={"Host": host})
            assert response.status_code in (200, 503), f"failed for {host}"

    def test_ipv6_loopback_host_header_accepted(self, client):
        """Starlette normalises IPv6 hosts with brackets before comparison."""
        response = client.get("/api/v1/health", headers={"Host": "[::1]:8000"})
        assert response.status_code in (200, 503)

    def test_nonstandard_port_host_accepted(self, client):
        response = client.get("/api/v1/health", headers={"Host": "localhost:9999"})
        assert response.status_code in (200, 503)

    def test_docs_enabled_by_default_for_local_use(self):
        from src import api as api_module
        assert api_module.app.docs_url == "/docs"
        assert api_module.app.openapi_url == "/openapi.json"
