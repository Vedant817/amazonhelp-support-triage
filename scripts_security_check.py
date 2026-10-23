"""Live security header verification against a running server."""
import sys
import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
c = httpx.Client(base_url=BASE, timeout=15)

r = c.get("/")
print("Dashboard status:", r.status_code)
for k in [
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "permissions-policy",
    "cross-origin-opener-policy",
    "cross-origin-resource-policy",
]:
    value = r.headers.get(k, "MISSING")
    print(f"  {k}: {value[:100]}")

print()
r = c.get("/api/v1/health")
print("API health status (503 = still training):", r.status_code)
print("  cache-control:", r.headers.get("cache-control"))
print("  access-control-allow-origin:", r.headers.get("access-control-allow-origin", "absent (good)"))

print()
r = c.post("/api/v1/triage", json={"text": "hello", "session_id": "bad space!"})
print("Invalid session_id status:", r.status_code)
r = c.post("/api/v1/triage", content=b"x" * 131072,
           headers={"Content-Type": "application/json", "Content-Length": "131072"})
print("Oversized body status:", r.status_code)
r = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
print("Evil-origin CORS header:", r.headers.get("access-control-allow-origin", "absent (good)"))
r = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
print("Vary header:", r.headers.get("vary", "none"))
c.close()