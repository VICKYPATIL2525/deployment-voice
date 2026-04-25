import json
import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

HOST    = os.environ.get("MINDSPACE_VOICE_HOST", "http://localhost:9100")
API_KEY = os.environ.get("MINDSPACE_VOICE_API_KEY", "")

if not API_KEY:
    print("ERROR: MINDSPACE_VOICE_API_KEY is not set in .env")
    sys.exit(1)

HEADERS      = {"X-API-Key": API_KEY}
BAD_HEADERS  = {"X-API-Key": "wrong-key"}
PAYLOAD      = json.loads((Path(__file__).parent / "payload.json").read_text())

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def print_result(label, response, elapsed_ms):
    status_ok = 200 <= response.status_code < 300
    badge = PASS if status_ok else FAIL
    print(f"\n{'='*60}")
    print(f"  [{badge}]  {label}")
    print(f"  HTTP {response.status_code}  |  {elapsed_ms:.0f} ms")
    print(f"{'='*60}")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)


def call(method, path, **kwargs):
    url = f"{HOST}{path}"
    t0  = time.perf_counter()
    r   = getattr(requests, method)(url, timeout=30, **kwargs)
    ms  = (time.perf_counter() - t0) * 1000
    return r, ms


print(f"\nTarget : {HOST}")
print(f"API Key: {API_KEY[:8]}...")

# ── 1. GET / — service info ───────────────────────────────────────────────────
r, ms = call("get", "/", headers=HEADERS)
print_result("GET /  — service info", r, ms)

# ── 2. GET /health — health check (no auth required) ─────────────────────────
r, ms = call("get", "/health")
print_result("GET /health  — health check (no auth)", r, ms)

# ── 3. GET / with wrong key — auth rejection ──────────────────────────────────
r, ms = call("get", "/", headers=BAD_HEADERS)
print_result("GET /  with wrong API key  — expect 403", r, ms)

# ── 4. POST /predict — full prediction ───────────────────────────────────────
r, ms = call("post", "/predict", json=PAYLOAD, headers=HEADERS)
print_result("POST /predict  — prediction", r, ms)

# ── 5. POST /predict with missing features — expect 422 ──────────────────────
r, ms = call("post", "/predict", json={"features": {"fake_feature": 0.0}}, headers=HEADERS)
print_result("POST /predict  with bad payload  — expect 422", r, ms)

# ── 6. GET /model/info — model structure ─────────────────────────────────────
r, ms = call("get", "/model/info", headers=HEADERS)
data  = r.json()
if isinstance(data.get("feature_names"), list):
    data["feature_names"] = f"[... {len(data['feature_names'])} features ...]"
print(f"\n{'='*60}")
badge = PASS if 200 <= r.status_code < 300 else FAIL
print(f"  [{badge}]  GET /model/info  — model structure")
print(f"  HTTP {r.status_code}  |  {ms:.0f} ms")
print(f"{'='*60}")
print(json.dumps(data, indent=2))

print(f"\n{'='*60}")
print("  All endpoint tests finished.")
print(f"{'='*60}\n")
