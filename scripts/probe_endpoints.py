import os
import sys
from pathlib import Path
from typing import List, Tuple

from starlette.testclient import TestClient

# Ensure src on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure data directory for SQLite exists
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from src.api.main import app  # noqa: E402


def probe_endpoints() -> List[Tuple[str, int, str]]:
    results = []
    with TestClient(app) as client:
        # Basic endpoints
        for path in ["/", "/health", "/status", "/docs", "/openapi.json"]:
            try:
                r = client.get(path)
                results.append((path, r.status_code, "OK" if r.status_code < 400 else r.text[:200]))
            except Exception as e:
                results.append((path, 0, f"ERROR: {e}"))

        # API v1 endpoints (safe GETs)
        for path in [
            "/api/v1/specs",
        ]:
            try:
                r = client.get(path)
                results.append((path, r.status_code, "OK" if r.status_code < 400 else r.text[:200]))
            except Exception as e:
                results.append((path, 0, f"ERROR: {e}"))

    return results


def main():
    results = probe_endpoints()
    print("=== API Probe Results ===")
    ok = 0
    for path, status, msg in results:
        print(f"{path}: {status} - {msg}")
        if isinstance(status, int) and 200 <= status < 400:
            ok += 1
    print(f"OK {ok}/{len(results)} checks passed")


if __name__ == "__main__":
    main()

