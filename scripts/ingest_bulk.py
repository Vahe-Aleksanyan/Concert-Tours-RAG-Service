from __future__ import annotations

import os
import json
import pathlib
import sys
from typing import List

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8003").rstrip("/")
INGEST_ENDPOINT = f"{API_BASE}/api/v1/ingest"

RELEVANT_DIR = pathlib.Path(__file__).parent / "synthetic_data" / "relevant"

if not RELEVANT_DIR.exists():
    sys.exit(f"❌ Directory not found: {RELEVANT_DIR}")


# ---------------------------------------------------------------------------

def ingest_file(path: pathlib.Path) -> tuple[int, str]:
    """Send one file, return (status_code, summary)"""
    text = path.read_text()
    resp = requests.post(INGEST_ENDPOINT, json={"text": text}, timeout=30)
    if resp.ok:
        return resp.status_code, resp.json().get("summary", "")
    return resp.status_code, resp.text[:200]


def main() -> None:
    files: List[pathlib.Path] = sorted(RELEVANT_DIR.glob("*.txt"))
    if not files:
        print("No .txt files found in", RELEVANT_DIR)
        return

    failures = []
    for p in tqdm(files, desc="Ingesting"):
        code, msg = ingest_file(p)
        if code != 201:
            failures.append((p.name, code, msg))

    print("\n✔ Done.  Success:", len(files) - len(failures), "/", len(files))
    if failures:
        print("\nFailures:")
        for name, code, msg in failures:
            print(f"  • {name}: HTTP {code} — {msg}")


if __name__ == "__main__":
    main()
