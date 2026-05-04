#!/usr/bin/env python3
"""Serve the local overlay viewer at http://localhost:8765/viewer/"""
import json
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)

stems = sorted(p.stem for p in (ROOT / "output" / "json").glob("*.json"))

# find the first stem whose JSON has at least one entry with a bbox
start_index = 0
for i, stem in enumerate(stems):
    try:
        data = json.loads((ROOT / "output" / "json" / f"{stem}.json").read_text())
        if data.get("entries") and data["entries"][0].get("entry_bbox"):
            start_index = i
            break
    except Exception:
        pass

manifest = {"stems": stems, "start_index": start_index}
(ROOT / "viewer" / "manifest.json").write_text(json.dumps(manifest))
print(f"Found {len(stems)} pages. Opening http://localhost:8765/viewer/")
webbrowser.open("http://localhost:8765/viewer/")
HTTPServer(("", 8765), SimpleHTTPRequestHandler).serve_forever()
