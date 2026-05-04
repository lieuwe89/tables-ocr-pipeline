#!/usr/bin/env python3
"""
Re-align + re-export per-page JSON/ALTO from cached OCR (output/hocr/) and
cached LLM raw (output/llm_raw/). No Surya re-run, no LLM calls. Use after
fixing alignment or bbox repair logic.

Run: .venv/bin/python pipeline/refresh_outputs.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import JSON_DIR, LLM_RAW_DIR, SCANS_DIR, SUPPORTED_IMAGE_EXTENSIONS, COMBINED_DIR
from pipeline.ocr import _load_cache
from pipeline.align import align_page
from pipeline.alto_export import export_page_alto
from pipeline.json_export import export_page_json, build_combined_indexes


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("refresh")


def discover_scans(scans_dir: Path) -> list[Path]:
    files = [p for p in scans_dir.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    files.sort(key=lambda p: p.stem)
    return files


def main() -> None:
    scan_paths = discover_scans(SCANS_DIR)
    logger.info(f"Discovered {len(scan_paths)} scans")

    aligned_count = 0
    skipped_no_cache = 0
    skipped_no_llm = 0
    failed = 0
    t0 = time.time()

    for i, scan_path in enumerate(scan_paths, start=1):
        filename = scan_path.name
        stem = scan_path.stem

        ocr_page = _load_cache(filename)
        if ocr_page is None:
            skipped_no_cache += 1
            continue

        llm_raw_path = LLM_RAW_DIR / f"{stem}.json"
        if llm_raw_path.exists():
            try:
                gemini_result = json.loads(llm_raw_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"  {filename}: llm_raw unreadable ({e}); fallback to existing JSON")
                gemini_result = None
        else:
            gemini_result = None

        if gemini_result is None:
            existing = JSON_DIR / f"{stem}.json"
            if existing.exists():
                try:
                    gemini_result = json.loads(existing.read_text(encoding="utf-8"))
                except Exception:
                    gemini_result = None

        if gemini_result is None:
            skipped_no_llm += 1
            continue

        try:
            aligned = align_page(gemini_result, ocr_page)
            export_page_alto(ocr_page, aligned)
            export_page_json(ocr_page, aligned)
            aligned_count += 1
        except Exception as e:
            logger.error(f"  {filename}: export failed: {e}")
            failed += 1
            continue

        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            logger.info(f"  [{i}/{len(scan_paths)}] {rate:.1f} pg/s")

    logger.info("Building combined indexes from disk...")
    on_disk: dict[str, dict] = {}
    for p in sorted(JSON_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            fn = data.get("scan_file") or f"{p.stem}.jpg"
            on_disk[fn] = data
        except Exception as e:
            logger.warning(f"  skip {p.name}: {e}")
    build_combined_indexes(on_disk)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Refresh done in {elapsed:.1f}s")
    logger.info(f"  re-aligned + exported: {aligned_count}")
    logger.info(f"  skipped (no OCR cache): {skipped_no_cache}")
    logger.info(f"  skipped (no LLM data):  {skipped_no_llm}")
    logger.info(f"  failed:                  {failed}")
    logger.info(f"  combined index pages:   {len(on_disk)}")


if __name__ == "__main__":
    main()
