"""
Secondary OCR worker — runs Surya from the *back* of the scan list forward,
sharing the per-page OCR cache with the main pipeline.

Use case: speed up a long OCR pass by running two workers in parallel on
the same scan directory. Worker A (the main pipeline) goes 1 → 838; this
worker goes 838 → 1. Both check `output/hocr/<stem>.ocr.json` before
computing, so they skip pages the other has already done. They meet in
the middle and exit when there's nothing left.

Race window: if both pick the same page in the same instant, both compute
it and the last writer wins. Wasted CPU is bounded to a handful of pages
near the meeting point, so we don't bother with a lock.
"""

import logging
import sys
import time
from pathlib import Path

# Make `pipeline.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import OUTPUT_DIR
from pipeline.ocr import _cache_path, run_ocr
from pipeline.preprocess import discover_scans, normalize_image


def main() -> int:
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ocr_reverse_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("ocr-reverse")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    scans = discover_scans()
    log.info(f"Reverse worker starting: {len(scans)} pages, will process from end → start")

    processed = skipped = 0
    for i, scan_path in enumerate(reversed(scans), start=1):
        if _cache_path(scan_path.name).exists():
            skipped += 1
            if skipped % 25 == 0:
                log.info(f"  Skipped {skipped} cached pages so far")
            continue

        log.info(f"[{i}/{len(scans)}] OCR (reverse): {scan_path.name}")
        try:
            normalized = normalize_image(scan_path)
            page = run_ocr(normalized, scan_path.name)
            log.info(f"  -> {len(page.all_words)} words")
            processed += 1
        except KeyboardInterrupt:
            log.warning("Interrupted; exiting.")
            break
        except Exception as e:
            log.error(f"  Failed on {scan_path.name}: {e}")

    log.info(
        f"Reverse worker done. Processed {processed}, skipped {skipped} cached, "
        f"{len(scans) - processed - skipped} not reached."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
