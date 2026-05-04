#!/usr/bin/env python3
"""
Groningen Adresboek 1926 — Data Extraction Pipeline

Main entry point that orchestrates all pipeline stages:
1. Pre-processing: Discover and normalize scan images
2. Tesseract OCR: Extract text with word-level bounding boxes
3. Gemini LLM: Correct, expand, and structure the text
4. Alignment: Link LLM output to OCR bounding boxes
5. Export: Generate ALTO XML and JSON output

Usage:
    python run_pipeline.py                    # Process all pages
    python run_pipeline.py --pages 1-10       # Process specific page range
    python run_pipeline.py --reprocess        # Force reprocess (ignore checkpoints)
    python run_pipeline.py --ocr-only         # Only run Tesseract (skip Gemini)
    python run_pipeline.py --test 3           # Test mode: process only N pages

Requirements:
    - Python 3.10+
    - Tesseract OCR installed with Dutch language pack
    - Google AI Studio API key (set in config_local.py or GOOGLE_AI_API_KEY env var)
    - Scan images in the scans/ directory
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    COMBINED_DIR,
    FAILURES_DIR,
    JSON_DIR,
    LLM_RAW_DIR,
    OUTPUT_DIR,
    SCANS_DIR,
)
from pipeline.preprocess import (
    discover_scans,
    normalize_image,
    get_scan_page_number,
)
from pipeline.ocr import run_ocr, OcrPage
from pipeline.llm import process_page_with_gemini, init_gemini, load_checkpoint, save_checkpoint, is_page_completed
from pipeline.align import align_page
from pipeline.alto_export import export_page_alto
from pipeline.json_export import export_page_json, build_combined_indexes


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging(verbose: bool = False):
    """Configure logging with console and file output."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # ASCII-only console formatter (Windows cp1252 stdout chokes on box-drawing chars)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Force UTF-8 on stdout where supported (Python 3.7+)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(console_fmt)
    
    # File handler
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)
    
    logging.info(f"Logging to {log_file}")


# ── Pipeline stages ───────────────────────────────────────────────────────────


def stage_ocr(scan_paths: list[Path], strategy: str = None, device: str = None) -> dict[str, OcrPage]:
    """
    Stage 2: Run Surya OCR on all scans.

    Surya handles layout analysis internally (column detection, reading order),
    so a separate layout stage is no longer needed.
    """
    logger = logging.getLogger("ocr")
    logger.info(f"═══ Stage 2: Surya OCR ({len(scan_paths)} pages) ═══")

    results = {}
    for i, scan_path in enumerate(scan_paths, start=1):
        logger.info(f"[{i}/{len(scan_paths)}] OCR: {scan_path.name}")
        normalized = normalize_image(scan_path)
        ocr_page = run_ocr(normalized, scan_path.name, strategy=strategy, device=device)
        results[scan_path.name] = ocr_page
        logger.info(f"  → {len(ocr_page.all_words)} words detected")

    logger.info(f"OCR complete: {len(results)} pages processed")
    return results


def _llm_raw_path(scan_path: Path) -> Path:
    return LLM_RAW_DIR / f"{scan_path.stem}.json"


def _save_llm_raw(scan_path: Path, payload: dict) -> None:
    """Atomically write the per-page raw LLM result so a later crash can't lose it."""
    LLM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = _llm_raw_path(scan_path)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _load_llm_raw(scan_path: Path) -> dict | None:
    p = _llm_raw_path(scan_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def stage_gemini(
    scan_paths: list[Path],
    ocr_results: dict[str, OcrPage],
    force_reprocess: bool = False,
) -> dict[str, dict]:
    """
    Stage 3: Process pages through the LLM.

    Persists each page's raw response to ``output/llm_raw/<stem>.json`` as
    soon as it's parsed. Resume logic prefers that file over re-calling the
    LLM. The in-memory `results` dict is only a convenience for the
    downstream stages — disk is the source of truth.
    """
    logger = logging.getLogger("gemini")
    logger.info(f"═══ Stage 3: Gemini LLM ({len(scan_paths)} pages) ═══")

    init_gemini()
    checkpoint = load_checkpoint()
    results: dict[str, dict] = {}

    for i, scan_path in enumerate(scan_paths, start=1):
        filename = scan_path.name

        # 1. Already aligned + exported? Reuse final JSON.
        final_json = JSON_DIR / f"{scan_path.stem}.json"
        if not force_reprocess and final_json.exists():
            try:
                results[filename] = json.loads(final_json.read_text(encoding="utf-8"))
                logger.debug(f"  [{i}/{len(scan_paths)}] Final JSON exists: {filename}")
                continue
            except Exception as e:
                logger.warning(f"  [{i}/{len(scan_paths)}] Final JSON unreadable, will redo: {e}")

        # 2. Raw LLM response on disk? Reuse — the LLM call already succeeded.
        if not force_reprocess:
            raw = _load_llm_raw(scan_path)
            if raw is not None:
                results[filename] = raw
                if filename not in checkpoint["completed"]:
                    checkpoint["completed"].append(filename)
                    save_checkpoint(checkpoint)
                logger.debug(f"  [{i}/{len(scan_paths)}] Loaded llm_raw: {filename}")
                continue

        # 3. No data on disk — call the LLM. (Note: an entry in
        # checkpoint["completed"] without a llm_raw file means the previous
        # run lost in-memory state; re-call.)
        ocr_page = ocr_results.get(filename)
        if not ocr_page:
            logger.warning(f"  [{i}/{len(scan_paths)}] No OCR data for {filename}, skipping")
            continue

        page_number = get_scan_page_number(scan_path, i)
        logger.info(f"[{i}/{len(scan_paths)}] Gemini: {filename} (est. page {page_number})")

        result = process_page_with_gemini(
            scan_image_path=scan_path,
            ocr_page=ocr_page,
            page_number=page_number,
        )

        if result is not None:
            _save_llm_raw(scan_path, result)
            results[filename] = result
            if filename not in checkpoint["completed"]:
                checkpoint["completed"].append(filename)
            # Drop any stale "failed" entry now that we've succeeded.
            checkpoint["failed"] = [f for f in checkpoint["failed"] if f != filename]
        else:
            if filename not in checkpoint["failed"]:
                checkpoint["failed"].append(filename)

        save_checkpoint(checkpoint)

        # Rate limiting (skip delay after last page)
        if i < len(scan_paths):
            from pipeline.config import GEMINI_DELAY_SECONDS
            time.sleep(GEMINI_DELAY_SECONDS)

    _retry_failed_pages(scan_paths, ocr_results, results, checkpoint, logger)

    succeeded = len(results)
    failed_count = len(checkpoint.get("failed", []))
    logger.info(f"Gemini processing complete: {succeeded} succeeded, {failed_count} failed")

    _write_failures_aggregate()

    return results


def _retry_failed_pages(
    scan_paths: list[Path],
    ocr_results: dict[str, OcrPage],
    results: dict[str, dict],
    checkpoint: dict,
    logger: logging.Logger,
) -> None:
    """
    End-of-run auto-retry. Pass 1: same model. Pass 2: fallback model
    (typically more verbose-tolerant — usually it's max_output_tokens that
    bit us on the dense pages).
    """
    failed = list(checkpoint.get("failed", []))
    if not failed:
        return
    from pipeline.config import OPENROUTER_FALLBACK_MODEL
    logger.info(f"Retrying {len(failed)} failed pages...")
    scan_by_name = {p.name: p for p in scan_paths}

    def _retry_one(name: str) -> bool:
        scan_path = scan_by_name.get(name)
        ocr_page = ocr_results.get(name) if scan_path else None
        if not (scan_path and ocr_page):
            return False
        page_number = get_scan_page_number(scan_path, 0)
        result = process_page_with_gemini(
            scan_image_path=scan_path,
            ocr_page=ocr_page,
            page_number=page_number,
            max_retries=2,
        )
        if result is None:
            return False
        _save_llm_raw(scan_path, result)
        results[name] = result
        if name not in checkpoint["completed"]:
            checkpoint["completed"].append(name)
        checkpoint["failed"] = [f for f in checkpoint["failed"] if f != name]
        save_checkpoint(checkpoint)
        return True

    for name in list(failed):
        logger.info(f"  Retry (primary): {name}")
        if _retry_one(name):
            logger.info(f"  Retry succeeded for {name}")

    still_failed = list(checkpoint.get("failed", []))
    if still_failed and OPENROUTER_FALLBACK_MODEL:
        logger.info(
            f"Falling back to {OPENROUTER_FALLBACK_MODEL} for "
            f"{len(still_failed)} stubborn pages..."
        )
        init_gemini(model_override=OPENROUTER_FALLBACK_MODEL)
        for name in still_failed:
            logger.info(f"  Retry (fallback): {name}")
            if _retry_one(name):
                logger.info(f"  Fallback retry succeeded for {name}")
        init_gemini()


def stage_ocr_llm_pipelined(
    scan_paths: list[Path],
    force_reprocess: bool = False,
    strategy: str = None,
    device: str = None,
) -> tuple[dict[str, OcrPage], dict[str, dict]]:
    """
    Pipelined OCR + LLM. OCR runs on the main thread (the producer); a single
    background worker consumes each completed OCR page and calls the LLM.
    Wall time becomes ``max(OCR, LLM)`` instead of ``OCR + LLM``.

    Single LLM worker — preserves the existing rate-limit semantics
    (`GEMINI_DELAY_SECONDS` between calls). A multi-worker pool would need a
    shared rate limiter; that's deferred until LLM becomes the long pole on
    GPU OCR.

    Resume: pages with an existing final JSON or ``llm_raw`` sidecar are
    enqueued anyway so the worker can repopulate the in-memory results dict
    cheaply (no LLM call). The OCR call itself is cached per page on disk,
    so re-running is fast.
    """
    import threading
    import queue as _queue
    from pipeline.config import GEMINI_DELAY_SECONDS

    logger_ocr = logging.getLogger("ocr")
    logger_llm = logging.getLogger("gemini")
    logger_ocr.info(
        f"═══ Stages 2+3 (pipelined): Surya OCR + Gemini LLM ({len(scan_paths)} pages) ═══"
    )

    init_gemini()
    checkpoint = load_checkpoint()

    ocr_results: dict[str, OcrPage] = {}
    llm_results: dict[str, dict] = {}

    # Bounded queue: caps memory if LLM is slower than OCR (it usually is on GPU OCR;
    # opposite on CPU OCR — bounded queue makes the OCR producer block harmlessly).
    q: _queue.Queue = _queue.Queue(maxsize=8)
    consumer_errors: list[BaseException] = []
    counters = {"llm_calls": 0, "from_cache": 0}

    def consumer():
        while True:
            item = q.get()
            try:
                if item is None:
                    return
                scan_path, ocr_page = item
                filename = scan_path.name

                # 1. final JSON exists? Reuse.
                final_json = JSON_DIR / f"{scan_path.stem}.json"
                if not force_reprocess and final_json.exists():
                    try:
                        llm_results[filename] = json.loads(
                            final_json.read_text(encoding="utf-8")
                        )
                        counters["from_cache"] += 1
                        continue
                    except Exception as e:
                        logger_llm.warning(
                            f"  final JSON unreadable for {filename}, will redo: {e}"
                        )

                # 2. raw LLM cached?
                if not force_reprocess:
                    raw = _load_llm_raw(scan_path)
                    if raw is not None:
                        llm_results[filename] = raw
                        counters["from_cache"] += 1
                        if filename not in checkpoint["completed"]:
                            checkpoint["completed"].append(filename)
                            save_checkpoint(checkpoint)
                        continue

                # 3. call LLM.
                page_number = get_scan_page_number(scan_path, 0)
                logger_llm.info(f"LLM: {filename} (page {page_number})")
                result = process_page_with_gemini(
                    scan_image_path=scan_path,
                    ocr_page=ocr_page,
                    page_number=page_number,
                )
                counters["llm_calls"] += 1
                if result is not None:
                    _save_llm_raw(scan_path, result)
                    llm_results[filename] = result
                    if filename not in checkpoint["completed"]:
                        checkpoint["completed"].append(filename)
                    checkpoint["failed"] = [
                        f for f in checkpoint["failed"] if f != filename
                    ]
                else:
                    if filename not in checkpoint["failed"]:
                        checkpoint["failed"].append(filename)
                save_checkpoint(checkpoint)

                time.sleep(GEMINI_DELAY_SECONDS)
            except BaseException as e:
                consumer_errors.append(e)
                logger_llm.exception(f"LLM consumer error on item: {e}")
            finally:
                q.task_done()

    t = threading.Thread(target=consumer, name="llm-worker", daemon=True)
    t.start()

    try:
        for i, scan_path in enumerate(scan_paths, start=1):
            logger_ocr.info(f"[{i}/{len(scan_paths)}] OCR: {scan_path.name}")
            normalized = normalize_image(scan_path)
            ocr_page = run_ocr(normalized, scan_path.name, strategy=strategy, device=device)
            ocr_results[scan_path.name] = ocr_page
            logger_ocr.info(f"  → {len(ocr_page.all_words)} words detected")
            q.put((scan_path, ocr_page))
    finally:
        q.put(None)
        logger_llm.info("OCR producer done; waiting for LLM worker to drain queue...")
        t.join()

    if consumer_errors:
        # Surface the first error but continue to retry pass — partial progress
        # is already on disk.
        logger_llm.error(f"LLM worker had {len(consumer_errors)} errors; first: {consumer_errors[0]}")

    logger_llm.info(
        f"Pipelined OCR+LLM complete: {counters['llm_calls']} LLM calls, "
        f"{counters['from_cache']} pages reused from cache"
    )

    _retry_failed_pages(scan_paths, ocr_results, llm_results, checkpoint, logger_llm)
    _write_failures_aggregate()

    succeeded = len(llm_results)
    failed_count = len(checkpoint.get("failed", []))
    logger_llm.info(
        f"Gemini processing complete: {succeeded} succeeded, {failed_count} failed"
    )

    return ocr_results, llm_results


def stage_align_and_export(
    scan_paths: list[Path],
    ocr_results: dict[str, OcrPage],
    gemini_results: dict[str, dict],
) -> dict[str, dict]:
    """
    Stages 4-5: Align and export all pages.

    Each page is written to disk immediately (ALTO + JSON) so a crash here
    can't lose work. Combined indexes are rebuilt from the on-disk JSONs at
    the very end, which means even a partial run produces useful indexes.
    """
    logger = logging.getLogger("export")
    logger.info(f"═══ Stages 4-5: Alignment & Export ═══")

    aligned_results: dict[str, dict] = {}

    for i, scan_path in enumerate(scan_paths, start=1):
        filename = scan_path.name
        ocr_page = ocr_results.get(filename)
        gemini_result = gemini_results.get(filename)

        if not ocr_page:
            continue

        if gemini_result:
            aligned = align_page(gemini_result, ocr_page)
        else:
            aligned = {
                "page_number": i,
                "section": "unknown",
                "scan_file": filename,
                "dimensions": {"width": ocr_page.width, "height": ocr_page.height},
            }

        try:
            export_page_alto(ocr_page, aligned)
            export_page_json(ocr_page, aligned)
        except Exception as e:
            logger.error(f"  Export failed for {filename}: {e}")
            continue

        aligned_results[filename] = aligned

        if i % 50 == 0:
            logger.info(f"  Aligned/exported {i}/{len(scan_paths)} pages")

    # Build combined indexes from whatever JSONs are on disk (may include
    # results from previous partial runs we never had in memory).
    logger.info("Building combined indexes from disk...")
    on_disk = _load_all_aligned_from_disk()
    build_combined_indexes(on_disk)
    logger.info(f"Export complete: {len(aligned_results)} pages this run, {len(on_disk)} pages on disk total")
    return aligned_results


# Approx OpenRouter prices ($/1M tokens) for cost summary. Update as needed.
# These are used only for the post-run cost report; they don't affect billing.
LLM_PRICING_USD_PER_M = {
    "google/gemini-2.5-flash-lite":  {"input": 0.10, "output": 0.40},
    "google/gemini-2.5-flash":       {"input": 0.30, "output": 2.50},
    "google/gemini-2.0-flash-001":   {"input": 0.10, "output": 0.40},
    "google/gemini-2.0-flash-lite-001": {"input": 0.075, "output": 0.30},
    "openai/gpt-4o-mini":            {"input": 0.15, "output": 0.60},
}


def report_cost_summary(logger: logging.Logger) -> None:
    """Aggregate per-page usage sidecars into a cost report."""
    from pipeline.config import LLM_USAGE_DIR
    if not LLM_USAGE_DIR.exists():
        return
    by_model: dict[str, dict[str, int]] = {}
    pages = 0
    for f in LLM_USAGE_DIR.glob("*.json"):
        try:
            u = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = u.get("model", "unknown")
        agg = by_model.setdefault(m, {"prompt_tokens": 0, "completion_tokens": 0, "pages": 0})
        agg["prompt_tokens"] += u.get("prompt_tokens") or 0
        agg["completion_tokens"] += u.get("completion_tokens") or 0
        agg["pages"] += 1
        pages += 1
    if not pages:
        return
    logger.info("LLM cost summary (per usage sidecars):")
    grand_total = 0.0
    for m, agg in by_model.items():
        in_tok = agg["prompt_tokens"]
        out_tok = agg["completion_tokens"]
        price = LLM_PRICING_USD_PER_M.get(m)
        if price:
            cost = in_tok / 1e6 * price["input"] + out_tok / 1e6 * price["output"]
            grand_total += cost
            cost_s = f"${cost:.2f}"
        else:
            cost_s = "(no price for this model)"
        logger.info(
            f"  {m}: {agg['pages']} pages, in={in_tok:,} out={out_tok:,} → {cost_s}"
        )
    if grand_total:
        logger.info(f"  Grand total (estimated): ${grand_total:.2f}")


def report_sanity_check(logger: logging.Logger, total_pages: int) -> None:
    """Sanity-check final output: warn if entry count is suspiciously low."""
    on_disk = _load_all_aligned_from_disk()
    n_pages_with_data = len(on_disk)
    n_entries = 0
    for d in on_disk.values():
        for k in ("entries", "entities", "advertisements"):
            n_entries += len(d.get(k, []))
        for s in d.get("streets", []):
            n_entries += len(s.get("entries", []))
        for o in d.get("occupations", []):
            n_entries += len(o.get("entries", []))
    logger.info(f"Sanity check: {n_pages_with_data}/{total_pages} pages with data, {n_entries} entries total")
    if n_pages_with_data < 0.95 * total_pages:
        logger.warning(
            f"  Only {n_pages_with_data}/{total_pages} pages exported — "
            "investigate failed pages in output/checkpoint.json"
        )
    if n_pages_with_data and n_entries / n_pages_with_data < 5:
        logger.warning(
            f"  Average {n_entries/n_pages_with_data:.1f} entries/page — much lower than expected. "
            "Check LLM output quality or model laziness (try a different model)."
        )


def _write_failures_aggregate() -> None:
    """
    Aggregate per-page failure sidecars in `output/failures/` into a single
    `output/failures.json` keyed by scan_file. Removes the aggregate when
    no per-page failure files remain.
    """
    aggregate = OUTPUT_DIR / "failures.json"
    if not FAILURES_DIR.exists():
        if aggregate.exists():
            try: aggregate.unlink()
            except OSError: pass
        return
    items = []
    for f in sorted(FAILURES_DIR.glob("*.json")):
        try:
            items.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not items:
        if aggregate.exists():
            try: aggregate.unlink()
            except OSError: pass
        return
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(items),
        "by_error_class": _count_by(items, "error_class"),
        "by_section": _count_by(items, "section_type"),
        "failures": items,
    }
    aggregate.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.getLogger("gemini").info(
        f"Wrote {aggregate} with {len(items)} failure records"
    )


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        v = it.get(key) or "unknown"
        out[v] = out.get(v, 0) + 1
    return out


def _load_all_aligned_from_disk() -> dict[str, dict]:
    """Load every per-page aligned JSON in JSON_DIR for combined-index build."""
    out: dict[str, dict] = {}
    for f in JSON_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            scan_file = data.get("scan_file") or f"{f.stem}.jpg"
            out[scan_file] = data
        except Exception:
            continue
    return out


# ── Main entry point ──────────────────────────────────────────────────────────


def parse_page_range(range_str: str) -> tuple[int, int]:
    """Parse a page range string like '1-10' or '5'."""
    parts = range_str.split("-")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    elif len(parts) == 1:
        n = int(parts[0])
        return n, n
    else:
        raise ValueError(f"Invalid page range: {range_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Groningen Adresboek 1926 — Data Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pages",
        type=str,
        help="Page range to process (e.g., '1-10' or '5'). Default: all pages.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing (ignore checkpoints).",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only run Tesseract OCR, skip Gemini processing.",
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Test mode: process only the first N pages.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging.",
    )
    parser.add_argument(
        "--scans-dir",
        type=str,
        help=f"Override scans directory (default: {SCANS_DIR})",
    )
    parser.add_argument(
        "--preflight",
        type=int,
        metavar="N",
        help=(
            "Pre-flight cost projection: process N spread-out sample pages, "
            "then report measured tokens/cost and the projected full-book cost. "
            "Does not change checkpoints."
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="OCR strategy: 'auto', 'surya', or 'loghi'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="OCR device: 'cpu', 'cuda', 'mps', 'directml', or 'auto'.",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    scans_dir = Path(args.scans_dir) if args.scans_dir else SCANS_DIR
    
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║  Groningen Adresboek 1926 — Data Extraction Pipeline      ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    logger.info(f"Scans directory: {scans_dir}")

    # Stage 1: Discover scans
    logger.info("═══ Stage 1: Discovering scans ═══")
    scan_paths = discover_scans(scans_dir)
    
    if not scan_paths:
        logger.error(f"No scan files found in {scans_dir}")
        logger.info("Place your JPEG/TIFF scans in the scans/ directory and try again.")
        sys.exit(1)

    # Apply page range filter
    if args.pages:
        start, end = parse_page_range(args.pages)
        scan_paths = scan_paths[start - 1:end]  # 1-indexed to 0-indexed
        logger.info(f"Filtered to pages {start}-{end}: {len(scan_paths)} scans")

    if args.test:
        scan_paths = scan_paths[:args.test]
        logger.info(f"Test mode: processing only {len(scan_paths)} pages")

    if args.preflight:
        run_preflight(scan_paths, args.preflight, logger)
        return

    logger.info(f"Total scans to process: {len(scan_paths)}")

    # Stages 2+3: OCR + LLM. Pipelined by default so wall time is
    # max(OCR, LLM) instead of OCR+LLM. Falls back to sequential `stage_ocr`
    # in --ocr-only mode.
    if args.ocr_only:
        t_start = time.time()
        ocr_results = stage_ocr(scan_paths, strategy=args.strategy, device=args.device)
        t_ocr = time.time() - t_start
        logger.info(f"OCR took {t_ocr:.1f}s ({t_ocr/len(scan_paths):.1f}s/page)")
        logger.info("Skipping Gemini processing (--ocr-only mode)")
        gemini_results: dict[str, dict] = {}
    else:
        t_start = time.time()
        ocr_results, gemini_results = stage_ocr_llm_pipelined(
            scan_paths, force_reprocess=args.reprocess,
            strategy=args.strategy, device=args.device
        )
        t_pipeline = time.time() - t_start
        logger.info(f"OCR+LLM (pipelined) took {t_pipeline:.1f}s")

    # Stages 4-5: Alignment & Export
    t_start = time.time()
    aligned_results = stage_align_and_export(scan_paths, ocr_results, gemini_results)
    t_export = time.time() - t_start
    logger.info(f"Alignment & export took {t_export:.1f}s")

    # Summary
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║  Pipeline Complete!                                       ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")
    logger.info(f"  Pages processed: {len(aligned_results)}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"  hOCR files: {OUTPUT_DIR / 'hocr'}")
    logger.info(f"  ALTO XML files: {OUTPUT_DIR / 'alto'}")
    logger.info(f"  JSON files: {OUTPUT_DIR / 'json'}")
    logger.info(f"  Combined indexes: {OUTPUT_DIR / 'combined'}")
    
    # Print some stats from combined indexes
    search_index_path = COMBINED_DIR / "search_index.json"
    if search_index_path.exists():
        with open(search_index_path, "r", encoding="utf-8") as f:
            search_data = json.load(f)
        logger.info(f"  Total entries extracted: {len(search_data)}")

    # Cost summary + sanity check
    report_cost_summary(logger)
    report_sanity_check(logger, total_pages=len(scan_paths))


def run_preflight(all_scans: list[Path], n: int, logger: logging.Logger) -> None:
    """
    Pre-flight: run OCR + LLM on `n` evenly-spaced sample pages, capture token
    usage, project full-book cost. Doesn't touch the main checkpoint.
    """
    if n <= 0:
        logger.error("preflight count must be > 0")
        return

    # Pick evenly-spaced samples across the whole list so we hit different sections.
    if n >= len(all_scans):
        samples = list(all_scans)
    else:
        step = len(all_scans) / n
        samples = [all_scans[int(i * step)] for i in range(n)]

    logger.info(f"═══ Pre-flight: {len(samples)} sample pages ═══")
    for s in samples:
        logger.info(f"  sample: {s.name}")

    # OCR
    ocr_results = stage_ocr(samples)

    # LLM (no checkpoint mutation in preflight; reuse stage_gemini but tag a
    # separate usage subdir would be ideal — for the pilot we accept usage
    # files going into the normal LLM_USAGE_DIR and the sample-only stats are
    # computed below from those files' mtimes).
    from pipeline.config import LLM_USAGE_DIR
    LLM_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in LLM_USAGE_DIR.glob("*.json")}

    init_gemini()
    for s in samples:
        ocr_page = ocr_results.get(s.name)
        if not ocr_page:
            continue
        page_number = get_scan_page_number(s, 0)
        from pipeline.llm import process_page_with_gemini
        process_page_with_gemini(scan_image_path=s, ocr_page=ocr_page, page_number=page_number)

    # Aggregate usage from JUST the sidecars added during preflight.
    after_files = list(LLM_USAGE_DIR.glob("*.json"))
    new_files = [f for f in after_files if f.name not in before]
    in_total = out_total = 0
    model = None
    for f in new_files:
        try:
            u = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        in_total += u.get("prompt_tokens") or 0
        out_total += u.get("completion_tokens") or 0
        model = u.get("model", model)

    n_sampled = len(new_files) or 1
    avg_in = in_total / n_sampled
    avg_out = out_total / n_sampled
    full_pages = len(all_scans)
    proj_in = avg_in * full_pages
    proj_out = avg_out * full_pages

    price = LLM_PRICING_USD_PER_M.get(model or "")
    logger.info("─── Pre-flight results ───")
    logger.info(f"  Model:       {model}")
    logger.info(f"  Sampled:     {n_sampled} pages")
    logger.info(f"  Avg tokens:  in={avg_in:,.0f}, out={avg_out:,.0f}")
    logger.info(f"  Full book:   {full_pages} pages")
    logger.info(f"  Projected:   in={proj_in:,.0f}, out={proj_out:,.0f}")
    if price:
        sample_cost = (in_total / 1e6 * price["input"]) + (out_total / 1e6 * price["output"])
        full_cost = (proj_in / 1e6 * price["input"]) + (proj_out / 1e6 * price["output"])
        logger.info(f"  Sample cost: ${sample_cost:.4f}")
        logger.info(f"  Projected full-book cost: ${full_cost:.2f}")
    else:
        logger.info("  (no price entry for this model — add to LLM_PRICING_USD_PER_M to project cost)")


if __name__ == "__main__":
    main()
