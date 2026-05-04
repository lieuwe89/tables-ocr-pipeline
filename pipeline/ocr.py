"""
OCR module — Orchestrator.

Manages the pipeline from image to structured OcrPage. 
Uses Surya for layout/detection and supports pluggable backends (Surya, Loghi)
for text recognition.
"""

import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from PIL import Image

from pipeline.config import HOCR_DIR, OCR_STRATEGY, OCR_DEVICE
from pipeline.classifier import classify_page, PageType

# Internal imports
from .types import OcrPage, OcrBlock, OcrLine, OcrWord
from .surya_backend import SuryaBackend
from .loghi_backend import LoghiBackend

logger = logging.getLogger(__name__)

OCR_CACHE_SCHEMA_VERSION = 2  # bump when cache shape changes

# ── Surya layout predictors (lazy, singletons) ───────────────────────────────

_detection_predictor = None

def _get_device(override=None):
    """Detect the best available device for PyTorch."""
    if override and override != "auto":
        return override

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _get_layout_predictor(device="auto"):
    """Lazy-init Surya detection predictor."""
    global _detection_predictor
    if _detection_predictor is None:
        from surya.detection import DetectionPredictor
        target_device = _get_device(device)
        logger.info(f"Loading Surya layout model on {target_device}...")
        _detection_predictor = DetectionPredictor(device=target_device)
    return _detection_predictor

# ── Post-OCR passes: column reorder + bbox repair ────────────────────────────

def _detect_column_gap(line_centers: list[float], page_width: int) -> float | None:
    if len(line_centers) < 6:
        return None
    sorted_centers = sorted(line_centers)
    gaps = [
        (sorted_centers[i + 1] - sorted_centers[i], sorted_centers[i], sorted_centers[i + 1])
        for i in range(len(sorted_centers) - 1)
    ]
    biggest = max(gaps, key=lambda g: g[0])
    gap_width, lo, hi = biggest
    gap_mid = (lo + hi) / 2
    if gap_width < 0.08 * page_width:
        return None
    if not (0.30 * page_width <= gap_mid <= 0.70 * page_width):
        return None
    left_count = sum(1 for c in line_centers if c < gap_mid)
    right_count = sum(1 for c in line_centers if c >= gap_mid)
    if min(left_count, right_count) < max(3, 0.15 * len(line_centers)):
        return None
    return gap_mid

def _reorder_columns(page: OcrPage) -> OcrPage:
    flat_lines: list[OcrLine] = [ln for b in page.blocks for ln in b.lines]
    if not flat_lines:
        return page

    centers = [(ln.bbox[0] + ln.bbox[2]) / 2 for ln in flat_lines]
    gap = _detect_column_gap(centers, page.width)
    if gap is None:
        return page

    left = [ln for ln, c in zip(flat_lines, centers) if c < gap]
    right = [ln for ln, c in zip(flat_lines, centers) if c >= gap]
    left.sort(key=lambda ln: ln.bbox[1])
    right.sort(key=lambda ln: ln.bbox[1])
    ordered = left + right

    new_blocks: list[OcrBlock] = []
    word_n = 0
    for li, ln in enumerate(ordered, start=1):
        new_line_id = f"line_{li:04d}"
        ln.id = new_line_id
        for w in ln.words:
            word_n += 1
            w.id = f"w_{word_n:04d}"
            w.line_id = new_line_id
        if not new_blocks or len(new_blocks[-1].lines) >= 50:
            new_blocks.append(OcrBlock(id=f"block_{len(new_blocks)+1}", bbox=list(ln.bbox)))
        new_blocks[-1].lines.append(ln)

    for b in new_blocks:
        if b.lines:
            b.bbox = _enclose_lines(b.lines)

    page.blocks = new_blocks
    return page

def _enclose_lines(lines: list[OcrLine]) -> list[int]:
    return [
        min(ln.bbox[0] for ln in lines),
        min(ln.bbox[1] for ln in lines),
        max(ln.bbox[2] for ln in lines),
        max(ln.bbox[3] for ln in lines),
    ]

def _repair_word_bboxes(page: OcrPage) -> OcrPage:
    repaired = 0
    for block in page.blocks:
        for ln in block.lines:
            if len(ln.words) < 2:
                continue

            rounded = [tuple(round(v / 10) * 10 for v in w.bbox) for w in ln.words]
            counts = Counter(rounded)
            shared = sum(1 for r in rounded if counts[r] > 1)
            if shared * 2 < len(ln.words):
                continue

            lx1, _, lx2, _ = ln.bbox
            wy1 = min(w.bbox[1] for w in ln.words)
            wy2 = max(w.bbox[3] for w in ln.words)
            span = max(1, lx2 - lx1)
            total_chars = sum(max(1, len(w.text)) for w in ln.words) + (len(ln.words) - 1)
            cursor = 0
            for w in ln.words:
                w_chars = max(1, len(w.text))
                wx1 = lx1 + int(cursor / total_chars * span)
                wx2 = lx1 + int((cursor + w_chars) / total_chars * span)
                w.bbox = [wx1, wy1, wx2, wy2]
                cursor += w_chars + 1
            repaired += 1
    if repaired:
        logger.debug(f"  Repaired bboxes on {repaired} lines for {page.scan_file}")
    return page

# ── Caching ──────────────────────────────────────────────────────────────────

def _cache_path(scan_filename: str) -> Path:
    return HOCR_DIR / f"{Path(scan_filename).stem}.ocr.json"

def _save_cache(page: OcrPage) -> None:
    HOCR_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": OCR_CACHE_SCHEMA_VERSION,
        "scan_file": page.scan_file,
        "width": page.width,
        "height": page.height,
        "blocks": [asdict(b) for b in page.blocks],
    }
    target = _cache_path(page.scan_file)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)

def _load_cache(scan_filename: str) -> OcrPage | None:
    p = _cache_path(scan_filename)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  Cache for {scan_filename} unreadable ({e}); treating as miss")
        return None

    blocks = []
    for b in data.get("blocks", []):
        lines = []
        for ln in b.get("lines", []):
            words = [OcrWord(**w) for w in ln.get("words", [])]
            lines.append(OcrLine(id=ln["id"], bbox=ln["bbox"], words=words))
        blocks.append(OcrBlock(id=b["id"], bbox=b["bbox"], lines=lines))
    page = OcrPage(
        scan_file=data["scan_file"],
        width=data["width"],
        height=data["height"],
        blocks=blocks,
    )
    page = _repair_word_bboxes(page)
    page = _reorder_columns(page)
    return page

# ── Backend Factory ──────────────────────────────────────────────────────────

def _get_recognition_backend(strategy: str):
    if strategy == "loghi":
        return LoghiBackend()
    return SuryaBackend()

# ── Main OCR runner ──────────────────────────────────────────────────────────

def run_ocr(
    image: Image.Image,
    scan_filename: str,
    save_debug: bool = True,
    use_cache: bool = True,
    strategy: str = None,
    device: str = None,
    image_path: Path | None = None,
) -> OcrPage:
    strategy = strategy or OCR_STRATEGY
    device = device or OCR_DEVICE

    if use_cache:
        cached = _load_cache(scan_filename)
        if cached is not None:
            logger.info(f"  Loaded cached OCR for {scan_filename} ({len(cached.all_words)} words)")
            return cached

    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Classification & Backend Selection
    if strategy == "auto":
        page_type = classify_page(image, image_path=image_path)
        # HANDWRITTEN or MIXED should use Loghi (HTR)
        actual_strategy = "loghi" if page_type in (PageType.HANDWRITTEN, PageType.MIXED) else "surya"
        logger.info(f"  Auto-strategy: detected {page_type.value} -> using {actual_strategy}")
    else:
        actual_strategy = strategy

    backend = _get_recognition_backend(actual_strategy)

    # 2. Layout Detection (Always Surya)
    # We use Surya's layout to get text lines/regions
    from surya.detection import batch_detection
    det_predictor = _get_layout_predictor(device=device)
    det_results = batch_detection([image], det_predictor)
    
    if not det_results:
        logger.warning(f"  Layout detection returned no results for {scan_filename}")
        return OcrPage(scan_file=scan_filename, width=image.width, height=image.height)
        
    # Surya's detection results contain text lines with bboxes
    text_lines = det_results[0].bboxes

    # 3. Text Recognition via Pluggable Backend
    logger.info(f"  Recognizing text using {backend.__class__.__name__} on {device}...")
    recognized_texts = backend.recognize(image, text_lines, device=device)

    # 4. Assembly into structured OcrPage
    page = OcrPage(
        scan_file=scan_filename,
        width=image.width,
        height=image.height,
    )

    word_counter = 0
    line_counter = 0
    block_counter = 0

    for tl_bbox, text in zip(text_lines, recognized_texts):
        line_bbox = [int(v) for v in tl_bbox]
        line_counter += 1
        line_id = f"line_{line_counter:04d}"
        line_obj = OcrLine(id=line_id, bbox=line_bbox)

        # Build words (either from recognition or naive split)
        words = text.split()
        for word_text in words:
            word_counter += 1
            word_id = f"w_{word_counter:04d}"
            # Naive word bbox for now (equally spaced)
            # Surya backend actually does better but we'll refine this later
            line_obj.words.append(OcrWord(
                id=word_id,
                text=word_text,
                bbox=line_bbox, # Placeholder, _repair_word_bboxes will fix this
                confidence=100,
                line_id=line_id,
            ))

        if not line_obj.words:
            continue

        if not page.blocks or len(page.blocks[-1].lines) >= 50:
            block_counter += 1
            page.blocks.append(OcrBlock(id=f"block_{block_counter}", bbox=line_bbox))
        page.blocks[-1].lines.append(line_obj)

    for block in page.blocks:
        if block.lines:
            block.bbox = _enclose([ln.bbox for ln in block.lines])

    # 5. Post-processing & Caching
    page = _repair_word_bboxes(page)
    page = _reorder_columns(page)

    if save_debug:
        _save_debug_dump(page)
    if use_cache:
        _save_cache(page)

    return page

def _enclose(bboxes: list[list[int]]) -> list[int]:
    if not bboxes: return [0, 0, 0, 0]
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes)]

def _save_debug_dump(page: OcrPage) -> None:
    HOCR_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(page.scan_file).stem
    out = HOCR_DIR / f"{stem}.words.txt"
    lines = [f"# {page.scan_file} — {page.width}×{page.height}"]
    for w in page.all_words:
        x1, y1, x2, y2 = w.bbox
        lines.append(f"{w.id}\t{x1},{y1},{x2},{y2}\tconf={w.confidence}\t{w.text}")
    out.write_text("\n".join(lines), encoding="utf-8")

def run_tesseract(image: Image.Image, scan_filename: str, save_hocr: bool = True) -> OcrPage:
    return run_ocr(image, scan_filename, save_debug=save_hocr)
