"""
OCR module — Surya backend.

Runs Surya OCR (https://github.com/VikParuchuri/surya) on scan images to
produce line- and word-level bounding boxes. The data shape (OcrPage,
OcrBlock, OcrLine, OcrWord) is preserved so downstream consumers
(align, llm, alto_export, json_export) work unchanged.

Per-page OCR results are cached as JSON next to the hOCR debug dump so
re-runs (e.g. iterating on Gemini prompts) don't repeat the expensive
OCR step.

Two post-OCR passes apply to both fresh OCR and cached reads:

- **Column reordering**: Surya emits text lines in y-coordinate order,
  which on a two-column page produces a left-right-left-right zigzag. We
  detect a column gap from the line x-centers and re-emit lines as
  full-left-then-full-right. Word IDs are renumbered after reordering.

- **Word-bbox repair**: occasionally Surya gives every word on a line
  the same bounding box (the line's). We detect that and split it
  proportionally by character count.
"""

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from pipeline.config import HOCR_DIR, OCR_STRATEGY, OCR_DEVICE, LOGHI_MODEL_PATH
from pipeline.classifier import classify_page, PageType

logger = logging.getLogger(__name__)

OCR_CACHE_SCHEMA_VERSION = 2  # bump when cache shape changes


# ── Data classes (preserved from Tesseract version) ──────────────────────────


@dataclass
class OcrWord:
    """A single word with its bounding box."""
    id: str          # Unique word ID (e.g., "w_0001")
    text: str        # Recognized text
    bbox: list[int]  # [x1, y1, x2, y2] in pixels
    confidence: int  # 0–100
    line_id: str     # Parent line ID


@dataclass
class OcrLine:
    """A line of text."""
    id: str
    bbox: list[int]
    words: list[OcrWord] = field(default_factory=list)


@dataclass
class OcrBlock:
    """A text block (region)."""
    id: str
    bbox: list[int]
    lines: list[OcrLine] = field(default_factory=list)


@dataclass
class OcrPage:
    """Full OCR result for a single page."""
    scan_file: str
    width: int
    height: int
    blocks: list[OcrBlock] = field(default_factory=list)
    hocr_raw: str = ""

    @property
    def all_words(self) -> list[OcrWord]:
        words = []
        for block in self.blocks:
            for line in block.lines:
                words.extend(line.words)
        return words

    @property
    def word_index(self) -> dict[str, OcrWord]:
        return {w.id: w for w in self.all_words}

    @property
    def line_index(self) -> dict[str, OcrLine]:
        """Index of all lines by their ID."""
        lines = {}
        for block in self.blocks:
            for line in block.lines:
                lines[line.id] = line
        return lines

    def to_numbered_word_list(self) -> str:
        """Format words as a numbered list for the LLM prompt."""
        return "\n".join(f"{w.id}: {w.text}" for w in self.all_words)


# ── Surya predictors (lazy, singletons) ──────────────────────────────────────

_recognition_predictor = None
_detection_predictor = None
_foundation_predictor = None


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


def _get_predictors(device="auto"):
    """Lazy-init Surya predictors so importing this module is cheap."""
    global _recognition_predictor, _detection_predictor, _foundation_predictor
    if _recognition_predictor is None:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        
        target_device = _get_device(device)
        logger.info(f"Loading Surya models on {target_device}...")
        
        _foundation_predictor = FoundationPredictor(device=target_device)
        _recognition_predictor = RecognitionPredictor(foundation_predictor=_foundation_predictor)
        _detection_predictor = DetectionPredictor(device=target_device)
        logger.info("Surya models loaded.")
    return _detection_predictor, _recognition_predictor


# ── Post-OCR passes: column reorder + bbox repair ────────────────────────────


def _detect_column_gap(line_centers: list[float], page_width: int) -> float | None:
    """
    Find a vertical gutter between two columns from line x-centers.

    Strategy: sort the centers, look at consecutive gaps. If the largest gap
    is wide (>= 8% of page width) and falls roughly in the middle third of
    the page, we treat the page as two-column and return the gap midpoint.
    Otherwise return None (single column or unclassifiable).
    """
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
    # Sanity: each side must hold a meaningful number of lines
    left_count = sum(1 for c in line_centers if c < gap_mid)
    right_count = sum(1 for c in line_centers if c >= gap_mid)
    if min(left_count, right_count) < max(3, 0.15 * len(line_centers)):
        return None
    return gap_mid


def _reorder_columns(page: "OcrPage") -> "OcrPage":
    """
    If the page is two-column, rewrite the line/word ordering so all left-column
    lines come first (top → bottom), then all right-column lines (top → bottom).
    Word IDs are renumbered sequentially. Block grouping is rebuilt to match.
    """
    flat_lines: list[OcrLine] = [ln for b in page.blocks for ln in b.lines]
    if not flat_lines:
        return page

    centers = [(ln.bbox[0] + ln.bbox[2]) / 2 for ln in flat_lines]
    gap = _detect_column_gap(centers, page.width)
    if gap is None:
        return page  # single-column, nothing to do

    left = [ln for ln, c in zip(flat_lines, centers) if c < gap]
    right = [ln for ln, c in zip(flat_lines, centers) if c >= gap]
    left.sort(key=lambda ln: ln.bbox[1])
    right.sort(key=lambda ln: ln.bbox[1])
    ordered = left + right

    # Renumber lines and words; rebuild blocks 50-lines-each like before.
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


def _enclose_lines(lines: list["OcrLine"]) -> list[int]:
    return [
        min(ln.bbox[0] for ln in lines),
        min(ln.bbox[1] for ln in lines),
        max(ln.bbox[2] for ln in lines),
        max(ln.bbox[3] for ln in lines),
    ]


def _repair_word_bboxes(page: "OcrPage") -> "OcrPage":
    """
    Defensive: when Surya collapses several words on a line into a single
    cluster bbox (a sub-range of the line, not the whole line), redistribute
    proportionally by character count across the *line* bbox so the leftmost
    and rightmost characters land where they actually appear in the image.

    Detection: count words that share their rounded (10-px grid) bbox with at
    least one other word on the same line. If a *majority* of words on the
    line participate in some shared cluster, the line is degenerate and we
    repair. Otherwise we leave it alone.

    This catches the dominant Surya failure mode on this dataset, where N-1
    words on a row collapse to one cluster and the trailing word (often a
    house-number or "tel.NNN") gets a different, wider bbox — which used to
    lift diversity above the old 30% threshold and skip repair, leaving the
    leftmost ~150–400 px of every line uncovered.

    Redistribution uses ``ln.bbox`` (the line bbox), which Surya reports
    correctly even when per-word bboxes collapse. The y-range comes from the
    words themselves; line-bbox y is often looser (includes line spacing).
    Idempotent: after a clean repair, words have distinct bboxes and the
    detection threshold no longer trips.
    """
    repaired = 0
    for block in page.blocks:
        for ln in block.lines:
            if len(ln.words) < 2:
                continue

            rounded = [tuple(round(v / 10) * 10 for v in w.bbox) for w in ln.words]
            counts = Counter(rounded)
            shared = sum(1 for r in rounded if counts[r] > 1)
            if shared * 2 < len(ln.words):
                continue  # word bboxes are distinct enough

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


# ── Main OCR runner ──────────────────────────────────────────────────────────


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
    """
    Load a cached OcrPage and apply post-OCR passes (column reorder + bbox
    repair). The passes are applied even on schema_version=1 caches (the
    original pilot run) so existing data benefits from the fixes without
    re-OCR.
    """
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

    # Apply post-OCR passes regardless of schema version. They're idempotent
    # on already-clean data (single-column + per-word bboxes).
    page = _repair_word_bboxes(page)
    page = _reorder_columns(page)
    return page


def run_loghi_recognition(image: Image.Image, text_lines: list) -> list[str]:
    """
    Call Loghi HTR as a separate subprocess.
    
    This keeps TensorFlow and PyTorch dependencies separate.
    Placeholder implementation: would crop line images and call the CLI.
    """
    if not LOGHI_MODEL_PATH:
        logger.warning("  Loghi selected but LOGHI_MODEL_PATH not set. Falling back to '???'")
        return ["???" for _ in text_lines]
    
    logger.info(f"  Calling Loghi HTR (model: {LOGHI_MODEL_PATH})...")
    # Subprocess logic would go here
    return ["HTR_RESULT" for _ in text_lines]


def run_ocr(
    image: Image.Image,
    scan_filename: str,
    save_debug: bool = True,
    use_cache: bool = True,
    strategy: str = None,
    device: str = None,
) -> OcrPage:
    """
    Run OCR on an image using the specified strategy (Surya, Loghi, or Auto).
    """
    strategy = strategy or OCR_STRATEGY
    device = device or OCR_DEVICE

    if use_cache:
        cached = _load_cache(scan_filename)
        if cached is not None:
            logger.info(f"  Loaded cached OCR for {scan_filename} ({len(cached.all_words)} words)")
            return cached

    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Classification (if auto)
    if strategy == "auto":
        page_type = classify_page(image)
        actual_engine = "loghi" if page_type == PageType.HANDWRITTEN else "surya"
        logger.info(f"  Auto-strategy: detected {page_type.value} -> using {actual_engine}")
    else:
        actual_engine = strategy

    logger.info(f"Running {actual_engine} OCR on {scan_filename}...")
    det, rec = _get_predictors(device=device)

    # 2. Detection (Always Surya for layout analysis)
    # Even if we use Loghi for recognition, we use Surya for bounding boxes.
    predictions = rec(
        [image],
        task_names=["ocr_with_boxes"],
        det_predictor=det,
        return_words=True,
        sort_lines=True,
    )

    if not predictions:
        logger.warning(f"  OCR returned no predictions for {scan_filename}")
        return OcrPage(scan_file=scan_filename, width=image.width, height=image.height)

    result = predictions[0]
    text_lines = getattr(result, "text_lines", []) or []

    # 3. Recognition Override (if Loghi)
    if actual_engine == "loghi":
        loghi_texts = run_loghi_recognition(image, text_lines)
        for tl, lt in zip(text_lines, loghi_texts):
            tl.text = lt
            # Naive word split to keep ID mapping if possible
            if hasattr(tl, "words") and tl.words:
                words = lt.split()
                if len(words) == len(tl.words):
                    for w, t in zip(tl.words, words):
                        w.text = t

    page = OcrPage(
        scan_file=scan_filename,
        width=image.width,
        height=image.height,
    )

    word_counter = 0
    line_counter = 0
    block_counter = 0

    for tl in text_lines:
        line_text = (getattr(tl, "text", "") or "").strip()
        line_bbox_raw = getattr(tl, "bbox", None)
        if not line_text or not line_bbox_raw:
            continue
        line_bbox = [int(v) for v in line_bbox_raw]
        line_conf = float(getattr(tl, "confidence", 0.0) or 0.0)
        words_raw = getattr(tl, "words", None) or []

        line_counter += 1
        line_id = f"line_{line_counter:04d}"
        line_obj = OcrLine(id=line_id, bbox=line_bbox)

        for w in words_raw:
            word_text = (getattr(w, "text", "") or "").strip()
            word_bbox_raw = getattr(w, "bbox", None)
            if not word_text or not word_bbox_raw:
                continue
            word_bbox = [int(v) for v in word_bbox_raw]
            word_conf = float(getattr(w, "confidence", line_conf) or line_conf)
            word_counter += 1
            word_id = f"w_{word_counter:04d}"
            line_obj.words.append(OcrWord(
                id=word_id,
                text=word_text,
                bbox=word_bbox,
                confidence=int(round(word_conf * 100)),
                line_id=line_id,
            ))

        if not line_obj.words:
            continue

        # Group lines into blocks (50 lines per block, matching old behavior)
        if not page.blocks or len(page.blocks[-1].lines) >= 50:
            block_counter += 1
            page.blocks.append(OcrBlock(
                id=f"block_{block_counter}",
                bbox=line_bbox,
            ))
        page.blocks[-1].lines.append(line_obj)

    # Recompute block bboxes to enclose their lines
    for block in page.blocks:
        if block.lines:
            block.bbox = _enclose([ln.bbox for ln in block.lines])

    logger.info(f"  Surya: {word_counter} words, {line_counter} lines, {len(page.blocks)} blocks")

    # Post-OCR passes (column reorder + bbox repair). Apply BEFORE caching
    # so the cache reflects the corrected ordering.
    page = _repair_word_bboxes(page)
    page = _reorder_columns(page)

    if save_debug:
        _save_debug_dump(page)
    if use_cache:
        _save_cache(page)

    return page


def _enclose(bboxes: list[list[int]]) -> list[int]:
    if not bboxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _save_debug_dump(page: OcrPage) -> None:
    """Save a simple per-word text dump for debugging."""
    HOCR_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(page.scan_file).stem
    out = HOCR_DIR / f"{stem}.words.txt"
    lines = [f"# {page.scan_file} — {page.width}×{page.height}"]
    for w in page.all_words:
        x1, y1, x2, y2 = w.bbox
        lines.append(f"{w.id}\t{x1},{y1},{x2},{y2}\tconf={w.confidence}\t{w.text}")
    out.write_text("\n".join(lines), encoding="utf-8")


# ── Backward-compat alias ────────────────────────────────────────────────────


def run_tesseract(image: Image.Image, scan_filename: str, save_hocr: bool = True) -> OcrPage:
    """Deprecated alias kept for any stale callers; delegates to run_ocr()."""
    return run_ocr(image, scan_filename, save_debug=save_hocr)
