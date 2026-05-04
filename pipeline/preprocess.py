"""
Pre-processing module for scan images.

Handles:
- Loading JPEG/TIFF images
- Normalizing DPI to 300 (Tesseract's sweet spot)
- Converting to grayscale for OCR (keeping color originals for display)
- Layout analysis to detect page structure (columns, ads, etc.)

NOTE: Multipage PDF support may be added in a future version.
"""

import logging
from pathlib import Path
from typing import Iterator

import pytesseract
from PIL import Image

from pipeline.config import (
    SCANS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    TARGET_DPI,
    TESSERACT_LANG,
    parse_scan_filename,
    scan_number_to_page_number,
)

logger = logging.getLogger(__name__)


def discover_scans(scans_dir: Path = SCANS_DIR) -> list[Path]:
    """
    Discover all scan files in the scans directory, sorted by filename.
    
    Only JPEG, PNG, and TIFF images are supported.
    Files are sorted by stem name, which ensures sequential scan order
    given the archiveNumber_recordNumber_scanNumber naming convention.
    
    Returns a sorted list of file paths.
    """
    files = []
    for f in scans_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(f)

    files.sort(key=lambda p: p.stem)
    logger.info(f"Discovered {len(files)} scan files in {scans_dir}")
    return files


def normalize_image(image_path: Path) -> Image.Image:
    """
    Load and normalize an image for OCR processing.
    
    - Ensures consistent DPI (TARGET_DPI)
    - Converts to grayscale for better OCR performance
    - Does NOT modify the original file
    
    Returns a PIL Image ready for Tesseract.
    """
    img = Image.open(image_path)

    # Get current DPI (default to 72 if not set in metadata)
    dpi = img.info.get("dpi", (72, 72))
    if isinstance(dpi, tuple):
        current_dpi = dpi[0]
    else:
        current_dpi = dpi

    # Rescale if DPI differs significantly from target
    if current_dpi > 0 and abs(current_dpi - TARGET_DPI) > 10:
        scale_factor = TARGET_DPI / current_dpi
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        logger.debug(
            f"  Rescaled {image_path.name}: {current_dpi} DPI → {TARGET_DPI} DPI "
            f"({img.width}×{img.height})"
        )

    # Convert to grayscale for OCR
    if img.mode != "L":
        img = img.convert("L")

    return img


def get_image_dimensions(image_path: Path) -> dict:
    """
    Get the original (color) image dimensions for bounding box reference.
    
    Returns dict with 'width', 'height', 'dpi'.
    """
    with Image.open(image_path) as img:
        dpi = img.info.get("dpi", (300, 300))
        if isinstance(dpi, tuple):
            dpi_val = dpi[0]
        else:
            dpi_val = dpi
        return {
            "width": img.width,
            "height": img.height,
            "dpi": dpi_val,
        }


# ── Layout Analysis ──────────────────────────────────────────────────────────


def analyze_layout(image: Image.Image, scan_filename: str) -> dict:
    """
    Run a pre-emptive layout analysis on a scan image using Tesseract.
    
    This detects the page structure BEFORE doing full OCR, helping to:
    - Identify two-column vs single-column layouts
    - Detect orientation issues
    - Classify page type (text-heavy, ad, table, etc.)
    - Determine the best PSM mode for OCR
    
    Returns a dict with layout information:
    - 'num_blocks': number of text blocks detected
    - 'num_columns': estimated number of columns (1 or 2)
    - 'column_split_x': x-coordinate of column boundary (if 2 columns)
    - 'blocks': list of block bounding boxes
    - 'suggested_psm': recommended Tesseract PSM mode
    - 'page_type': estimated page type
    """
    logger.debug(f"  Analyzing layout for {scan_filename}...")

    # Use Tesseract's layout analysis (block-level detection)
    try:
        data = pytesseract.image_to_data(
            image,
            lang=TESSERACT_LANG,
            config="--psm 1",  # Auto with OSD — best for layout detection
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        logger.warning(f"  Layout analysis failed for {scan_filename}: {e}")
        return {
            "num_blocks": 0,
            "num_columns": 1,
            "column_split_x": None,
            "blocks": [],
            "suggested_psm": 1,
            "page_type": "unknown",
        }

    # Collect unique block bounding boxes
    blocks = []
    seen_blocks = set()
    for i in range(len(data["level"])):
        if data["level"][i] == 2:  # Block level
            block_num = data["block_num"][i]
            if block_num not in seen_blocks:
                seen_blocks.add(block_num)
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                if w > 20 and h > 20:  # Filter noise
                    blocks.append({
                        "block_num": block_num,
                        "bbox": [x, y, x + w, y + h],
                        "width": w,
                        "height": h,
                    })

    # Estimate column count
    img_width = image.width
    mid_x = img_width / 2
    num_columns = 1
    column_split_x = None

    if len(blocks) >= 2:
        # Check if blocks cluster on left and right halves
        left_blocks = [b for b in blocks if b["bbox"][2] < mid_x + 50]
        right_blocks = [b for b in blocks if b["bbox"][0] > mid_x - 50]

        if left_blocks and right_blocks:
            num_columns = 2
            # Estimate split point as the gap between left and right blocks
            left_max_x = max(b["bbox"][2] for b in left_blocks)
            right_min_x = min(b["bbox"][0] for b in right_blocks)
            column_split_x = (left_max_x + right_min_x) // 2

    # Estimate page type
    total_text_area = sum(b["width"] * b["height"] for b in blocks)
    page_area = image.width * image.height
    text_coverage = total_text_area / page_area if page_area > 0 else 0

    if text_coverage > 0.5 and num_columns == 2:
        page_type = "two_column_text"  # Likely name register
    elif text_coverage > 0.5:
        page_type = "single_column_text"  # Institutional, street register
    elif text_coverage > 0.2:
        page_type = "mixed"  # Possibly ads with text
    elif len(blocks) <= 2:
        page_type = "sparse"  # Title page, separator, cover
    else:
        page_type = "other"

    # Suggest PSM mode
    if num_columns == 2:
        suggested_psm = 1  # Auto with OSD handles columns
    elif page_type == "single_column_text":
        suggested_psm = 3  # Fully automatic
    else:
        suggested_psm = 1  # Auto with OSD as safe default

    result = {
        "num_blocks": len(blocks),
        "num_columns": num_columns,
        "column_split_x": column_split_x,
        "blocks": blocks,
        "suggested_psm": suggested_psm,
        "page_type": page_type,
        "text_coverage": round(text_coverage, 3),
    }

    logger.info(
        f"  Layout: {scan_filename} → {page_type}, "
        f"{num_columns} col(s), {len(blocks)} blocks, "
        f"{text_coverage:.0%} coverage"
    )

    return result


def get_scan_page_number(scan_path: Path, scan_index: int) -> int | None:
    """
    Extract the printed page number from a scan file.
    
    Uses the scan_number from the filename (3rd part of archive_record_scan)
    and applies the configured offset to estimate the printed page number.
    If the filename doesn't contain a scan number, falls back to the
    sequential index of the file in the directory listing.
    """
    info = parse_scan_filename(scan_path.name)
    scan_number = info.get("scan_number")

    if scan_number is not None:
        return scan_number_to_page_number(scan_number)
    else:
        # Fallback: use the file's position in the sorted directory listing
        return scan_number_to_page_number(scan_index)


def iterate_scan_pages(
    scans_dir: Path = SCANS_DIR,
) -> Iterator[tuple[Path, Image.Image]]:
    """
    Iterate over all scan pages, yielding (original_path, normalized_image) pairs.
    """
    files = discover_scans(scans_dir)

    for f in files:
        normalized = normalize_image(f)
        yield f, normalized
