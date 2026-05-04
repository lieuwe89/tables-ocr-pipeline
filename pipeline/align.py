"""
Alignment module.

Links Gemini's semantically structured output back to Tesseract's
precise word-level bounding boxes. This is the critical bridge that gives
us both intelligent text understanding AND pixel-accurate highlighting.
"""

import hashlib
import logging
from difflib import SequenceMatcher
from pathlib import Path

from pipeline.ocr import OcrPage, OcrWord

logger = logging.getLogger(__name__)


def merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    """
    Merge multiple bounding boxes into a single enclosing bbox.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
    
    Returns:
        [x1, y1, x2, y2] of the enclosing rectangle
    """
    if not bboxes:
        return [0, 0, 0, 0]
    
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def _coerce_word_ids(raw, _depth: int = 0) -> list[str]:
    """
    Best-effort flatten an LLM-emitted word_ids field into a list of canonical
    `w_NNNN` string IDs. Tolerates:
      - integers like `42` (advertisement.txt prompt sometimes emits these)
      - strings like `42` (no `w_` prefix)
      - nested lists `[[w_001, w_002], [w_003]]`
      - dicts like `{"id": "w_0001", "text": "Berg"}` (institutional prompt
        occasionally returns word objects instead of bare IDs)
      - None / non-iterable junk → returns []
    """
    if raw is None or _depth > 5:
        return []
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            out.extend(_coerce_word_ids(item, _depth + 1))
        return out
    if isinstance(raw, dict):
        # Pull a recognizable id key out; otherwise drop.
        for k in ("id", "word_id", "wid", "ID"):
            if k in raw:
                return _coerce_word_ids(raw[k], _depth + 1)
        return []
    if isinstance(raw, int):
        return [f"w_{raw:04d}"]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("w_"):
            return [s]
        if s.isdigit():
            return [f"w_{int(s):04d}"]
        return [s]
    return []


def validate_word_ids(
    word_ids,
    word_index: dict[str, OcrWord],
    context: str = "",
) -> tuple[list[str], list[str]]:
    """
    Validate that word IDs from the LLM exist in the OCR word index.

    Coerces malformed inputs (integers, nested lists, missing prefix) before
    lookup so a single bad page can't crash the alignment stage.

    Returns:
        (valid_ids, invalid_ids) tuple
    """
    coerced = _coerce_word_ids(word_ids)
    valid: list[str] = []
    invalid: list[str] = []
    for wid in coerced:
        if wid in word_index:
            valid.append(wid)
        else:
            invalid.append(wid)

    if invalid:
        logger.warning(
            f"  {context}: {len(invalid)} invalid word IDs: {invalid[:5]}..."
        )

    return valid, invalid


def resolve_bbox_for_word_ids(
    word_ids: list[str],
    ocr_page: OcrPage,
    context: str = "",
) -> list[int] | None:
    """
    Get the merged bounding box for a list of word IDs.
    
    Vertical bounds (y1, y2) are 'snapped' to the parent line bboxes for 
    stability. Horizontal bounds (x1, x2) use the words themselves for precision.
    
    Returns None if no valid word IDs are found.
    """
    word_index = ocr_page.word_index
    line_index = ocr_page.line_index
    
    valid_ids, _ = validate_word_ids(word_ids, word_index, context)
    if not valid_ids:
        return None
    
    words = [word_index[wid] for wid in valid_ids]
    word_bboxes = [w.bbox for w in words]
    
    # Start with the union of word bboxes
    merged = merge_bboxes(word_bboxes)
    
    # Snap vertical bounds to the union of lines
    line_ids = {w.line_id for w in words if w.line_id in line_index}
    if line_ids:
        line_bboxes = [line_index[lid].bbox for lid in line_ids]
        lines_merged = merge_bboxes(line_bboxes)
        merged[1] = lines_merged[1]  # Snap y1
        merged[3] = lines_merged[3]  # Snap y2
    
    return merged


def fuzzy_find_word(
    target_text: str,
    word_index: dict[str, OcrWord],
    threshold: float = 0.6,
) -> OcrWord | None:
    """
    Find the best-matching word in the index using fuzzy matching.
    Used as a fallback when Gemini references a word ID that doesn't exist.
    """
    best_match = None
    best_score = 0.0
    
    for word in word_index.values():
        score = SequenceMatcher(None, target_text.lower(), word.text.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = word
    
    return best_match


def _normalize_text(text: str | None) -> str:
    """Normalize text for fingerprinting: lowercase, alphanumeric only."""
    if not text:
        return ""
    # Keep alphanumeric characters and lowercase everything
    return "".join(c for c in str(text).lower() if c.isalnum())


def calculate_fingerprint(entry: dict) -> str:
    """
    Calculate a stable fingerprint for an entry based on its content.
    Uses SHA1 of normalized name, address, and occupation.
    """
    # Use expanded fields where possible for better stability
    name = entry.get("name") or entry.get("business_name") or ""
    address = entry.get("address_full") or ""
    occ = entry.get("occupation_expanded") or entry.get("occupation") or ""
    
    # Combine normalized strings
    payload = "|".join([
        _normalize_text(name),
        _normalize_text(address),
        _normalize_text(occ)
    ])
    
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def align_entry(
    entry: dict,
    ocr_page: OcrPage,
) -> dict:
    """
    Align a single Gemini entry to Tesseract bounding boxes.
    
    Takes a Gemini-structured entry and adds bbox information by
    looking up the word_ids in the Tesseract word index.
    
    The entry is modified in-place and also returned.
    """
    word_index = ocr_page.word_index
    entry_id = entry.get("name", entry.get("business_name", "unknown"))

    # Normalize word_ids fields in place so downstream consumers (alto_export
    # in particular) see plain string IDs instead of nested dicts/integers.
    entry["word_ids"] = _coerce_word_ids(entry.get("word_ids", []))
    entry["name_word_ids"] = _coerce_word_ids(entry.get("name_word_ids", []))
    entry["address_word_ids"] = _coerce_word_ids(entry.get("address_word_ids", []))
    word_ids = entry["word_ids"]
    name_word_ids = entry["name_word_ids"]
    address_word_ids = entry["address_word_ids"]

    entry["entry_bbox"] = resolve_bbox_for_word_ids(
        word_ids, ocr_page, context=f"entry '{entry_id}'"
    )
    entry["name_bbox"] = resolve_bbox_for_word_ids(
        name_word_ids, ocr_page, context=f"name of '{entry_id}'"
    )
    entry["address_bbox"] = resolve_bbox_for_word_ids(
        address_word_ids, ocr_page, context=f"address of '{entry_id}'"
    )
    
    # Build the searchable text from all fields
    searchable_parts = []
    for field_name in [
        "name", "initials", "name_prefix", "name_prefix_expanded",
        "occupation", "occupation_expanded",
        "address_street", "address_street_expanded",
        "address_number", "phone",
    ]:
        value = entry.get(field_name)
        if value:
            searchable_parts.append(str(value))
    entry["searchable_text"] = " ".join(searchable_parts)
    
    # Build the full address string. Some street-register pages have the LLM
    # echo the house number into street_name (so street == number); in that
    # case fall back to the number alone instead of emitting "51 51".
    street = entry.get("address_street_expanded") or entry.get("address_street", "") or ""
    number = entry.get("address_number", "") or ""
    street_str = str(street).strip()
    number_str = str(number).strip()
    if street_str and number_str and street_str != number_str:
        # Avoid "Street 29b 29b" if the number is already in the street name
        if street_str.endswith(number_str):
            entry["address_full"] = street_str
        else:
            entry["address_full"] = f"{street_str} {number_str}"
    elif street_str:
        entry["address_full"] = street_str
    elif number_str:
        entry["address_full"] = number_str
    else:
        entry["address_full"] = None
    
    # Calculate fingerprint for stabilization
    entry["fingerprint"] = calculate_fingerprint(entry)
    
    # Validate word IDs
    all_ids = word_ids + name_word_ids + address_word_ids
    valid, invalid = validate_word_ids(all_ids, word_index, context=entry_id)
    entry["_alignment_valid_count"] = len(valid)
    entry["_alignment_invalid_count"] = len(invalid)
    entry["_alignment_confidence"] = (
        len(valid) / len(all_ids) if all_ids else 0.0
    )
    
    return entry


def align_page(
    gemini_result: dict,
    ocr_page: OcrPage,
) -> dict:
    """
    Align all entries on a page from Gemini's output to Tesseract bounding boxes.
    
    Handles different section types (name_register, street_register, etc.)
    by looking for entries in the appropriate nested structure.
    
    Returns the modified gemini_result with bboxes added.
    """
    section = gemini_result.get("section", "generic")
    word_index = ocr_page.word_index
    stem = Path(ocr_page.scan_file).stem
    
    # Add page dimensions
    gemini_result["dimensions"] = {
        "width": ocr_page.width,
        "height": ocr_page.height,
    }
    gemini_result["scan_file"] = ocr_page.scan_file
    
    # Resolve header/footer bboxes (and normalize their word_ids in place).
    for region in ["header", "footer"]:
        region_data = gemini_result.get(region)
        if region_data and "word_ids" in region_data:
            region_data["word_ids"] = _coerce_word_ids(region_data["word_ids"])
            region_data["bbox"] = resolve_bbox_for_word_ids(
                region_data["word_ids"], ocr_page, context=region
            )
    
    # Align entries based on section type
    if section == "name_register":
        entries = gemini_result.get("entries", [])
        for i, entry in enumerate(entries):
            entry["uid"] = f"{stem}:{i:04d}"
            align_entry(entry, ocr_page)
        _report_alignment_stats(entries, ocr_page.scan_file)
    
    elif section == "street_register":
        i = 0
        for street in gemini_result.get("streets", []):
            # Resolve street heading bbox
            heading_ids = street.get("street_heading_word_ids", [])
            street["heading_bbox"] = resolve_bbox_for_word_ids(
                heading_ids, ocr_page, context=f"street '{street.get('street_name')}'"
            )
            for entry in street.get("entries", []):
                # Street register entries inherit the street name
                if not entry.get("address_street"):
                    entry["address_street"] = street.get("street_name")
                    entry["address_street_expanded"] = street.get("street_name_expanded")
                entry["uid"] = f"{stem}:{i:04d}"
                align_entry(entry, ocr_page)
                i += 1
    
    elif section == "occupation_register":
        i = 0
        for occ in gemini_result.get("occupations", []):
            heading_ids = occ.get("heading_word_ids", [])
            occ["heading_bbox"] = resolve_bbox_for_word_ids(
                heading_ids, ocr_page,
                context=f"occupation '{occ.get('occupation_name')}'"
            )
            for entry in occ.get("entries", []):
                entry["uid"] = f"{stem}:{i:04d}"
                align_entry(entry, ocr_page)
                i += 1
    
    elif section == "institutional":
        for i, entity in enumerate(gemini_result.get("entities", [])):
            entity["uid"] = f"{stem}:{i:04d}"
            align_entry(entity, ocr_page)
    
    elif section == "advertisement":
        for i, ad in enumerate(gemini_result.get("advertisements", [])):
            ad["uid"] = f"{stem}:{i:04d}"
            align_entry(ad, ocr_page)
    
    else:
        # Generic: align any addresses found
        for i, addr in enumerate(gemini_result.get("addresses_found", [])):
            addr["uid"] = f"{stem}:{i:04d}"
            addr_ids = addr.get("address_word_ids", [])
            addr["address_bbox"] = resolve_bbox_for_word_ids(
                addr_ids, ocr_page, context="generic address"
            )
    
    return gemini_result


def _report_alignment_stats(entries: list[dict], scan_file: str):
    """Log alignment quality statistics for a set of entries."""
    if not entries:
        return
    
    confidences = [e.get("_alignment_confidence", 0) for e in entries]
    avg_conf = sum(confidences) / len(confidences)
    low_conf = sum(1 for c in confidences if c < 0.8)
    
    logger.info(
        f"  Alignment for {scan_file}: "
        f"{len(entries)} entries, "
        f"avg confidence: {avg_conf:.1%}, "
        f"{low_conf} entries with <80% confidence"
    )
