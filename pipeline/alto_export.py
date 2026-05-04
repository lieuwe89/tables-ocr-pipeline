"""
ALTO XML export module.

Generates ALTO 4.x XML files from the combined OCR + LLM results.
One ALTO file per page, containing:
- Page dimensions
- Text blocks, lines, and words with bounding boxes
- Semantic tags linking words to entries
"""

import json
import logging
from pathlib import Path

from lxml import etree

from pipeline.config import ALTO_DIR
from pipeline.ocr import OcrPage

logger = logging.getLogger(__name__)

# ALTO 4.x namespace
ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
NSMAP = {None: ALTO_NS, "xsi": XSI_NS}


def create_alto_document(
    ocr_page: OcrPage,
    gemini_result: dict | None = None,
) -> etree._Element:
    """
    Create an ALTO 4.x XML document for a single page.
    
    Args:
        ocr_page: Tesseract OCR results with word bounding boxes
        gemini_result: Optional aligned Gemini results (for semantic tags)
    
    Returns:
        lxml Element tree root
    """
    # Root element. lxml handles xmlns:xsi declaration via nsmap; the
    # schemaLocation attribute is namespaced with the xsi prefix.
    alto = etree.Element(f"{{{ALTO_NS}}}alto", nsmap=NSMAP)
    alto.set(
        f"{{{XSI_NS}}}schemaLocation",
        "http://www.loc.gov/standards/alto/ns-v4# "
        "http://www.loc.gov/standards/alto/v4/alto-4-4.xsd",
    )

    # Description
    description = etree.SubElement(alto, "Description")
    source_info = etree.SubElement(description, "sourceImageInformation")
    file_name = etree.SubElement(source_info, "fileName")
    file_name.text = ocr_page.scan_file

    processing = etree.SubElement(description, "Processing")
    processing.set("ID", "proc_1")

    step_ocr = etree.SubElement(processing, "processingStepDescription")
    step_ocr.text = "Surya OCR (Dutch + English)"

    if gemini_result:
        step_llm = etree.SubElement(processing, "processingStepDescription")
        step_llm.text = "Gemini 2.5 Flash correction and semantic structuring"

    # Tags section (for semantic annotations)
    tags = etree.SubElement(alto, "Tags")
    if gemini_result:
        _add_semantic_tags(tags, gemini_result)

    # Layout
    layout = etree.SubElement(alto, "Layout")
    page = etree.SubElement(layout, "Page")
    page.set("ID", f"page_{gemini_result.get('page_number', 0) if gemini_result else 0}")
    page.set("WIDTH", str(ocr_page.width))
    page.set("HEIGHT", str(ocr_page.height))
    page.set("PHYSICAL_IMG_NR", "1")

    # PrintSpace (the entire page)
    print_space = etree.SubElement(page, "PrintSpace")
    print_space.set("HPOS", "0")
    print_space.set("VPOS", "0")
    print_space.set("WIDTH", str(ocr_page.width))
    print_space.set("HEIGHT", str(ocr_page.height))

    # Build word-to-entry mapping from Gemini results
    word_entry_map = {}
    if gemini_result:
        word_entry_map = _build_word_entry_map(gemini_result)

    # Add text blocks, lines, and words
    for block in ocr_page.blocks:
        text_block = etree.SubElement(print_space, "TextBlock")
        text_block.set("ID", block.id)
        text_block.set("HPOS", str(block.bbox[0]))
        text_block.set("VPOS", str(block.bbox[1]))
        text_block.set("WIDTH", str(block.bbox[2] - block.bbox[0]))
        text_block.set("HEIGHT", str(block.bbox[3] - block.bbox[1]))

        for line in block.lines:
            text_line = etree.SubElement(text_block, "TextLine")
            text_line.set("ID", line.id)
            text_line.set("HPOS", str(line.bbox[0]))
            text_line.set("VPOS", str(line.bbox[1]))
            text_line.set("WIDTH", str(line.bbox[2] - line.bbox[0]))
            text_line.set("HEIGHT", str(line.bbox[3] - line.bbox[1]))

            for j, word in enumerate(line.words):
                string_el = etree.SubElement(text_line, "String")
                string_el.set("ID", word.id)
                string_el.set("CONTENT", word.text)
                string_el.set("HPOS", str(word.bbox[0]))
                string_el.set("VPOS", str(word.bbox[1]))
                string_el.set("WIDTH", str(word.bbox[2] - word.bbox[0]))
                string_el.set("HEIGHT", str(word.bbox[3] - word.bbox[1]))
                string_el.set("WC", f"{word.confidence / 100:.2f}")

                # Add semantic tag reference if this word belongs to an entry
                if word.id in word_entry_map:
                    entry_ref = word_entry_map[word.id]
                    string_el.set("TAGREFS", entry_ref)

                # Add space between words (except after last word in line)
                if j < len(line.words) - 1:
                    next_word = line.words[j + 1]
                    sp = etree.SubElement(text_line, "SP")
                    sp.set("HPOS", str(word.bbox[2]))
                    sp.set("VPOS", str(word.bbox[1]))
                    sp.set("WIDTH", str(max(0, next_word.bbox[0] - word.bbox[2])))

    return alto


def _add_semantic_tags(tags: etree._Element, gemini_result: dict):
    """Add semantic tag definitions to the ALTO Tags section."""
    section = gemini_result.get("section", "generic")

    # Collect all entries across section types
    entries = _collect_all_entries(gemini_result)

    for i, entry in enumerate(entries):
        tag = etree.SubElement(tags, "OtherTag")
        tag_id = f"entry_{i + 1}"
        _set_attr(tag, "ID", tag_id)
        _set_attr(tag, "LABEL", entry.get("name") or entry.get("business_name") or f"entry_{i + 1}")
        _set_attr(tag, "TYPE", entry.get("entity_type") or "person")

        if entry.get("address_full"):
            _set_attr(tag, "ADDRESS", entry["address_full"])
        if entry.get("occupation_expanded") or entry.get("occupation"):
            _set_attr(tag, "OCCUPATION", entry.get("occupation_expanded") or entry.get("occupation") or "")

        # Store the tag_id on the entry for word mapping
        entry["_alto_tag_id"] = tag_id


def _set_attr(elem: etree._Element, key: str, value) -> None:
    """Set an XML attribute, coercing None / non-strings safely.

    Defensive shim because LLM output occasionally has nested dicts or None
    where a scalar string is expected, and lxml refuses both."""
    if value is None:
        return
    if isinstance(value, (list, dict)):
        elem.set(key, json.dumps(value, ensure_ascii=False)[:200])
        return
    elem.set(key, str(value))


def _build_word_entry_map(gemini_result: dict) -> dict[str, str]:
    """Build a mapping from word_id → ALTO tag ID."""
    word_map = {}
    entries = _collect_all_entries(gemini_result)

    for entry in entries:
        tag_id = entry.get("_alto_tag_id", "")
        if not tag_id:
            continue
        for wid in entry.get("word_ids", []):
            word_map[wid] = tag_id

    return word_map


def _collect_all_entries(gemini_result: dict) -> list[dict]:
    """Collect all entries from a Gemini result regardless of section type."""
    section = gemini_result.get("section", "generic")
    entries = []

    if section == "name_register":
        entries = gemini_result.get("entries", [])
    elif section == "street_register":
        for street in gemini_result.get("streets", []):
            entries.extend(street.get("entries", []))
    elif section == "occupation_register":
        for occ in gemini_result.get("occupations", []):
            entries.extend(occ.get("entries", []))
    elif section == "institutional":
        entries = gemini_result.get("entities", [])
    elif section == "advertisement":
        entries = gemini_result.get("advertisements", [])

    return entries


# ── File I/O ──────────────────────────────────────────────────────────────────


def save_alto(
    alto_root: etree._Element,
    scan_filename: str,
    output_dir: Path = ALTO_DIR,
) -> Path:
    """
    Save an ALTO XML document to disk.
    
    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(scan_filename).stem
    output_path = output_dir / f"{stem}.xml"

    tree = etree.ElementTree(alto_root)
    tree.write(
        str(output_path),
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )

    logger.debug(f"  Saved ALTO XML: {output_path}")
    return output_path


def export_page_alto(
    ocr_page: OcrPage,
    gemini_result: dict | None = None,
) -> Path:
    """
    Full ALTO export for a single page: create + save.
    """
    alto_root = create_alto_document(ocr_page, gemini_result)
    return save_alto(alto_root, ocr_page.scan_file)
