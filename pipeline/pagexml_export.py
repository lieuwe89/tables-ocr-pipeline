"""
PageXML export module.

Generates PageXML 2019-07-15 files from OCR + LLM results.
Standard format for HTR tools like Loghi and Transkribus.
"""

import logging
import time
from pathlib import Path
from lxml import etree

from pipeline.config import PAGEXML_DIR
from pipeline.ocr import OcrPage

logger = logging.getLogger(__name__)

# PageXML Namespace
PAGEXML_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NSMAP = {None: PAGEXML_NS}

def _bbox_to_points(bbox: list[int]) -> str:
    """Convert [x1, y1, x2, y2] to PageXML points 'x1,y1 x2,y1 x2,y2 x1,y2'."""
    x1, y1, x2, y2 = bbox
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"

def create_pagexml_document(
    ocr_page: OcrPage,
    gemini_result: dict | None = None,
) -> etree._Element:
    """
    Create a PageXML 2019-07-15 document for a single page.
    """
    root = etree.Element(f"{{{PAGEXML_NS}}}PcGts", nsmap=NSMAP)
    
    # Metadata
    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Tables OCR Pipeline (Gemini + Surya)"
    created = etree.SubElement(metadata, "Created")
    created.text = time.strftime("%Y-%m-%dT%H:%M:%S")
    last_mod = etree.SubElement(metadata, "LastChange")
    last_mod.text = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Page
    page = etree.SubElement(root, "Page")
    page.set("imageFilename", ocr_page.scan_file)
    page.set("imageWidth", str(ocr_page.width))
    page.set("imageHeight", str(ocr_page.height))
    
    # ReadingOrder (optional but good practice)
    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup")
    ordered_group.set("id", "ro_1")
    ordered_group.set("caption", "Reading order")
    
    for i, block in enumerate(ocr_page.blocks):
        # Reading Order Ref
        region_ref = etree.SubElement(ordered_group, "RegionRefIndexed")
        region_ref.set("index", str(i))
        region_ref.set("regionRef", block.id)
        
        # Text Region
        region = etree.SubElement(page, "TextRegion")
        region.set("id", block.id)
        region.set("type", "paragraph")
        
        coords = etree.SubElement(region, "Coords")
        coords.set("points", _bbox_to_points(block.bbox))
        
        for line in block.lines:
            text_line = etree.SubElement(region, "TextLine")
            text_line.set("id", line.id)
            
            line_coords = etree.SubElement(text_line, "Coords")
            line_coords.set("points", _bbox_to_points(line.bbox))
            
            for word in line.words:
                word_el = etree.SubElement(text_line, "Word")
                word_el.set("id", word.id)
                
                word_coords = etree.SubElement(word_el, "Coords")
                word_coords.set("points", _bbox_to_points(word.bbox))
                
                equiv = etree.SubElement(word_el, "TextEquiv")
                unicode_el = etree.SubElement(equiv, "Unicode")
                unicode_el.text = word.text
            
            # Line-level text fallback
            line_equiv = etree.SubElement(text_line, "TextEquiv")
            line_unicode = etree.SubElement(line_equiv, "Unicode")
            line_unicode.text = " ".join(w.text for w in line.words)
            
    return root

def save_pagexml(
    root: etree._Element,
    scan_filename: str,
    output_dir: Path = PAGEXML_DIR,
) -> Path:
    """Save PageXML document to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(scan_filename).stem
    output_path = output_dir / f"{stem}.xml"

    tree = etree.ElementTree(root)
    tree.write(
        str(output_path),
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
    )

    logger.debug(f"  Saved PageXML: {output_path}")
    return output_path

def export_page_xml(
    ocr_page: OcrPage,
    gemini_result: dict | None = None,
) -> Path:
    """Full PageXML export for a single page."""
    root = create_pagexml_document(ocr_page, gemini_result)
    return save_pagexml(root, ocr_page.scan_file)
