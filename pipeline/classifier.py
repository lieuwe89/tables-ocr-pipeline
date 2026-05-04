"""
Classifier module.

Determines if a page scan is primarily "print" or "handwritten" (or a mixture)
to decide which OCR engine to use.
"""

import logging
from enum import Enum
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

class PageType(Enum):
    PRINT = "print"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"

def classify_page(image: Image.Image, image_path: Path | None = None) -> PageType:
    """
    Determine the type of text on a page using vision-based analysis.
    """
    from pipeline.llm import _call_llm, _extract_json_from_response
    
    # If we don't have a path, we have to save a temporary one because _call_llm takes paths
    temp_path = None
    if image_path is None:
        temp_path = Path("/tmp/classification_temp.jpg")
        image.save(temp_path, quality=80)
        image_path = temp_path

    prompt = """
    Analyze this page image and determine if the primary text content is:
    1. "print" (Clean typeset or machine-printed text)
    2. "handwritten" (Cursive, script, or hand-printed text)
    3. "mixed" (A combination, such as a printed form with handwritten entries)
    
    IMPORTANT: If there is ANY significant handwriting (like filled-in table rows), classify it as "mixed" or "handwritten".
    
    Return ONLY valid JSON:
    {"type": "print" | "handwritten" | "mixed", "reasoning": "..."}
    """

    try:
        # Use filename for logging if available
        fname = image_path.name if image_path else "unknown"
        logger.debug(f"Classifying page {fname}...")
        
        response_text, _ = _call_llm(prompt, image_path, timeout=30)
        result = _extract_json_from_response(response_text)
        page_type_str = result.get("type", "print").lower()
        
        if page_type_str == "handwritten":
            return PageType.HANDWRITTEN
        elif page_type_str == "mixed":
            return PageType.MIXED
        return PageType.PRINT
    except Exception as e:
        logger.warning(f"Classification failed, falling back to PRINT: {e}")
        return PageType.PRINT
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
