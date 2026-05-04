"""
Classifier module.

Determines if a page scan is primarily "print" or "handwritten" (or a mixture)
to decide which OCR engine to use.
"""

import logging
from enum import Enum
from PIL import Image

logger = logging.getLogger(__name__)

class PageType(Enum):
    PRINT = "print"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"

def classify_page(image: Image.Image) -> PageType:
    """
    Determine the type of text on a page.
    
    Current implementation is a placeholder. 
    In the future, this could use:
    - A dedicated CNN/Transformer classifier.
    - Heuristics from Surya's layout analysis (e.g. line variance).
    - Metadata from the scan filename or collection.
    """
    # For the 1926 pilot, everything is print.
    logger.debug("Classifying page (placeholder: defaulting to PRINT)")
    return PageType.PRINT
