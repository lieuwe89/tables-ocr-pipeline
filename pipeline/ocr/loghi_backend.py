import logging
from PIL import Image
from .base import OcrBackend
from pipeline.config import LOGHI_MODEL_PATH

logger = logging.getLogger(__name__)

class LoghiBackend(OcrBackend):
    """Loghi HTR-based recognition backend (specialized for handwriting)."""
    
    def recognize(self, image: Image.Image, text_lines: list, device: str = "auto") -> list[str]:
        if not LOGHI_MODEL_PATH:
            logger.warning("  Loghi selected but LOGHI_MODEL_PATH not set. Falling back to '???'")
            return ["???" for _ in text_lines]
        
        logger.info(f"  Calling Loghi HTR (model: {LOGHI_MODEL_PATH}) on {len(text_lines)} lines...")
        
        # In a real implementation, this would:
        # 1. Crop the image according to text_lines bboxes.
        # 2. Save them to a temp directory.
        # 3. Call Loghi CLI/Docker via subprocess.
        # 4. Read the results from PageXML/text files.
        
        # Placeholder for now:
        return ["HTR_RESULT" for _ in text_lines]
