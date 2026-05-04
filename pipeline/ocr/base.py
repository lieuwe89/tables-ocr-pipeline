from typing import Protocol, runtime_checkable
from PIL import Image
from .types import OcrPage

@runtime_checkable
class OcrBackend(Protocol):
    """Protocol for OCR recognition backends."""
    
    def recognize(self, image: Image.Image, text_lines: list, device: str = "auto") -> list[str]:
        """
        Recognize text for a set of provided line bounding boxes.
        
        Args:
            image: The full page image.
            text_lines: List of Surya-style line objects (with .bbox).
            device: Hardware device to use.
            
        Returns:
            List of recognized text strings, one per input line.
        """
        ...
