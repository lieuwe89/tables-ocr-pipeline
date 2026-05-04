import logging
from PIL import Image
from .base import OcrBackend

logger = logging.getLogger(__name__)

_recognition_predictor = None
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

class SuryaBackend(OcrBackend):
    """Surya-based recognition backend (default for print)."""
    
    def recognize(self, image: Image.Image, text_lines: list, device: str = "auto") -> list[str]:
        global _recognition_predictor, _foundation_predictor
        
        if _recognition_predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            
            target_device = _get_device(device)
            logger.info(f"Loading Surya recognition models on {target_device}...")
            
            _foundation_predictor = FoundationPredictor(device=target_device)
            _recognition_predictor = RecognitionPredictor(foundation_predictor=_foundation_predictor)
        
        # Surya recognition usually runs inside the main Surya pipeline,
        # but for this modular architecture, we assume the orchestrator 
        # passes the line bboxes.
        # However, Surya's recognition predictor expects the full image 
        # and line bboxes to perform its own recognition pass.
        
        # Implementation note: In our current run_ocr flow, we call the predictor
        # with det_predictor. To keep it modular, this backend will handle
        # the recognition-only pass if lines are already provided.
        
        from surya.recognition import recognition
        target_device = _get_device(device)
        
        # The actual 'recognition' call needs a list of images and lists of bboxes
        langs = ["nl", "en"] # Default languages
        results = recognition([image], [text_lines], _recognition_predictor, langs=langs)
        
        if not results:
            return ["???" for _ in text_lines]
            
        # Extract text from OcrResult.text_lines
        return [getattr(tl, "text", "") for tl in results[0].text_lines]
