"""
Configuration for the Groningen Adresboek 1926 extraction pipeline.

Copy this file to config_local.py and fill in your API key.
The pipeline will try config_local.py first, then fall back to this file.
"""

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

# Project root (parent of this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input scans directory
SCANS_DIR = PROJECT_ROOT / "scans"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
HOCR_DIR = OUTPUT_DIR / "hocr"
ALTO_DIR = OUTPUT_DIR / "alto"
JSON_DIR = OUTPUT_DIR / "json"
LLM_RAW_DIR = OUTPUT_DIR / "llm_raw"  # Per-page raw LLM responses (pre-alignment)
LLM_USAGE_DIR = OUTPUT_DIR / "llm_usage"  # Per-page token usage sidecars
FAILURES_DIR = OUTPUT_DIR / "failures"  # Per-page structured failure records
COMBINED_DIR = OUTPUT_DIR / "combined"
PAGEXML_DIR = OUTPUT_DIR / "pagexml"

# Prompts and Schemas
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"

# Checkpoint file for resumable processing
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# ── LLM Configuration ─────────────────────────────────────────────────────────

# Provider selection: "openrouter" (OpenAI-compatible, pay-per-token, Gemini via proxy)
# or "google" (Google AI Studio, free tier with daily caps).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")

# Google AI Studio
GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# OpenRouter — overridden in config_local.py
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
# Fallback model used by the end-of-run failure retry. A page that defeated
# the primary model — typically because its JSON output exceeds the model's
# max_output_tokens — gets one more shot with this. Pick a more capable /
# more verbose-tolerant model. Set to None to disable.
OPENROUTER_FALLBACK_MODEL = "google/gemini-2.5-flash"

# vLLM (Local DGX Spark Backend)
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "microsoft/Florence-2-large")

# Parallelism
LLM_WORKERS = int(os.environ.get("LLM_WORKERS", 1))

# Rate limiting. OpenRouter has no per-minute cap on its end for paid usage,
# but we still pace requests to be polite and stay under upstream provider limits.
GEMINI_REQUESTS_PER_MINUTE = 30  # ~2s spacing
GEMINI_DELAY_SECONDS = 60.0 / GEMINI_REQUESTS_PER_MINUTE

# ── OCR Configuration (Surya) ────────────────────────────────────────────────

# Languages for Surya recognition. Dutch is the source language; including
# English helps with foreign words, names, and addresses occasionally present.
SURYA_LANGS = ["nl", "en"]

# OCR Strategy: "auto" (use classifier per page), "surya" (force Surya), "loghi" (force Loghi)
OCR_STRATEGY = os.environ.get("OCR_STRATEGY", "auto")

# OCR Device: "cpu", "cuda", "mps", "directml", or "auto"
OCR_DEVICE = os.environ.get("OCR_DEVICE", "auto")

# Loghi Model Path (swappable)
LOGHI_MODEL_PATH = os.environ.get("LOGHI_MODEL_PATH", None)

# ── Image Processing ─────────────────────────────────────────────────────────

# Minimum image width (pixels) for OCR. Scans below this are upscaled.
# JPEG DPI metadata is unreliable (typically reports 72 regardless of content),
# so we use a pixel-width heuristic. The Groningen scans are ~1900 px wide,
# which is plenty for Surya — no upscaling needed at this width.
MIN_OCR_WIDTH = 1500

# Supported input formats (JPEG only for now)
# NOTE: Multipage PDF support may be added later if needed.
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ── Scan Number to Page Number ────────────────────────────────────────────────
#
# Scan numbers do NOT match printed page numbers because the cover and
# unnumbered front matter are also scanned.
# Example: scan 138 = printed page 136 → offset = 2
# Adjust this after inspecting the first few scans.

SCAN_TO_PAGE_OFFSET = 2  # printed_page = scan_number - offset


def scan_number_to_page_number(scan_number: int) -> int:
    """Convert a sequential scan number to an estimated printed page number."""
    return max(1, scan_number - SCAN_TO_PAGE_OFFSET)


# ── Section Mapping ───────────────────────────────────────────────────────────
#
# Maps printed page number ranges to section types and prompt templates.
# Adjust these based on the actual table of contents.
# The printed page number is derived from the scan number using the offset above.

SECTION_MAP = [
    # (start_page, end_page, section_type, prompt_file)
    (1, 8, "front_matter", "generic.txt"),
    (9, 13, "institutional", "institutional.txt"),
    (14, 16, "institutional", "institutional.txt"),
    (16, 18, "institutional", "institutional.txt"),  # Begraafplaatsen, Bad- en Zweminrichtingen
    (19, 22, "institutional", "institutional.txt"),  # Verkeerswezen, Postwezen, Telefoonwezen, Telegraafwezen
    (23, 28, "advertisement", "advertisement.txt"),  # Advertentiën Autobusondernemingen etc.
    (29, 62, "institutional", "institutional.txt"),  # Autobusondernemingen, Bodediensten
    (63, 63, "institutional", "institutional.txt"),  # Stoombootdiensten
    (64, 67, "institutional", "institutional.txt"),  # Kiesverenigingen, Eerediensten
    (68, 75, "institutional", "institutional.txt"),  # Weldadigheid, Gasthuizen
    (76, 80, "institutional", "institutional.txt"),  # Ziekenverpleging
    (81, 85, "institutional", "institutional.txt"),  # Nuttige Instellingen
    (86, 95, "institutional", "institutional.txt"),  # Onderwijs
    (96, 100, "institutional", "institutional.txt"),  # Wetenschap en Kunst
    (101, 103, "institutional", "institutional.txt"),  # Handel, Nijverheid
    (104, 107, "institutional", "institutional.txt"),  # Bank- en Verzekeringswezen
    (108, 109, "institutional", "institutional.txt"),  # Belastingen, Registratie
    (110, 111, "institutional", "institutional.txt"),  # Militaire Zaken, Rechts- en Politiewezen
    (112, 115, "institutional", "institutional.txt"),  # Societeits- en Vereenigingsleven
    (116, 117, "institutional", "institutional.txt"),  # Sport
    (118, 118, "institutional", "institutional.txt"),  # Bezienswaardigheden
    (119, 603, "name_register", "name_register.txt"),  # Alphabetisch Naamregister
    (604, 799, "street_register", "street_register.txt"),  # Straten, Pleinen, Grachten
    (800, 999, "occupation_register", "occupation_register.txt"),  # Beroepen en bedrijven
]


def get_section_for_page(page_number: int) -> tuple[str, str, str | None]:
    """
    Return (section_type, prompt_filename, ocr_engine) for a given printed page number.
    Falls back to 'generic' if no section is mapped.
    """
    for entry in SECTION_MAP:
        start, end, section_type, prompt_file = entry[:4]
        ocr_engine = entry[4] if len(entry) > 4 else None
        if start <= page_number <= end:
            return section_type, prompt_file, ocr_engine
    return "generic", "generic.txt", None


# ── Scan Filename Parsing ─────────────────────────────────────────────────────

def parse_scan_filename(filename: str) -> dict:
    """
    Parse scan filename into archive number, record id, year, and scan number.

    Expected format: archiveNumber_recordNumber-year_scanNumber.jpg
    Example: 1769_19525-1926_0001.jpg
    - archive_number (1769): archive collection identifier
    - record_number (19525): record/volume id within the collection
    - year (1926): year of the source document
    - scan_number (0001): sequential scan number within the record

    The scan_number does NOT match the printed page number (covers and
    unnumbered front matter are also scanned). Use scan_number_to_page_number().

    Returns dict with 'archive_number', 'record_number', 'year', 'scan_number', 'stem'.
    Any field that cannot be parsed is set to None.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    result = {
        "archive_number": None,
        "record_number": None,
        "year": None,
        "scan_number": None,
        "stem": stem,
    }

    def _try_int(s: str):
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    if len(parts) >= 1:
        result["archive_number"] = _try_int(parts[0])
    if len(parts) >= 2:
        record_year = parts[1].split("-", 1)
        result["record_number"] = _try_int(record_year[0])
        if len(record_year) == 2:
            result["year"] = _try_int(record_year[1])
    if len(parts) >= 3:
        result["scan_number"] = _try_int(parts[-1])
    return result


# ── Local overrides ───────────────────────────────────────────────────────────
# Import config_local.py if it exists, to override any settings above.

try:
    from pipeline.config_local import *  # noqa: F401, F403
except ImportError:
    pass
