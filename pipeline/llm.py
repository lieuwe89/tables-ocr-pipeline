"""
LLM module.

Sends scan images + OCR word lists to a vision LLM (Gemini via OpenRouter
by default, with a Google AI Studio fallback) for:
- OCR error correction
- Abbreviation expansion
- Semantic structuring of entries
- Address extraction
- Cross-reference detection

Includes rate limiting, checkpointing, and retry logic.
"""

import base64
import json
import logging
import time
from pathlib import Path

from pipeline.config import (
    CHECKPOINT_FILE,
    FAILURES_DIR,
    GEMINI_DELAY_SECONDS,
    GEMINI_MODEL,
    GOOGLE_AI_API_KEY,
    LLM_PROVIDER,
    LLM_USAGE_DIR,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    PROMPTS_DIR,
    SCHEMAS_DIR,
    get_section_for_page,
)
from pipeline.ocr import OcrPage

logger = logging.getLogger(__name__)


# ── Prompt loading ────────────────────────────────────────────────────────────

_prompt_cache: dict[str, str] = {}


def load_prompt(prompt_filename: str) -> str:
    """Load and cache a prompt template from the prompts directory."""
    if prompt_filename not in _prompt_cache:
        prompt_path = PROMPTS_DIR / prompt_filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        _prompt_cache[prompt_filename] = prompt_path.read_text(encoding="utf-8")
    return _prompt_cache[prompt_filename]


# ── Schema loading ────────────────────────────────────────────────────────────

_schema_cache: dict[str, dict] = {}


def load_schema(section_name: str) -> dict | None:
    """Load and cache a JSON schema for a section from the schemas directory."""
    if section_name not in _schema_cache:
        schema_path = SCHEMAS_DIR / f"{section_name}.json"
        if not schema_path.exists():
            return None
        try:
            _schema_cache[section_name] = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to load schema for {section_name}: {e}")
            return None
    return _schema_cache[section_name]


# ── Checkpoint management ─────────────────────────────────────────────────────


def load_checkpoint() -> dict:
    """Load the checkpoint file, or create a new one."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def save_checkpoint(checkpoint: dict):
    """Save checkpoint to disk."""
    checkpoint["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def is_page_completed(scan_filename: str, checkpoint: dict) -> bool:
    """Check if a page has already been successfully processed."""
    return scan_filename in checkpoint.get("completed", [])


# ── LLM client ────────────────────────────────────────────────────────────────

_client = None
_client_model: str = ""


def init_gemini(model_override: str | None = None):
    """Initialize the LLM client based on LLM_PROVIDER.

    Pass `model_override` to switch the active model on subsequent calls
    without re-initializing the underlying HTTP client (used by the
    end-of-run fallback retry).
    """
    global _client, _client_model

    if LLM_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Add it to pipeline/config_local.py."
            )
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
            )
        _client_model = model_override or OPENROUTER_MODEL
        logger.info(f"LLM (OpenRouter) using model: {_client_model}")
    elif LLM_PROVIDER == "google":
        if not GOOGLE_AI_API_KEY:
            raise ValueError(
                "GOOGLE_AI_API_KEY not set. Add it to pipeline/config_local.py."
            )
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_AI_API_KEY)
        _client = genai
        _client_model = GEMINI_MODEL
        logger.info(f"LLM (Google AI) initialized with model: {_client_model}")
    elif LLM_PROVIDER == "vllm":
        from openai import OpenAI
        from pipeline.config import VLLM_API_BASE, VLLM_MODEL
        _client = OpenAI(
            base_url=VLLM_API_BASE,
            api_key="vllm-token", # Placeholder
        )
        _client_model = model_override or VLLM_MODEL
        logger.info(f"LLM (vLLM) initialized with model: {_client_model} at {VLLM_API_BASE}")
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def _encode_image_data_url(image_path: Path) -> str:
    """Encode an image file as a base64 data URL for OpenAI-compatible APIs."""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def _call_openrouter(prompt: str, image_paths: Path | list[Path], timeout: int = 120, schema: dict | None = None, model_override: str | None = None) -> tuple[str, dict]:
    """Send prompt + image(s) via OpenRouter."""
    if isinstance(image_paths, Path):
        image_paths = [image_paths]
        
    content = [{"type": "text", "text": prompt}]
    for img in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(img)},
        })

    active_model = model_override or _client_model
    kwargs = {
        "model": active_model,
        "timeout": timeout,
        "temperature": 0.1,
        "max_tokens": 65536,
        "messages": [{"role": "user", "content": content}],
        "extra_headers": {
            "HTTP-Referer": "https://github.com/groningen-adresboek-1926",
            "X-Title": "Groningen Adresboek 1926 Pipeline",
        },
    }
    
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "strict": True,
                "schema": schema
            }
        }

    response = _client.chat.completions.create(**kwargs)
    usage = {}
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "model": active_model,
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    return (response.choices[0].message.content or ""), usage


def _call_vllm(prompt: str, image_paths: Path | list[Path], timeout: int = 120, schema: dict | None = None, model_override: str | None = None) -> tuple[str, dict]:
    """Send prompt + image(s) via local vLLM."""
    if isinstance(image_paths, Path):
        image_paths = [image_paths]
        
    content = [{"type": "text", "text": prompt}]
    for img in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(img)},
        })

    active_model = model_override or _client_model
    kwargs = {
        "model": active_model,
        "timeout": timeout,
        "temperature": 0.1,
        "max_tokens": 32768,
        "messages": [{"role": "user", "content": content}],
    }
    
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction",
                "strict": True,
                "schema": schema
            }
        }

    response = _client.chat.completions.create(**kwargs)
    usage = {}
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "model": active_model,
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
        }
    return (response.choices[0].message.content or ""), usage


def _call_google(prompt: str, image_paths: Path | list[Path], timeout: int = 120, schema: dict | None = None, model_override: str | None = None) -> tuple[str, dict]:
    """Send prompt + image(s) via Google AI Studio."""
    from PIL import Image as PILImage
    if isinstance(image_paths, Path):
        image_paths = [image_paths]
        
    parts = [prompt] + [PILImage.open(img) for img in image_paths]
    
    gen_config = {
        "temperature": 0.1,
        "max_output_tokens": 32768,
    }
    
    if schema:
        gen_config["response_mime_type"] = "application/json"
        gen_config["response_schema"] = schema

    active_model = model_override or _client_model
    model = _client.GenerativeModel(
        active_model,
        generation_config=_client.GenerationConfig(**gen_config),
    )
    response = model.generate_content(
        parts,
        request_options={"timeout": timeout},
    )
    usage = {}
    md = getattr(response, "usage_metadata", None)
    if md:
        usage = {
            "model": active_model,
            "prompt_tokens": getattr(md, "prompt_token_count", None),
            "completion_tokens": getattr(md, "candidates_token_count", None),
            "total_tokens": getattr(md, "total_token_count", None),
        }
    return (response.text or ""), usage


def _call_llm(prompt: str, image_paths: Path | list[Path], timeout: int = 120, schema: dict | None = None, model_override: str | None = None) -> tuple[str, dict]:
    if LLM_PROVIDER == "openrouter":
        return _call_openrouter(prompt, image_paths, timeout=timeout, schema=schema, model_override=model_override)
    elif LLM_PROVIDER == "vllm":
        return _call_vllm(prompt, image_paths, timeout=timeout, schema=schema, model_override=model_override)
    return _call_google(prompt, image_paths, timeout=timeout, schema=schema, model_override=model_override)


def _save_usage_sidecar(image_path: Path, usage: dict, attempt: int, section_type: str) -> None:
    """Persist per-page token usage so cost can be reconciled after the fact."""
    if not usage:
        return
    LLM_USAGE_DIR.mkdir(parents=True, exist_ok=True)
    target = LLM_USAGE_DIR / f"{image_path.stem}.json"
    payload = {
        "scan_file": image_path.name,
        "section_type": section_type,
        "attempt": attempt,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **usage,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _active_model_name() -> str:
    return _client_model or OPENROUTER_MODEL or GEMINI_MODEL or "unknown"


def _failure_path(image_path: Path) -> Path:
    return FAILURES_DIR / f"{image_path.stem}.json"


def _record_failure(
    image_path: Path,
    section_type: str,
    error_class: str,
    error_message: str,
    attempts: int,
    last_response_excerpt: str | None,
    model: str,
) -> None:
    """
    Write a structured failure record for a page that all retries failed on.
    A subsequent successful run for the same page should clear this file via
    `_clear_failure`.
    """
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scan_file": image_path.name,
        "section_type": section_type,
        "error_class": error_class,
        "error_message": error_message,
        "attempts": attempts,
        "model": model,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "partial_response": last_response_excerpt,
    }
    target = _failure_path(image_path)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def _clear_failure(image_path: Path) -> None:
    p = _failure_path(image_path)
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def _extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from a Gemini response, handling several wrapping styles.

    Handles:
    - Raw JSON: {...}
    - Fenced: ```json\n{...}\n``` (with or without language tag, with or without closing fence)
    - Prose-wrapped: "Here is the result:\n{...}"
    """
    text = response_text.strip()

    # Strip a leading fence (with optional language hint), then a trailing fence if present
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    # Fall back to outer-brace extraction if there's still surrounding prose
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


def _crop_header(image_path: Path) -> Path:
    """Crop the top 15% of the image to help the LLM read the header."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    width, height = img.size
    header_height = int(height * 0.15)
    header = img.crop((0, 0, width, header_height))
    
    crop_path = image_path.parent / f"{image_path.stem}_header.tmp.jpg"
    header.save(crop_path, quality=95)
    return crop_path


def identify_page_section(scan_image_path: Path, ocr_page: OcrPage) -> str:
    """
    Use Gemini to identify the section type of a page based on its visual
    layout and a snippet of OCR text.
    
    Uses both the full image and a high-res crop of the header for accuracy.
    """
    prompt_template = load_prompt("classify_section.txt")
    
    # Use first 50 words as a representative sample
    sample_words = ocr_page.all_words[:50]
    sample_text = " ".join(w.text for w in sample_words)
    
    # Pre-process images: full page + header crop
    header_crop_path = None
    try:
        header_crop_path = _crop_header(scan_image_path)
        images = [scan_image_path, header_crop_path]
        
        prompt = prompt_template.format(ocr_sample=sample_text)
        # Append instruction to the prompt about the images
        prompt += "\n\nNote: You have been provided with two images. The first is the full page. The second is a high-resolution crop of the top 15% (the header) to help you read small titles."

        logger.info(f"  Identifying section for {scan_image_path.name} (with header crop)...")
        response_text, _ = _call_llm(prompt, images, timeout=60)
        
        result = _extract_json_from_response(response_text)
        section = result.get("section", "other")
        logger.info(f"  → Identified as '{section}' (reason: {result.get('reasoning', 'none')})")
        return section
    except Exception as e:
        logger.warning(f"  Section identification failed: {e}")
        return "other"
    finally:
        if header_crop_path and header_crop_path.exists():
            header_crop_path.unlink()


def process_page_with_gemini(
    scan_image_path: Path,
    ocr_page: OcrPage,
    page_number: int | None = None,
    section_override: str | None = None,
    max_retries: int = 3,
) -> dict | None:
    """
    Send a scan image and its OCR word list to Gemini for intelligent processing.
    
    Args:
        scan_image_path: Path to the original scan image (color JPEG)
        ocr_page: Parsed Tesseract OCR results with word bounding boxes
        page_number: Printed page number (for section detection). If None,
                     the page is processed with the generic prompt.
        section_override: Force a specific section type instead of auto-detecting
        max_retries: Number of retry attempts on API errors
    
    Returns:
        Parsed JSON dict from Gemini, or None if all retries failed
    """
    # Determine which prompt to use
    if section_override:
        section_type = section_override
    elif page_number is not None:
        section_type, _, _ = get_section_for_page(page_number)
    else:
        section_type = "generic"

    # Auto-identify if we are generic/unknown and not in the very first pages
    if section_type in ("generic", "other", "unknown") and (page_number is None or page_number > 5):
        section_type = identify_page_section(scan_image_path, ocr_page)

    prompt_file = f"{section_type}.txt"
    if not (PROMPTS_DIR / prompt_file).exists():
        logger.warning(f"  Prompt file {prompt_file} missing, falling back to generic.txt")
        prompt_file = "generic.txt"

    logger.info(
        f"Processing {scan_image_path.name} as '{section_type}' "
        f"(page {page_number}, prompt: {prompt_file})"
    )

    # Load and format the prompt
    prompt_template = load_prompt(prompt_file)
    word_list_str = ocr_page.to_numbered_word_list()
    prompt = prompt_template.format(word_list=word_list_str)

    # Load explicit schema if available
    schema = load_schema(section_type)
    if schema:
        logger.info(f"  Using explicit JSON schema for '{section_type}'")

    response_text = ""
    last_error_class = "unknown"
    last_error_message = ""
    for attempt in range(1, max_retries + 1):
        try:
            response_text, usage = _call_llm(prompt, scan_image_path, timeout=120, schema=schema)

            if not response_text:
                last_error_class = "empty_response"
                last_error_message = "LLM returned an empty response"
                logger.warning(
                    f"  Empty response (attempt {attempt}/{max_retries})"
                )
                if attempt < max_retries:
                    time.sleep(GEMINI_DELAY_SECONDS)
                continue

            result = _extract_json_from_response(response_text)
            
            # Ensure the section type is preserved in the result
            if "section" not in result:
                result["section"] = section_type
                
            _save_usage_sidecar(scan_image_path, usage, attempt, section_type)
            _clear_failure(scan_image_path)
            logger.info(
                f"  Parsed LLM response for {scan_image_path.name} "
                f"(in={usage.get('prompt_tokens','?')}, out={usage.get('completion_tokens','?')} tokens)"
            )
            return result

        except json.JSONDecodeError as e:
            last_error_class = "json_parse_error"
            last_error_message = str(e)
            logger.warning(
                f"  JSON parse error (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                logger.debug(f"  Raw response: {response_text[:500]}...")
                time.sleep(GEMINI_DELAY_SECONDS)

        except Exception as e:
            last_error_class = type(e).__name__
            last_error_message = str(e)
            logger.warning(
                f"  LLM API error (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                wait_time = GEMINI_DELAY_SECONDS * (2 ** (attempt - 1))
                logger.info(f"  Waiting {wait_time:.0f}s before retry...")
                time.sleep(wait_time)

    logger.error(f"  All {max_retries} attempts failed for {scan_image_path.name}")
    _record_failure(
        scan_image_path,
        section_type=section_type,
        error_class=last_error_class,
        error_message=last_error_message,
        attempts=max_retries,
        last_response_excerpt=(response_text[:2000] if response_text else None),
        model=_active_model_name(),
    )
    return None


# ── Batch processing ──────────────────────────────────────────────────────────


def process_all_pages(
    pages: list[tuple[Path, OcrPage, int | None]],
    force_reprocess: bool = False,
) -> dict[str, dict]:
    """
    Process all pages through Gemini with rate limiting and checkpointing.
    
    Args:
        pages: List of (scan_path, ocr_page, page_number) tuples
        force_reprocess: If True, ignore checkpoints and reprocess everything
    
    Returns:
        Dict mapping scan filenames to Gemini results
    """
    init_gemini()
    checkpoint = load_checkpoint()
    results = {}

    total = len(pages)
    skipped = 0
    succeeded = 0
    failed = 0

    for i, (scan_path, ocr_page, page_number) in enumerate(pages, start=1):
        filename = scan_path.name

        # Check checkpoint
        if not force_reprocess and is_page_completed(filename, checkpoint):
            skipped += 1
            logger.debug(f"  [{i}/{total}] Skipping {filename} (already completed)")
            continue

        logger.info(f"[{i}/{total}] Processing {filename}...")

        # Call Gemini
        result = process_page_with_gemini(
            scan_image_path=scan_path,
            ocr_page=ocr_page,
            page_number=page_number,
        )

        if result is not None:
            results[filename] = result
            checkpoint["completed"].append(filename)
            succeeded += 1
        else:
            checkpoint["failed"].append(filename)
            failed += 1

        # Save checkpoint after each page
        save_checkpoint(checkpoint)

        # Rate limit delay (skip for last page)
        if i < total:
            time.sleep(GEMINI_DELAY_SECONDS)

    logger.info(
        f"Gemini processing complete: "
        f"{succeeded} succeeded, {failed} failed, {skipped} skipped"
    )

    return results
