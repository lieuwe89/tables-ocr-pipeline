# Pipeline Backlog & Future Work

This document tracks known issues, technical debt, and the roadmap for the Tables OCR Pipeline, consolidated from the project's original development notes.

## 1. Known Data Quality Issues
These issues were identified during the pilot run of the 1926 Groningen Address Book and remain in the backlog for the core pipeline logic:

- **Address Duplication**: `address_full` sometimes duplicates the house number when a street name is missing (e.g., `"29b 29b"`). Needs a fix in `pipeline/align.py`.
- **Cross-Reference Extraction**: Only ~218 cross-references were detected in the pilot run, which is suspiciously low for a 900-page book. This likely requires a refinement of the `name_register.txt` prompt.
- **Entry Bounding Boxes**: The `entry_bbox` calculation sometimes excludes parts of the name. A structural improvement would be to use `OcrLine.bbox` as a "sanity ceiling" for entry bboxes, clipping word-based unions to the line bounds.

## 2. Technical Roadmap (v2 Architecture)

The vision for the next version of the pipeline involves moving toward a more modular, engine-agnostic system.

### Core Objectives
- **Stable Entry IDs**: Implement a fingerprinting system (`sha1(normalized_name + normalized_address + normalized_occupation)`) to allow pipeline re-runs without orphaning manual corrections from the CRM.
- **PageXML Support**: Standardize on PageXML as the intermediate on-disk format for better interoperability with Loghi, Transkribus, and Escriptorium.
- **Backend Abstraction**: Refactor `pipeline/ocr.py` into a pluggable interface (`OcrBackend`) so different engines (Surya, Loghi, Tesseract) can be swapped per-region or per-document.
- **Local Vision Models**: On high-performance hardware (like NVIDIA DGX), replace cloud LLM calls with local vision-language models (e.g., Qwen2.5-VL or InternVL2.5) using vLLM.

### Integration Tasks
- **ALTO Write-back**: Wire manual overrides from the CRM back into `pipeline/alto_export.py` so archival exports reflect human corrections.
- **Auto-Section Detection**: Implement an LLM-based pre-flight check that reads page headers to automatically determine the section type and appropriate prompt, reducing reliance on manual `SECTION_MAP` configuration.
- **Streaming Indexes**: Rebuild combined indexes (search, street, etc.) incrementally during the run rather than waiting for the final stage.

## 3. Maintenance Lessons
### The Surya Bbox Cluster Bug
A critical bug was identified where Surya's word bboxes would "collapse" into a single cluster, missing the leftmost part of a line. 
- **Fix**: The `_repair_word_bboxes` function in `ocr.py` redistributes word bboxes proportionally across the (correct) `OcrLine.bbox` span based on character count.
- **Lesson**: Always verify that the leftmost `OcrWord` on a line starts near the `OcrLine`'s left edge. If there is a significant gap, the detection threshold in `_repair_word_bboxes` may need adjustment.
