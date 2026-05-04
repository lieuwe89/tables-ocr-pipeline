# Pipeline Backlog & Future Work

This document tracks known issues, technical debt, and the roadmap for the Tables OCR Pipeline.

## 1. Known Data Quality Issues
The following patterns have been identified in register-style extractions and require core pipeline logic refinements:

- **Field Duplication Logic**: In some edge cases, field merging (like address numbers) can result in duplicates if the expected parent entity (e.g., street name) is missing. Improved deduplication is needed in `pipeline/align.py`.
- **Entity Linking Confidence**: Recognition of cross-references and internal links can vary based on section density. Refinement of section-specific prompts is an ongoing task to improve recall.
- **Bounding Box Precision**: To improve alignment on dense pages, the pipeline should move toward using `OcrLine.bbox` as a "sanity ceiling" for entry-level bounding boxes, clipping word-level unions to the physical line bounds.

## 2. Technical Roadmap (v2 Architecture)

The vision for the pipeline is a fully modular, high-performance system for archival collections.

### Core Objectives
- **Stable Entry Fingerprinting**: Implement a content-based fingerprinting system (e.g., SHA1 hashes of normalized fields) to allow pipeline re-runs without losing manual corrections or orphaning database records.
- **PageXML Standardization**: Adopt PageXML as the primary intermediate format on-disk to maximize interoperability with established archival toolchains (Loghi, Transkribus, Escriptorium).
- **Universal Backend Interface**: Refactor the OCR logic into a pluggable `OcrBackend` protocol, enabling per-page or per-region selection of engines (Surya, Loghi, Tesseract) through simple configuration.
- **On-Premise LLM Inference**: Enable local vision-language model support (e.g., Qwen2.5-VL via vLLM) for high-security or high-volume projects where cloud API usage is restricted.

### Automation & Integration
- **Dynamic Section Discovery**: Use a vision-LLM pre-flight check to read page headers and automatically determine the document section, reducing the need for manual page-range mapping.
- **Real-time Indexing**: Rebuild search and cross-reference indexes incrementally during the run to provide immediate feedback on large corpora.

## 3. Maintenance & Lessons Learned
### OCR Cluster Management
A known challenge with some deep-learning OCR backends is "word clustering," where multiple word bboxes collapse into a sub-segment of the line.
- **Strategy**: The pipeline includes a `_repair_word_bboxes` pass that redistributes words proportionally across the correct line span based on character count.
- **Validation**: When adapting to new engines, always verify that the leftmost word of a line aligns with the line-level bounding box.
