# Tables OCR Pipeline v2 — Architecture & Vision

Status: **Planning / Core Implemented**. This document outlines the roadmap for transitioning the pipeline from a single-source pilot to a generalized, reusable system for large-scale archival extraction.

## 1. Goal

A modular pipeline that extracts structured, bbox-anchored data from a wide range of archival registers — primarily **table-based or register-style** material — with minimal per-document adaptation.

### Target Sources
- **Printed Registers**: Address books, business directories, and lists.
- **Handwritten Ledgers**: Notarial deeds, civil registers, and historical tax records.
- **Hybrid Documents**: Printed forms with handwritten entries (common in 19th–20th c. archives).

## 2. Multi-Platform Support

The pipeline is designed to be runnable across diverse hardware environments:

| Target | Deployment | Primary Use | OCR Backends |
|---|---|---|---|
| **High-Performance Cluster** | NVIDIA DGX, Ubuntu ARM64 | Production runs for full collections | Surya (GPU) + Loghi HTR |
| **Developer Workstation** | Windows / WSL2 / macOS | Test runs, prompt iteration, and debugging | Surya (CPU/DirectML/MPS) |

### Design Rules
- **Hardware Auto-Detection**: Selects CUDA, DirectML, or MPS when available; falls back to CPU gracefully.
- **Containerization**: Recommended Docker Compose deployment for cluster environments to isolate TensorFlow (Loghi) and PyTorch (Surya) dependencies.
- **Path Portability**: Strict use of `pathlib` to ensure config and code are portable across Windows and POSIX systems.

## 3. Modular Architecture

The v2 architecture decouples the primary extraction concerns so engines can be swapped based on region or document type.

### The Extraction Flow
1.  **Layout / Region Detection**: (Surya / Laypa / Vision-LLM) Produces region polygons and reading order.
2.  **Text Recognition**: (Surya for print / Loghi for HTR) Produces normalized `OcrPage` data (word/line bboxes).
3.  **LLM Structuring**: (Gemini / Local Vision Model) Transforms OCR results into semantic JSON using word-ID anchoring.
4.  **Alignment & Export**: (Engine-agnostic) Final mapping and generation of JSON/ALTO/SQLite outputs.

### Intermediate Formats
- **PageXML**: Transitioning to PageXML as the primary on-disk intermediate format to maximize compatibility with international archival projects (Nationaal Archief, Transkribus).
- **Normalized OcrPage**: An internal data structure that preserves word-level coordinates regardless of which OCR engine produced them.

## 4. Integration Roadmap

### 4.1 Loghi HTR Support
- Subprocess-based integration to keep dependency trees separate.
- Service-based architecture (`docker compose`) to reduce initialization latency during batch runs.

### 4.2 Local LLM Inference
- Goal: Enable local vision-language model support (e.g., Qwen2.5-VL via vLLM) on high-performance hardware.
- Replace cloud API calls with local inference for high-volume or sensitive datasets.

### 4.3 Automated Section Routing
- Move away from manual page-range mapping (`SECTION_MAP`) toward vision-based header detection.
- A pre-flight LLM check identifies the document section and automatically selects the correct prompt and recognition strategy.

## 5. Maintenance Lessons

### The Bbox Cluster Repair
One of the most valuable lessons from the initial development was managing "word clustering" in deep-learning OCR backends. The pipeline includes a specialized redistribution pass that ensures word-level bounding boxes are correctly spread across the physical line, preventing alignment drift on dense pages.

## 6. Reference Documentation

- [README.md](../README.md): Setup and usage instructions.
- [PIPELINE_BACKLOG.md](../PIPELINE_BACKLOG.md): Technical debt and ongoing refinements.
