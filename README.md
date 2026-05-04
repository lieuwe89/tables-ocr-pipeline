# Tables OCR Pipeline

A modular, high-performance OCR and LLM-based extraction pipeline designed for historical Dutch documents, specifically optimized for table-heavy registers (address books, census records, civil registers, etc.).

This pipeline is designed to be engine-agnostic and easily adaptable to any archival source requiring structured data extraction from scanned images.

## Features

- **Hybrid OCR Strategy**: Leverage **Surya** for state-of-the-art layout analysis and printed text recognition, with built-in hooks for **Loghi HTR**.
- **Dynamic Section Discovery**: Automatically identifies page types (Name Register, Street Register, Ads) using Vision+LLM to route them to the correct extraction schema without manual mapping.
- **Hardware Agnostic**: Run on everything from developer laptops (CPU/MPS) to high-performance clusters (NVIDIA CUDA) and Windows workstations (AMD DirectML).
- **vLLM (Local LLM) Support**: Support for local high-performance vision models (e.g., Florence-2, Llava) via vLLM on GPU hardware, removing per-token API costs.
- **Multi-Worker Pipelining**: Highly parallelized execution of OCR and LLM stages, optimized for high-memory environments like the DGX Spark.
- **Rich Export Formats**: Generates structured JSON, **ALTO XML (v4.4)**, **PageXML**, and SQLite databases ready for research and publication.

## Repository Structure

- `pipeline/`: Core logic for OCR orchestration, LLM interfacing, and data alignment.
- `scripts/`: Shared utilities for geocoding, database builds, and post-processing.
- `docs/`: In-depth architecture and design documentation.
- `requirements.txt`: Python dependencies.

## Quick Start

1. **Environment Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**:
   Place your scans in `scans/` and create `pipeline/config_local.py`. 
   
   For **Gemini API**:
   ```python
   GOOGLE_AI_API_KEY = "your-key"
   LLM_PROVIDER = "google"
   ```
   
   For **Local vLLM** (DGX/High-end GPU):
   ```python
   LLM_PROVIDER = "vllm"
   VLLM_API_BASE = "http://localhost:8000/v1"
   LLM_WORKERS = 4
   ```

3. **Execution**:
   ```bash
   python run_pipeline.py --strategy auto --device auto
   ```

## Specialized Branches

We maintain environment-specific optimizations:
- `main`: The stable, engine-agnostic core.
- `pipeline-windows`: Tuned for Windows/WSL2 with AMD GPU support (DirectML).
- `pipeline-dgx-spark`: Optimized for high-concurrency cluster environments (NVIDIA/CUDA/vLLM).

## License

MIT License
