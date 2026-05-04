# Tables OCR Pipeline

A modular, high-performance OCR and LLM-based extraction pipeline designed for historical Dutch documents, specifically optimized for table-heavy registers (address books, census records, civil registers, etc.).

This pipeline is designed to be engine-agnostic and easily adaptable to any archival source requiring structured data extraction from scanned images.

## Features

- **Hybrid OCR Strategy**: Leverage **Surya** for state-of-the-art layout analysis and printed text recognition, with built-in hooks for **Loghi HTR** for handwritten content.
- **Hardware Agnostic**: Run on everything from developer laptops (CPU/MPS) to high-performance clusters (NVIDIA CUDA) and Windows workstations (AMD DirectML).
- **Automated Classification**: Logic to detect "Print" vs "Handwritten" pages and route them to the appropriate recognition engine automatically.
- **LLM-Powered Structuring**: Uses Gemini (via OpenRouter or Google AI Studio) to correct OCR errors, expand abbreviations, and transform text into validated JSON schemas.
- **Pipelined Throughput**: Concurrent execution of OCR and LLM stages to maximize hardware utilization.
- **Archival Export**: Generates structured JSON, ALTO XML, and SQLite databases ready for public-facing websites or research analysis.

## Adapting to a New Source

To use this pipeline for a new document or collection:

1.  **Define Sections**: Update `pipeline/config.py` with the page ranges and logical sections of your document.
2.  **Tailor Prompts**: Create or modify the text files in `pipeline/prompts/` to define the JSON schema and extraction rules for your specific data.
3.  **Configure Recognition**: Set your default `OCR_STRATEGY` (e.g., `surya` for clean print, `loghi` for HTR, or `auto` for mixed content).

## Repository Structure

- `pipeline/`: Core logic for OCR orchestration, LLM interfacing, and data alignment.
- `scripts/`: Shared utilities for geocoding, database builds, and post-processing.
- `viewer/`: A local web interface for quality assurance and verification of results.
- `requirements.txt`: Python dependencies.

## Quick Start

1. **Environment Setup**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install surya-ocr
   ```

2. **Configuration**:
   Copy your scans into `scans/` and create `pipeline/config_local.py` with your API keys:
   ```python
   OPENROUTER_API_KEY = "your-key"
   LLM_PROVIDER = "openrouter"
   ```

3. **Execution**:
   ```bash
   python pipeline/run_pipeline.py --strategy auto --device auto
   ```

## Specialized Branches

We maintain environment-specific optimizations:
- `pipeline-windows`: Tuned for Windows/WSL2 with AMD GPU support (DirectML).
- `pipeline-dgx-spark`: Optimized for high-concurrency cluster environments (NVIDIA/CUDA).

## License

MIT License
