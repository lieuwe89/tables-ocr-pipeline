# Tables OCR Pipeline

A modular, high-performance OCR and LLM-based extraction pipeline designed for historical Dutch documents, specifically optimized for table-heavy registers (address books, census records, etc.).

## Features

- **Hybrid OCR Engine**: Uses **Surya** for robust layout analysis and printed text recognition, with a pluggable hook for **Loghi HTR** for handwritten content.
- **Hardware Agnostic**: Supports CPU, NVIDIA GPUs (CUDA), Apple Silicon (MPS), and AMD GPUs on Windows (DirectML).
- **Intelligent Classification**: Built-in logic to detect "Print" vs "Handwritten" pages and route them to the appropriate engine.
- **LLM Refinement**: Integrates with Gemini (via OpenRouter or Google AI Studio) for error correction, expansion of abbreviations, and semantic structuring of extracted data.
- **Pipelined Execution**: High-throughput mode where OCR and LLM stages run in parallel.
- **Multi-format Export**: Outputs structured JSON, ALTO XML, and SQLite databases.
- **Resumable**: Checkpoint-based processing with local caching of OCR and LLM results.

## Repository Structure

- `pipeline/`: Core logic for OCR, LLM interaction, alignment, and export.
- `scripts/`: Utilities for geocoding, database building, and data backfilling.
- `viewer/`: A local web-based viewer for QA of OCR and alignment results.
- `viewer.py`: Entry point for the local viewer.
- `requirements.txt`: Python dependencies.

## Setup

1. **Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **OCR Backend**:
   Install `surya-ocr` for the primary backend:
   ```bash
   pip install surya-ocr
   ```

3. **API Keys**:
   Create `pipeline/config_local.py`:
   ```python
   OPENROUTER_API_KEY = "your-key"
   LLM_PROVIDER = "openrouter"
   ```

## Usage

Run the full pipeline:
```bash
python pipeline/run_pipeline.py
```

Arguments:
- `--strategy [auto|surya|loghi]`: Select the OCR engine strategy.
- `--device [cpu|cuda|mps|directml|auto]`: Select hardware acceleration.
- `--pages 1-10`: Process a specific range.
- `--ocr-only`: Skip the LLM stage.

## Specialized Branches

This repository maintains environment-specific branches:
- `pipeline-windows`: Optimized for Windows with AMD GPU support (DirectML).
- `pipeline-dgx-spark`: Optimized for high-performance clusters (NVIDIA/CUDA).

## License

MIT License
