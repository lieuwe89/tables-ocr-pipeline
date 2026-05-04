# Pipeline v2 — Reusable Extraction for Groninger Archieven Collections

Status: **planning**. Supersedes the pipeline portions of `NEXT_STEPS.md` §7 and `docs/archive/handoff.md`. The current pilot pipeline (1926 Groningen address book) keeps running as-is for the website's data layer; v2 is the next-book architecture.

Created 2026-05-03.

---

## 1. Goal

A single pipeline that extracts structured, bbox-anchored data from a wide range of Groninger Archieven documents — primarily **table-based / register-style** material — with as few per-document adaptations as possible.

Must handle:
- Printed registers and address books (the pilot case)
- **Handwritten ledgers, notarial deeds, civil registers** (where Loghi earns its keep)
- **Hybrid pages** (printed forms with handwritten entries — common in 19th–20th c. archival material)

Not in scope: continuous prose (literature, narrative reports). Different problem, different tooling.

---

## 2. Target environment

**Primary deployment: NVIDIA DGX Spark, Ubuntu, ARM64 (Grace+Blackwell, sm_121), 128 GB unified memory.** The DGX Spark is on-site at work — access is not always available, so the pipeline must remain runnable in a degraded-but-functional mode on a developer laptop.

**Two maintained forks:**

| Fork | Where | Purpose | OCR backends | LLM |
|---|---|---|---|---|
| `linux-dgx` (canonical) | DGX Spark, Ubuntu ARM64 | Production runs, full collection | Surya (GPU) + Loghi (Laypa + HTR) | Local vision model (Qwen2.5-VL or similar) |
| `windows-pilot` | Developer Windows box | Small test runs, prompt iteration, debug | Surya (CPU) only | OpenRouter / Gemini |

macOS is supported as a developer convenience for code editing and unit tests, **not** for OCR runs (Loghi explicitly does not support Apple Silicon; Surya works but slowly).

**Design rules:**
- POSIX paths via `pathlib` everywhere; no backslashes in defaults.
- Process management examples in docs use Linux primitives (`systemd-run`, `tmux`, `nohup`, `systemd-inhibit`); Windows recipes (`Start-Process -WindowStyle Hidden`) move to a "Windows fork" appendix, not the main runbook.
- GPU detection auto-selects CUDA when present; CPU fallback for laptop runs.
- Containerization: Docker Compose on Linux as the canonical deployment shape; Windows can run the Python pipeline in a venv.

---

## 3. Architecture changes from pilot

The pilot fused three concerns: layout + OCR + structuring. v2 splits them behind interfaces so engines can swap per-document or per-region.

```
JPEG/TIFF scans
   │
   ▼
[1] Layout / region detection                       backends: Surya · Laypa · vision-LLM
       ↓ produces region polygons + reading order
       ↓
[2] Text recognition per region                     backends: Surya (print) · Loghi-HTR (handwritten)
       ↓ produces normalized OcrPage (word/line bboxes + IDs)
       ↓
[3] LLM structuring (vision + word-ID list)         backends: OpenRouter (Gemini/Claude) · local (Qwen2.5-VL)
       ↓ produces section-typed entries referencing word IDs
       ↓
[4] Alignment                                       engine-agnostic (current code, lightly generalized)
       ↓
[5] Export — ALTO XML + per-page JSON + indexes     engine-agnostic
```

Key principles:
- Stages 3, 4, 5 must not know which OCR engine produced the bboxes.
- Stage 2 may run different recognizers on different regions of the same page (printed header by Surya, handwritten body by Loghi-HTR).
- The LLM-with-word-IDs trick — the most valuable invention of the pilot — is preserved verbatim. Only the source of the word IDs changes.

### Intermediate format

Standardize on a normalized in-memory `OcrPage` (already exists in `pipeline/ocr.py`) plus **PageXML** as the on-disk lingua franca with archival NL projects. ALTO conversion happens at export. Rationale:
- Loghi reads/writes PageXML natively.
- Transkribus, Escriptorium, and Nationaal Archief tooling all speak PageXML.
- Existing ALTO export becomes a converter step from PageXML → ALTO instead of a special path.

---

## 4. OCR backend integration plan

### 4.1 Backend interface

Define `pipeline/ocr/base.py`:

```python
class OcrBackend(Protocol):
    def supports(self, region: Region) -> bool: ...
    def recognize(self, image: Image, region: Region) -> RegionResult: ...
    # RegionResult contains lines, words, bboxes, confidence, source-engine tag
```

Existing `pipeline/ocr.py` becomes `pipeline/ocr/surya_backend.py`. Add `pipeline/ocr/loghi_backend.py` invoking Loghi via subprocess (or HTTP if running as a service). The pipeline orchestrator picks backend per region based on `region.type` (printed / handwritten / mixed) and a config-driven priority list.

### 4.2 Layout

Two layout backends behind a common interface:
- **Surya layout** for clean printed multi-column work (current pilot default). Fast, no extra deps.
- **Laypa** (Loghi's layout model) for messy material — registers with marginalia, damaged pages, handwritten ledgers. Outputs PageXML region polygons. Models swappable by century/genre.

A simple per-document config picks one. Auto-routing (LLM looks at first 3 pages, picks layout backend) is a v2.5 nice-to-have.

### 4.3 Loghi integration

**Deployment on DGX Spark:**
- Loghi-HTR is **TensorFlow** (Python 3.9–3.11). Laypa is **PyTorch + detectron2**. Both must be containerized on Blackwell — TF/PyTorch wheels for sm_121 don't exist upstream yet, so we build images on top of NVIDIA NGC base images (`nvcr.io/nvidia/tensorflow:*` and `nvcr.io/nvidia/pytorch:*`), which are multi-arch and tuned for Grace.
- Official Loghi Docker images are amd64-only as of the last check. We rebuild from source for arm64 in our own registry. **Risk to verify before committing time:** detectron2 builds on Blackwell ARM64. If detectron2 is broken on sm_121, Laypa is blocked and we fall back to Surya layout for the v2 release.
- Run as a long-lived service (`docker compose up loghi`) with a thin HTTP wrapper, so the Python pipeline doesn't pay container-start latency per page.

**Deployment on Windows pilot fork:**
- Loghi is not supported. Windows fork uses Surya only. Documents requiring HTR can be processed on the DGX or skipped on Windows.

**Model selection:**
- Loghi ships several HTR checkpoints (Dutch 17th/18th/19th c., Latin, etc.). Per-document config picks the right one. Default for Groninger Archieven 19th–20th c. material: latest `dutch-19c-print-and-htr` (or current equivalent — verify against the release page when starting integration).

### 4.4 Pilot OCR keeps working

The Surya backend stays the default for any reproduction of the 1926 Groningen run. Existing per-page caches (`output/hocr/<stem>.ocr.json`) keep their format — the new backend interface just wraps the existing code path.

---

## 5. LLM structuring — local model on DGX

Goal: replace OpenRouter/Gemini calls on the DGX fork with a local vision-language model. Windows fork keeps OpenRouter.

### 5.1 Model candidates (verify when implementing)

- **Qwen2.5-VL 32B / 72B** — strong at structured output, good Dutch handling, runs on 128 GB unified memory comfortably.
- **InternVL2.5** — competitive, larger ecosystem of fine-tunes.
- **Llama 3.2 Vision** — weaker at structured-output adherence in our domain; backup only.

Inference engine: **vLLM** (NVIDIA confirms Blackwell support, NGC builds available) running as a service, exposing an OpenAI-compatible API. The pipeline's LLM client gets a `local-vllm` provider alongside `openrouter` and `google-direct`.

### 5.2 Architecture fork implications

| Concern | OpenRouter (Windows) | Local vLLM (DGX) |
|---|---|---|
| Rate limiting | Per-minute caps, real | Concurrency cap, no per-minute |
| Cost telemetry | Per-call USD via `output/llm_usage/` | N/A — replace with tokens-per-second + GPU-hour approximation |
| Output token budget | 65k cap matters | 128k+, less worry about truncation on dense pages |
| Determinism | Provider-side, not under our control | Pinnable seed |
| Failure modes | Network, 429, upstream rate limit | OOM, model crash |

The provider abstraction in `pipeline/llm.py` already supports multiple backends; this is a new provider, not a fork of the LLM stage. Cost-telemetry code becomes `cost-or-perf-telemetry` and emits provider-appropriate metrics.

### 5.3 Hybrid mode

Worth keeping the option of "OCR locally on DGX, structure remotely on a frontier model" for the highest-quality runs. The provider abstraction makes this trivial — just pick `openrouter` for stage 3 even on the DGX fork.

---

## 6. Cross-platform strategy details

### 6.1 Path / process / runtime

| Concern | Linux (DGX) | Windows pilot |
|---|---|---|
| Long-running detached process | `systemd-run --user --scope` or `tmux new -d` | `Start-Process -WindowStyle Hidden` |
| Prevent sleep during run | `systemd-inhibit --what=sleep` | Power settings (manual) |
| GPU access | Native CUDA via NGC containers | None (Surya CPU) |
| Loghi | Docker container (rebuilt for arm64) | Not available |
| Local LLM | vLLM container | Not available |
| Python | 3.11 in the project venv; 3.9–3.11 in Loghi container | 3.11 in the project venv |

### 6.2 What stays unified

- Source code (one branch per fork — e.g. `main` for Linux, `windows-pilot` branch maintained alongside, or feature flags / config). Decide once we hit the first divergence.
- Config-as-code: backend selection is config, not a code path. A Linux-default config and a Windows-default config; user can override.
- Test data: same fixtures for unit tests on both platforms.

### 6.3 Branch / fork mechanics — open

Two reasonable shapes:

1. **Single branch, runtime config.** All backends importable; engines unavailable on a platform raise on construction. Pro: one codebase. Con: Loghi/vLLM Python wrappers may pull TF/Torch deps that break Windows install.
2. **Long-lived `windows-pilot` branch.** Linux fork is `main`; Windows fork is rebased periodically. Pro: clean platform isolation. Con: cherry-picking discipline.

Decide when first divergence forces it. Default to (1) until pip install pain proves otherwise.

---

## 7. Pipeline backlog (moved from `NEXT_STEPS.md` §7)

Don't preempt. Fix when a real document or website use proves it matters.

- `address_full` duplicates the number — 5-line fix in `pipeline/align.py` (still applies)
- `entry_bbox` excludes the name — see original §7 (still applies)
- 218 cross-references is suspiciously low — diagnostic pass, likely prompt fix (still applies)
- Pipeline OCR → LLM is sequential staged; should be `max(OCR, LLM)` — **partially fixed in pilot** via `stage_ocr_llm_pipelined`. v2 generalizes this to a multi-worker pool with shared rate limiter (or shared concurrency cap on local vLLM).
- Streaming combined indexes during run, not only at end (still applies)
- Pre-OCR section detection (LLM reads page header, no manual `SECTION_MAP`) — **becomes important for v2** since each new book otherwise needs hand-tuned config.
- Real OCR worker pool, not the reverse-worker hack — **resolved by GPU on DGX**; one process saturates the device.
- Structured `output/failures.json` aggregate — **already fixed in pilot.**

Also-deferred from `NEXT_STEPS.md` §9 that belongs here:
- Cross-platform paths in runbooks — addressed by §6 above.

---

## 8. Risks and open questions

| Risk | Severity | Mitigation |
|---|---|---|
| **TF on Blackwell ARM64** lacks upstream wheels | High | Use NGC TF base images. Verify `tf.config.list_physical_devices('GPU')` works in NGC container before committing to Loghi-HTR. |
| **Detectron2 on Blackwell ARM64** may need patches | High | If broken, fall back to Surya layout for v2. Re-evaluate Laypa for v2.5. |
| **No official Loghi ARM64 images** | Medium | Build multi-arch from source. Plan a 1-day spike on the DGX before scheduling integration work. |
| **vLLM Blackwell stability** | Medium | NVIDIA-supported, but new arch. Pin a known-good NGC vLLM image; treat upgrades as scheduled work. |
| **DGX access is intermittent** | Medium | Keep Windows pilot fork functional for prompt iteration on small samples. Critical path development happens on whichever environment is available. |
| **Two CUDA stacks (TF for HTR, Torch for vLLM)** in the same machine | Low | Separate containers; no shared venv. |
| **Per-book config drift** | Medium | Push toward auto-detection (LLM reads header, picks section type and layout backend) before book #3. |

---

## 9. Phasing

Rough order; revise once first DGX spike confirms or denies the TF/detectron2 risks.

**Phase 0 — DGX feasibility spike** (~2 days on the DGX)
- Bring up NGC TF and PyTorch containers, confirm GPU access.
- Try to build Laypa (detectron2) in the PyTorch container. Go/no-go on Laypa for v2.
- Try a Loghi-HTR run on a sample handwritten page in the TF container. Confirm Dutch model works.
- Bring up vLLM with Qwen2.5-VL, confirm a structured-output prompt round-trips.

**Phase 1 — Backend abstraction** (~1 week, off-DGX OK)
- Refactor `pipeline/ocr.py` into `pipeline/ocr/surya_backend.py` behind a `OcrBackend` protocol.
- Refactor `pipeline/llm.py` provider selection (already partially done) so a `local-vllm` provider plugs in trivially.
- Adopt PageXML as the on-disk intermediate; ALTO becomes a converter.

**Phase 2 — Loghi integration** (~1 week, requires DGX)
- Containerize Loghi-HTR + Laypa using NGC bases.
- Wire `LoghiBackend` to the OCR interface via HTTP.
- Run a pilot handwritten document end-to-end.

**Phase 3 — Local LLM integration** (~3 days, requires DGX)
- Containerize vLLM + chosen vision model.
- Implement `local-vllm` provider in `pipeline/llm.py`.
- Re-run pilot Groningen 1926 with local LLM, compare entries vs Gemini baseline.

**Phase 4 — Auto-routing & polish** (~1 week)
- Header-based section detection (per §7 backlog item).
- Per-page backend selection (printed → Surya, handwritten → Loghi).
- Stream combined indexes during run.
- Update runbooks for Linux + Windows forks.

---

## 10. Reference

- Pilot architecture & gotchas: `CLAUDE.md`
- Original pilot runbook (PowerShell-centric, supersedes for Linux): `docs/archive/handoff.md`
- Web app plan (independent of v2 pipeline): `NEXT_STEPS.md`
- Loghi: <https://github.com/knaw-huc/loghi>
- Loghi-HTR: <https://github.com/knaw-huc/loghi-htr>
- DGX Spark Docker / NGC docs: <https://docs.nvidia.com/dgx/dgx-spark/>
