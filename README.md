# FACTR — Fact‑Checking & Truth‑Reliability AI Framework

FACTR is an MSc‑level research and prototype system developed as part of the **MSc in Artificial Intelligence (University of Hull)**. The project investigates **automated fact‑checking, evidence retrieval, and verifiability scoring** in structured debates, with a primary experimental focus on **theological discourse** (e.g. Christian–Muslim debates), while remaining extensible to other domains of misinformation.

This repository contains **the complete research artefact**: source code, notebooks, datasets, vector indices, evaluation outputs, and a working Streamlit application used to support the dissertation.

---

## 1. Research Objectives

The FACTR project aims to:

1. Extract **atomic factual claims** from long‑form dialogue or debate transcripts
2. Retrieve **supporting or contradicting evidence** from curated, provenance‑aware knowledge bases
3. Score claims based on **retrievability, semantic alignment, and evidential polarity**
4. Support **human‑in‑the‑loop evaluation** rather than autonomous truth adjudication
5. Provide **quantitative evaluation artefacts** suitable for academic assessment

The system explicitly avoids making ontological or theological truth claims. Its purpose is to **assist structured analysis**, not replace scholars, theologians, or fact‑checkers.

---

## 2. System Architecture (High‑Level)

FACTR follows a modular, pipeline‑oriented architecture:

1. **Ingestion** — transcripts or audio sources
2. **ASR & diarisation** — Whisper‑based speech‑to‑text (optional)
3. **Utterance normalisation** — structured speaker turns
4. **Claim extraction** — identification of discrete factual propositions
5. **Embedding & indexing** — FAISS‑based semantic search
6. **Evidence retrieval** — per‑KB querying
7. **Verification & scoring** — support / contradiction / unverifiable
8. **Human review & export** — Streamlit UI, CSV / JSONL outputs

Each stage persists intermediate artefacts to support reproducibility and analysis.

---

## 3. Repository Structure (Verified)

The structure below reflects the **actual repository contents** as interrogated programmatically.

```text
FACTR/
├── factr_app.py                 # Primary Streamlit research application
├── factrConfig.py               # Global configuration helpers
├── glossary.py                  # Terminology & glossary utilities
├── ui_faq.py                    # UI help / FAQ components
├── interrogate_repo.py          # Repo interrogation & audit tool
│
├── factr/                        # Core FACTR Python package
│   ├── app.py                   # Internal app entry helpers
│   ├── asr.py                   # ASR pipeline (Whisper integration)
│   ├── claims.py                # Claim extraction logic
│   ├── ingest.py                # Data ingestion utilities
│   ├── kb.py                    # Knowledge base & FAISS handling
│   ├── verify.py                # Verification & scoring logic
│   ├── config.py                # Package‑level configuration
│   └── __init__.py
│
├── data/
│   ├── processed/               # Persisted pipeline artefacts
│   │   ├── UTTERANCES.parquet
│   │   ├── CLAIMS.jsonl
│   │   ├── VERIFICATION.jsonl
│   │   ├── KB_*.faiss            # FAISS vector indices (Islamic / Christian / All)
│   │   ├── *.map.jsonl           # Passage ↔ vector mappings
│   │   ├── *_embeddings.npy      # Embedding matrices
│   │   └── LAST_*.json           # Pipeline state snapshots
│   │
│   ├── kb_inspect_exports/       # KB composition summaries (CSV)
│   └── kb_inspect_exports_full/  # Extended KB inspection outputs
│
├── notebooks/
│   ├── FACTR‑KB‑Contruction/     # KB construction & ingestion notebooks
│   ├── FACTR‑PIpeline/           # End‑to‑end pipeline notebooks
│   └── FACTR‑Projet Anaysis/     # Evaluation & dissertation analysis
│
├── reports/
│   └── analysis_outputs/         # Figures & feedback summaries
│
├── models/
│   └── whisper/                  # Local Whisper model cache (large‑v2)
│
├── snapshots/                    # Ingest & verify state snapshots
├── Claim‑Reports/                # Exported verification results
│
├── .streamlit/config.toml        # Streamlit UI configuration
├── Dockerfile                    # Containerised execution
├── requirements.txt              # CPU dependencies
├── requirements_gpu.txt          # GPU‑enabled dependencies
├── build_and_push.ps1            # Build helper (Windows)
└── README.md
```

> **Note:** The repository intentionally includes large artefacts (FAISS indices, Whisper model binaries) to preserve full experimental reproducibility.

---

## 4. Knowledge Bases

FACTR uses **multiple, independently constructed knowledge bases** to preserve provenance and enable comparative analysis:

- Islamic primary sources
- Christian primary sources
- Structured secondary material (clearly labelled)

Each KB is:
- Deterministically chunked
- Embedded using sentence‑level transformers
- Indexed via **FAISS**
- Mapped back to original source passages

KB inspection exports (`data/kb_inspect_exports*`) are used to support **threat‑to‑validity analysis** in the dissertation.

---

## 5. Evaluation & Analysis

Evaluation focuses on **retrieval quality and evidential usefulness**, not correctness of belief.

Artefacts include:

- Precision@K / Hit@K experiments
- Similarity‑threshold sweeps
- Claim‑level verification outputs
- Human feedback summaries
- Error and failure‑mode inspection notebooks

The notebook `FACTR_05_Search+Eval_v2025‑09‑16_v2.0.ipynb` forms the **primary evaluation reference**.

---

## 6. Streamlit Application

The Streamlit interface is a **research UI**, enabling:

- Transcript ingestion
- Claim‑by‑claim evidence inspection
- Verifiability scoring
- Annotated feedback collection
- CSV / JSONL export for offline analysis

The UI is intentionally constrained (single‑user / low‑concurrency) to preserve experimental control.

---

## 7. Running the Project

### 7.1 Local (CPU)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run factr_app.py
```

### 7.2 GPU / Docker

```bash
docker build -t factr .
docker run -p 8501:8501 factr
```

GPU execution is recommended for ASR and embedding stages.

---

## 8. Ethics & Scope

- All data is public, synthetic, or manually curated
- No personal or private datasets are included
- Outputs are **decision‑support only**
- Epistemic limits and bias risks are explicitly acknowledged

---

## 9. Project Status

- **Type:** MSc research prototype
- **Stability:** Experimental
- **Audience:** Examiners, researchers, technical reviewers

This repository should be read alongside the dissertation submission.

---

## 10. Citation

If referencing this work:

> Lukmaan (2025). *FACTR: A Fact‑Checking & Truth‑Reliability Framework for Structured Debate Analysis*. MSc Dissertation, University of Hull.

---

*FACTR explores how AI can assist careful, principled truth‑seeking without replacing human judgement.*

