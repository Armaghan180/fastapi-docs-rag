# FastAPI Docs RAG

Chat with FastAPI's documentation, with answers grounded in real citations back to the docs.
Includes an eval harness comparing chunking and retrieval strategies.

## Why this project

RAG demos are common; the differentiator here is the eval: an ablation across chunking
strategy (fixed-size vs. heading-based) and retrieval method (dense vs. hybrid dense+BM25),
measured with both retrieval metrics (Recall@k, MRR) and LLM-judged answer quality
(correctness, faithfulness, citation accuracy).

## Architecture

```
Streamlit (chat UI) -> FastAPI backend -> RAG pipeline -> Chroma (dense) + BM25 (sparse)
```

## Project layout

- `src/ingestion/` — pulls FastAPI's docs from GitHub, chunks them (2 strategies), builds indexes
- `src/retrieval/` — dense and hybrid retrievers
- `src/generation/` — answer generation with enforced citation tags
- `src/eval/` — ~20-question eval set with ground-truth sources, metrics, and the eval runner
- `src/api/` — FastAPI backend (`/chat` endpoint)
- `app/` — Streamlit frontend
- `scripts/` — CLI entrypoints for ingestion and evaluation

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then fill in OPENAI_API_KEY
```

## Usage

```bash
python scripts/ingest.py      # fetch + chunk + index the docs (all strategies)
python scripts/evaluate.py    # run the eval matrix, print comparison report
uvicorn src.api.main:app --reload   # start the backend
streamlit run app/streamlit_app.py  # start the UI
```

## Status

Project scaffolded. Ingestion, retrieval, generation, eval, and UI are being built next.
