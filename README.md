# FastAPI Docs RAG

Chat with FastAPI's documentation, with answers grounded in real citations back to the docs.
Includes an eval harness comparing chunking and retrieval strategies.

## Why this project

RAG demos are common; the differentiator here is the eval: an ablation across chunking
strategy (fixed-size vs. heading-based) and retrieval method (dense vs. hybrid dense+BM25),
measured with both retrieval metrics (Recall@k, MRR) and LLM-judged answer quality
(correctness, faithfulness, citation accuracy).

## Results

Eval run against 20 ground-truth questions (19 answerable + 1 deliberately out-of-scope,
to test whether the system declines instead of hallucinating), `top_k=5`:

| Config | Recall@k | MRR | Citation Accuracy | Correctness (0-2) | Faithfulness (0-2) |
|---|---|---|---|---|---|
| fixed_size + dense (baseline) | 0.95 | 0.86 | 0.84 | 1.90 | 1.95 |
| heading_based + dense | 1.00 | 0.90 | 0.95 | 1.95 | 2.00 |
| heading_based + hybrid | 0.95 | 0.82 | 0.89 | 2.00 | 2.00 |

Heading-based chunking beats the fixed-size baseline on every metric — the core hypothesis
(coherent section boundaries retrieve better than arbitrary token windows) held up.
Hybrid retrieval didn't clearly beat dense-only on Recall@k/MRR here, plausibly because most
of these questions are conceptual rather than exact-identifier lookups (`BackgroundTasks`,
`status_code=201`) where BM25's keyword matching would matter more.

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
python -m uvicorn src.api.main:app --reload   # start the backend
streamlit run app/streamlit_app.py  # start the UI
```

## Status

Complete: ingestion, retrieval, generation, eval, FastAPI backend, and Streamlit UI are all
built and verified end-to-end.
