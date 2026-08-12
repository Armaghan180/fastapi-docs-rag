"""FastAPI backend: a thin HTTP layer over the RAG pipeline. Streamlit is a pure client of
this API rather than importing the RAG modules directly, which keeps the pipeline reusable
by any other client and gives the project a typed request/response contract.
"""

from fastapi import FastAPI, HTTPException

import config
from src.generation.answer import generate_answer

from . import dependencies, schemas

app = FastAPI(
    title="FastAPI Docs RAG",
    description="Chat with FastAPI's documentation, with citations back to the exact section.",
)

# heading_based+dense had the best Recall@k/MRR in the eval (see README) -- the default a
# caller gets if it doesn't ask for a specific strategy.
DEFAULT_STRATEGY = ("heading_based", "dense")
VALID_STRATEGIES = set(config.STRATEGIES)


def _strategy_option(chunking_strategy: str, retrieval_method: str) -> schemas.StrategyOption:
    return schemas.StrategyOption(
        chunking_strategy=chunking_strategy,
        retrieval_method=retrieval_method,
        label=f"{chunking_strategy} + {retrieval_method}",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/strategies", response_model=schemas.StrategiesResponse)
def list_strategies() -> schemas.StrategiesResponse:
    return schemas.StrategiesResponse(
        strategies=[_strategy_option(cs, rm) for cs, rm in config.STRATEGIES],
        default=_strategy_option(*DEFAULT_STRATEGY),
    )


@app.post("/chat", response_model=schemas.ChatResponse)
def chat(request: schemas.ChatRequest) -> schemas.ChatResponse:
    key = (request.chunking_strategy, request.retrieval_method)
    if key not in VALID_STRATEGIES:
        valid = ", ".join(f"{cs}+{rm}" for cs, rm in config.STRATEGIES)
        raise HTTPException(status_code=400, detail=f"Unknown strategy {key}. Valid options: {valid}")

    retriever = dependencies.get_cached_retriever(*key)
    retrieved = retriever.retrieve(request.question, top_k=config.TOP_K)
    rag_answer = generate_answer(request.question, retrieved)

    return schemas.ChatResponse(
        answer=rag_answer.answer,
        citations=[
            schemas.CitationOut(index=c.index, url=c.url, doc_path=c.doc_path, heading_path=c.heading_path)
            for c in rag_answer.citations
        ],
        retrieved_chunks=[
            schemas.RetrievedChunkOut(doc_path=c.doc_path, heading_path=c.heading_path, url=c.url, score=c.score)
            for c in retrieved
        ],
        chunking_strategy=request.chunking_strategy,
        retrieval_method=request.retrieval_method,
    )
