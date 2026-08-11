"""Shared retrieval types and the factory that maps a (chunking_strategy, retrieval_method)
pair from config.STRATEGIES to a concrete Retriever.
"""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    id: str
    doc_path: str
    heading_path: list[str]
    anchor: str | None
    text: str
    url: str
    score: float
    rank: int


class Retriever:
    chunking_strategy: str
    retrieval_method: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


def get_retriever(chunking_strategy: str, retrieval_method: str) -> Retriever:
    # Imported here, not at module level, to avoid a circular import: dense.py and hybrid.py
    # both import this module for RetrievedChunk/Retriever.
    from . import dense, hybrid

    if retrieval_method == "dense":
        return dense.DenseRetriever(chunking_strategy)
    if retrieval_method == "hybrid":
        return hybrid.HybridRetriever(chunking_strategy)
    raise ValueError(f"Unknown retrieval method: {retrieval_method!r}")
