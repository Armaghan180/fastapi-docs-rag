"""Caches retrievers per (chunking_strategy, retrieval_method) so each one's Chroma
connection and BM25 index (built from a full pass over the strategy's chunks) are built
once and reused across requests, not rebuilt on every /chat call.
"""

from functools import lru_cache

from src.retrieval.base import Retriever, get_retriever


@lru_cache(maxsize=None)
def get_cached_retriever(chunking_strategy: str, retrieval_method: str) -> Retriever:
    return get_retriever(chunking_strategy, retrieval_method)
