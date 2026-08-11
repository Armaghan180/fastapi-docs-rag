"""Hybrid retrieval: fuse dense (embedding) and BM25 (keyword) rankings with Reciprocal
Rank Fusion. Dense embeddings catch semantic similarity but can miss exact-term lookups
that matter a lot in API docs (`Depends`, `BackgroundTasks`, `status_code`); BM25 catches
those. RRF combines the two ranked lists without needing to calibrate/normalize scores
from two different retrieval methods onto the same scale.
"""

import dataclasses
import json
import re

from rank_bm25 import BM25Okapi

import config

from . import base, dense

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25Index:
    def __init__(self, chunking_strategy: str):
        path = config.DATA_PROCESSED_DIR / f"{chunking_strategy}.jsonl"
        with path.open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]
        self._bm25 = BM25Okapi([_tokenize(r["text"]) for r in self.records])

    def retrieve(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.records[i], scores[i]) for i in ranked]


class HybridRetriever(base.Retriever):
    retrieval_method = "hybrid"

    def __init__(self, chunking_strategy: str):
        self.chunking_strategy = chunking_strategy
        self._dense = dense.DenseRetriever(chunking_strategy)
        self._bm25 = _BM25Index(chunking_strategy)

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[base.RetrievedChunk]:
        pool = config.CANDIDATE_POOL_SIZE
        dense_results = self._dense.retrieve(query, top_k=pool)
        bm25_results = self._bm25.retrieve(query, top_k=pool)

        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, base.RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + 1 / (config.RRF_K + rank)
            chunk_lookup[chunk.id] = chunk

        for rank, (record, _score) in enumerate(bm25_results, start=1):
            chunk_id = record["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (config.RRF_K + rank)
            if chunk_id not in chunk_lookup:
                chunk_lookup[chunk_id] = base.RetrievedChunk(
                    id=chunk_id,
                    doc_path=record["doc_path"],
                    heading_path=record["heading_path"],
                    anchor=record["anchor"],
                    text=record["text"],
                    url=record["url"],
                    score=0.0,
                    rank=0,
                )

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [
            dataclasses.replace(chunk_lookup[chunk_id], score=rrf_scores[chunk_id], rank=rank)
            for rank, chunk_id in enumerate(ranked_ids, start=1)
        ]
