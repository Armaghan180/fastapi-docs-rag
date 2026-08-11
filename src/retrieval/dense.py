"""Dense retrieval: embed the query, search the strategy's Chroma collection by cosine
similarity.
"""

import chromadb
from openai import OpenAI

import config

from . import base


class DenseRetriever(base.Retriever):
    retrieval_method = "dense"

    def __init__(self, chunking_strategy: str):
        self.chunking_strategy = chunking_strategy
        self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._collection = chroma_client.get_collection(chunking_strategy)

    def embed_query(self, query: str) -> list[float]:
        resp = self._openai.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[base.RetrievedChunk]:
        embedding = self.embed_query(query)
        results = self._collection.query(query_embeddings=[embedding], n_results=top_k)

        chunks = []
        rows = zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
        for rank, (chunk_id, text, meta, distance) in enumerate(rows, start=1):
            chunks.append(
                base.RetrievedChunk(
                    id=chunk_id,
                    doc_path=meta["doc_path"],
                    heading_path=meta["heading_path"].split(" > ") if meta["heading_path"] else [],
                    anchor=meta["anchor"] or None,
                    text=text,
                    url=meta["url"],
                    score=1 - distance,  # cosine distance -> cosine similarity
                    rank=rank,
                )
            )
        return chunks
