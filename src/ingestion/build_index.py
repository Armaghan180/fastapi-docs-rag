"""Chunk every fetched doc under each strategy, save the chunks as JSONL (for eval/debugging
without needing Chroma), and embed + index them into one Chroma collection per strategy.
"""

import json

import chromadb
from openai import OpenAI
from tqdm import tqdm

import config

from .chunkers import base, fixed_size, heading_based

CHUNKERS: dict[str, base.Chunker] = {
    "fixed_size": fixed_size.FixedSizeChunker(),
    "heading_based": heading_based.HeadingBasedChunker(),
}


def load_raw_docs() -> list[tuple[str, str]]:
    """Returns (doc_path, markdown) pairs, doc_path relative to DATA_RAW_DIR."""
    docs = []
    for path in sorted(config.DATA_RAW_DIR.rglob("*.md")):
        rel = path.relative_to(config.DATA_RAW_DIR).as_posix()
        docs.append((rel, path.read_text(encoding="utf-8")))
    return docs


def run_chunker(strategy: str, docs: list[tuple[str, str]]) -> list[base.Chunk]:
    chunker = CHUNKERS[strategy]
    chunks = []
    for doc_path, markdown in docs:
        chunks.extend(chunker.chunk(doc_path, markdown))
    return chunks


def save_chunks(strategy: str, chunks: list[base.Chunk]) -> None:
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED_DIR / f"{strategy}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "id": c.id,
                "doc_path": c.doc_path,
                "heading_path": c.heading_path,
                "anchor": c.anchor,
                "text": c.text,
                "strategy": c.strategy,
                "index": c.index,
                "token_count": c.token_count,
                "url": c.url,
            }
            f.write(json.dumps(record) + "\n")


def embed_texts(client: OpenAI, texts: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def build_chroma_collection(strategy: str, chunks: list[base.Chunk]) -> None:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = embed_texts(client, [c.text for c in chunks])

    chroma_client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    try:
        chroma_client.delete_collection(strategy)
    except Exception:
        pass
    # OpenAI embeddings are meant to be compared by cosine similarity; Chroma defaults to
    # raw L2 distance unless told otherwise.
    collection = chroma_client.create_collection(strategy, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_path": c.doc_path,
                "heading_path": " > ".join(c.heading_path),
                "anchor": c.anchor or "",
                "url": c.url,
                "index": c.index,
                "token_count": c.token_count,
            }
            for c in chunks
        ],
    )


def main() -> None:
    docs = load_raw_docs()
    if not docs:
        raise SystemExit(f"No docs found in {config.DATA_RAW_DIR} -- run fetch_docs.py first.")
    print(f"Loaded {len(docs)} raw doc pages.")

    for strategy in CHUNKERS:
        print(f"[{strategy}] chunking...")
        chunks = run_chunker(strategy, docs)
        save_chunks(strategy, chunks)
        print(f"[{strategy}] {len(chunks)} chunks -> {config.DATA_PROCESSED_DIR / (strategy + '.jsonl')}")

        print(f"[{strategy}] embedding + indexing into Chroma...")
        build_chroma_collection(strategy, chunks)
        print(f"[{strategy}] done.")


if __name__ == "__main__":
    main()
