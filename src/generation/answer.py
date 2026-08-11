"""Generates a RAG answer from retrieved chunks and resolves its citation markers back to
real source URLs -- citations are structural (parsed from the model's [[n]] markers and
mapped to the chunk that was actually retrieved), not just the model asserting a source.
"""

import re
from dataclasses import dataclass

from openai import OpenAI

import config
from src.retrieval.base import RetrievedChunk

from . import prompts

CITATION_RE = re.compile(r"\[\[(\d+)\]\]")


@dataclass
class Citation:
    index: int
    url: str
    doc_path: str
    heading_path: list[str]


@dataclass
class RAGAnswer:
    query: str
    answer: str  # citation markers rewritten as clickable markdown links
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


def _linkify_citations(text: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Replaces each [[n]] marker with a markdown link to that chunk's URL, and returns the
    deduplicated, in-order-of-first-appearance list of citations actually used."""
    seen: dict[int, Citation] = {}

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if not (1 <= n <= len(chunks)):
            return match.group(0)  # leave an out-of-range marker untouched rather than guess
        chunk = chunks[n - 1]
        if n not in seen:
            seen[n] = Citation(index=n, url=chunk.url, doc_path=chunk.doc_path, heading_path=chunk.heading_path)
        return f"[[{n}]]({chunk.url})"

    linked_text = CITATION_RE.sub(_replace, text)
    return linked_text, list(seen.values())


def generate_answer(query: str, retrieved_chunks: list[RetrievedChunk], client: OpenAI | None = None) -> RAGAnswer:
    client = client or OpenAI(api_key=config.OPENAI_API_KEY)
    context = prompts.format_context(retrieved_chunks)

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        temperature=0,  # deterministic answers, so eval runs are comparable across strategies
        messages=[
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.USER_TEMPLATE.format(context=context, question=query)},
        ],
    )
    raw_answer = response.choices[0].message.content
    linked_answer, citations = _linkify_citations(raw_answer, retrieved_chunks)

    return RAGAnswer(query=query, answer=linked_answer, citations=citations, retrieved_chunks=retrieved_chunks)
