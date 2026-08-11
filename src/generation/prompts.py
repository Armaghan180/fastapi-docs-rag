"""Prompt templates for RAG answer generation.

Citation markers use double brackets ([[1]], not [1]) deliberately: FastAPI's docs are
code-heavy, and the model's answers routinely echo code containing single-bracket indexing
(items[0], response_model[i]). A single-bracket citation format would be ambiguous to parse
back out of an answer that also contains real code. [[n]] essentially never collides with
real Python syntax.
"""

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about FastAPI using ONLY the numbered \
documentation excerpts provided in each user message.

Rules:
- Answer using only the information in the excerpts. Do not use outside knowledge about FastAPI.
- Every factual claim must end with a citation marker in double brackets, like [[1]], naming \
the excerpt number(s) it came from. If multiple excerpts support one claim, cite all of them, \
e.g. [[1]][[3]].
- If the excerpts don't contain enough information to answer, say so plainly instead of guessing.
- Prefer concise, direct answers. Include a short code example from the excerpts when one is \
relevant and available, and keep code blocks verbatim from the excerpts.
"""

USER_TEMPLATE = """\
Documentation excerpts:

{context}

Question: {question}

Answer the question using only the excerpts above. Cite sources with double-bracket markers \
like [[1]] after each claim.\
"""


def format_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        breadcrumb = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.doc_path
        parts.append(f"[{i}] ({breadcrumb})\n{chunk.text}")
    return "\n\n".join(parts)
