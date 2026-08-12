"""Eval metrics, split by how they're computed:

- hit_at_k / reciprocal_rank: was the known-correct doc actually retrieved? Programmatic,
  since we have ground truth -- no need to ask an LLM whether retrieval worked.
- citation_accuracy: does the answer cite the known-correct doc? Also programmatic for the
  same reason.
- judge_answer: correctness (does the answer match the reference answer?) and faithfulness
  (is the answer actually supported by the retrieved context, or did the model add things
  not in it?) can't be checked mechanically, so an LLM judge scores those two, separately,
  on a 0-2 scale. Keeping them separate matters: a wrong-but-grounded answer means retrieval
  failed, while a correct-but-unfaithful answer means the model added outside knowledge --
  different failure modes that a single blended score would hide.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

import config

JUDGE_SYSTEM_PROMPT = """\
You are grading a RAG system's generated answer against a reference answer and the \
retrieved context it was generated from. Score two independent dimensions on a 0/1/2 scale:

correctness: Does the generated answer convey the same key facts as the reference answer?
  2 = fully correct, matches the key facts
  1 = partially correct, or missing some key facts
  0 = wrong, or contradicts the reference answer
If the reference answer states the topic isn't covered by the documentation, score \
correctness 2 only if the generated answer similarly declines or says it isn't covered, \
rather than inventing an answer.

faithfulness: Is every claim in the generated answer actually supported by the provided \
retrieved context (not outside knowledge)?
  2 = fully grounded in the retrieved context
  1 = mostly grounded, but with some unsupported claims
  0 = largely unsupported or hallucinated relative to the context

Respond with strict JSON only: {"correctness": 0|1|2, "faithfulness": 0|1|2, "reasoning": "one sentence"}\
"""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference answer:
{expected_answer}

Retrieved context the generated answer was based on:
{context}

Generated answer:
{generated_answer}\
"""


@dataclass
class JudgeScore:
    correctness: int
    faithfulness: int
    reasoning: str


def hit_at_k(retrieved_doc_paths: list[str], expected_doc_path: str | None) -> bool | None:
    if expected_doc_path is None:
        return None
    return expected_doc_path in retrieved_doc_paths


def reciprocal_rank(retrieved_doc_paths: list[str], expected_doc_path: str | None) -> float | None:
    if expected_doc_path is None:
        return None
    for rank, doc_path in enumerate(retrieved_doc_paths, start=1):
        if doc_path == expected_doc_path:
            return 1 / rank
    return 0.0


def citation_accuracy(cited_doc_paths: list[str], expected_doc_path: str | None) -> bool | None:
    if expected_doc_path is None:
        return None
    return expected_doc_path in cited_doc_paths


def judge_answer(
    client: OpenAI, question: str, expected_answer: str, context: str, generated_answer: str
) -> JudgeScore:
    response = client.chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question,
                    expected_answer=expected_answer,
                    context=context,
                    generated_answer=generated_answer,
                ),
            },
        ],
    )
    data = json.loads(response.choices[0].message.content)
    return JudgeScore(
        correctness=int(data["correctness"]),
        faithfulness=int(data["faithfulness"]),
        reasoning=data.get("reasoning", ""),
    )
