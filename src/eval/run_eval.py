"""Runs the eval question set against every (chunking_strategy, retrieval_method) config in
config.STRATEGIES, scoring retrieval (Recall@k, MRR), citation accuracy, and LLM-judged
generation quality (correctness, faithfulness). Saves per-config raw results plus an
aggregate comparison report.
"""

import json

from openai import OpenAI
from tqdm import tqdm

import config
from src.generation import prompts
from src.generation.answer import generate_answer
from src.retrieval.base import get_retriever

from . import metrics, report


def load_questions() -> list[dict]:
    with config.EVAL_QUESTIONS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def run_config(chunking_strategy: str, retrieval_method: str, questions: list[dict], judge_client: OpenAI) -> list[dict]:
    retriever = get_retriever(chunking_strategy, retrieval_method)
    results = []

    for q in tqdm(questions, desc=f"{chunking_strategy}/{retrieval_method}"):
        retrieved = retriever.retrieve(q["question"], top_k=config.TOP_K)
        retrieved_doc_paths = [c.doc_path for c in retrieved]

        rag_answer = generate_answer(q["question"], retrieved)
        cited_doc_paths = [c.doc_path for c in rag_answer.citations]

        judge = metrics.judge_answer(
            judge_client,
            question=q["question"],
            expected_answer=q["expected_answer"],
            context=prompts.format_context(retrieved),
            generated_answer=rag_answer.answer,
        )

        results.append(
            {
                "id": q["id"],
                "topic": q["topic"],
                "question": q["question"],
                "expected_doc_path": q["source_doc_path"],
                "retrieved_doc_paths": retrieved_doc_paths,
                "hit": metrics.hit_at_k(retrieved_doc_paths, q["source_doc_path"]),
                "reciprocal_rank": metrics.reciprocal_rank(retrieved_doc_paths, q["source_doc_path"]),
                "cited_doc_paths": cited_doc_paths,
                "citation_accuracy": metrics.citation_accuracy(cited_doc_paths, q["source_doc_path"]),
                "generated_answer": rag_answer.answer,
                "correctness": judge.correctness,
                "faithfulness": judge.faithfulness,
                "judge_reasoning": judge.reasoning,
            }
        )

    return results


def main() -> None:
    questions = load_questions()
    judge_client = OpenAI(api_key=config.OPENAI_API_KEY)
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    for chunking_strategy, retrieval_method in config.STRATEGIES:
        key = f"{chunking_strategy}+{retrieval_method}"
        results = run_config(chunking_strategy, retrieval_method, questions, judge_client)
        all_results[key] = results

        out_path = config.EVAL_RESULTS_DIR / f"{chunking_strategy}_{retrieval_method}.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report.build_report(all_results)


if __name__ == "__main__":
    main()
