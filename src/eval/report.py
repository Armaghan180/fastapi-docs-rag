"""Aggregates per-question eval results into a markdown comparison table across strategies."""

import config


def _aggregate(results: list[dict]) -> dict:
    answerable = [r for r in results if r["expected_doc_path"] is not None]
    n = len(answerable)

    return {
        "n_questions": len(results),
        "recall_at_k": (sum(1 for r in answerable if r["hit"]) / n) if n else float("nan"),
        "mrr": (sum(r["reciprocal_rank"] for r in answerable) / n) if n else float("nan"),
        "citation_accuracy": (sum(1 for r in answerable if r["citation_accuracy"]) / n) if n else float("nan"),
        "mean_correctness": sum(r["correctness"] for r in results) / len(results),
        "mean_faithfulness": sum(r["faithfulness"] for r in results) / len(results),
    }


def build_report(all_results: dict[str, list[dict]]) -> str:
    lines = [
        f"# Eval Report (top_k={config.TOP_K}, n={len(next(iter(all_results.values())))})",
        "",
        "Recall@k / MRR are computed only over questions with a known-correct source doc "
        "(the out-of-scope question is excluded from those two columns). Correctness and "
        "faithfulness are averaged over all questions, on a 0-2 LLM-judge scale.",
        "",
        "| Config | Recall@k | MRR | Citation Accuracy | Correctness (0-2) | Faithfulness (0-2) |",
        "|---|---|---|---|---|---|",
    ]
    for key, results in all_results.items():
        agg = _aggregate(results)
        lines.append(
            f"| {key} | {agg['recall_at_k']:.2f} | {agg['mrr']:.2f} | "
            f"{agg['citation_accuracy']:.2f} | {agg['mean_correctness']:.2f} | {agg['mean_faithfulness']:.2f} |"
        )

    report_text = "\n".join(lines)
    out_path = config.EVAL_RESULTS_DIR / "report.md"
    out_path.write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    return report_text
