import json
from pathlib import Path

from src.retrieval.vector_store import search

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "tests" / "evaluation_questions.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation_results.json"

TOP_K = 5


def evaluate_query(item: dict) -> dict:
    """Evaluate whether the top-k FAISS results contain expected evidence."""

    results = search(item["question"], top_k=TOP_K)

    expected_pages = {
        (source["source"], page)
        for source in item.get("expected_sources", [])
        for page in source.get("pages", [])
    }

    retrieved = []

    for rank, result in enumerate(results, start=1):
        document = result["document"]
        metadata = document.metadata

        source = metadata["source"]
        page = metadata["page"]

        relevant = (source, page) in expected_pages

        retrieved.append(
            {
                "rank": rank,
                "source": source,
                "page": page,
                "score": float(result["score"]),
                "relevant": relevant,
            }
        )

    relevant_ranks = [result["rank"] for result in retrieved if result["relevant"]]

    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_pages": [
            {
                "source": source,
                "page": page,
            }
            for source, page in sorted(expected_pages)
        ],
        "retrieved": retrieved,
        "hit_at_5": first_relevant_rank is not None,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
    }


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    # Evaluate questions with explicitly specified relevant pages.
    evaluation_set = [
        item
        for item in questions
        if item.get("expected_sources")
        and item["category"]
        in {
            "financial_retrieval",
            "multi_step",
        }
    ]

    if not evaluation_set:
        raise ValueError("No evaluation questions have expected source pages.")

    results = []

    print("=" * 75)
    print("RETRIEVAL EVALUATION")
    print(f"Questions: {len(evaluation_set)}")
    print(f"Top-k: {TOP_K}")
    print("=" * 75)

    for item in evaluation_set:
        result = evaluate_query(item)
        results.append(result)

        print(f"\n{result['id']} — {result['category']}")
        print(f"Question: {result['question']}")
        print(f"Hit@{TOP_K}: {result['hit_at_5']}")

        if result["first_relevant_rank"] is not None:
            print(f"First relevant result rank: {result['first_relevant_rank']}")
        else:
            print("First relevant result rank: Not found")

        for retrieved in result["retrieved"]:
            marker = "RELEVANT" if retrieved["relevant"] else "other"
            print(
                f"  #{retrieved['rank']} "
                f"{retrieved['source']} | "
                f"Page {retrieved['page']} | "
                f"Score {retrieved['score']:.4f} | "
                f"{marker}"
            )

    hit_rate = sum(result["hit_at_5"] for result in results) / len(results)

    mrr = sum(result["reciprocal_rank"] for result in results) / len(results)

    report = {
        "top_k": TOP_K,
        "evaluated_questions": len(results),
        "hit_at_5_rate": hit_rate,
        "mean_reciprocal_rank": mrr,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("SUMMARY")
    print(f"Evaluated questions: {len(results)}")
    print(f"Hit@{TOP_K}: {hit_rate:.1%}")
    print(f"MRR: {mrr:.4f}")
    print(f"Report saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
