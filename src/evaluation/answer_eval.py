import json
import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.graph import agent_graph

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "tests" / "evaluation_questions.json"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "answer_evaluation_results.json"

# Expected numerical facts and tolerances.
# Rounded figures such as 67,347 are accepted for 67,347.4.
EXPECTED_NUMBERS = {
    "E01": [(67347.4, 0.5)],
    "E02": [(60812.3, 0.5)],
    "E03": [(3.48, 0.02)],
    "E04": [(19.6, 0.1)],
    "E05": [(35.0, 0.001)],
    "E06": [(6535.1, 0.1)],
    "E07": [
        (60812.3, 0.5),
        (67347.4, 0.5),
        (6535.1, 0.2),
        (10.7, 0.1),
    ],
}

NUMBER_PATTERN = re.compile(
    r"(?<![\w\[])"
    r"-?\d[\d,]*(?:\.\d+)?"
    r"(?![\]\w])"
)

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

ABSTENTION_PATTERN = re.compile(
    r"\b("
    r"not found|"
    r"not available|"
    r"not provided|"
    r"not included|"
    r"not present|"
    r"does not contain|"
    r"do not contain|"
    r"doesn't contain|"
    r"don't contain|"
    r"cannot find|"
    r"could not find|"
    r"can't find|"
    r"unable to find|"
    r"cannot verify|"
    r"could not verify|"
    r"can't verify|"
    r"cannot establish|"
    r"insufficient information|"
    r"no information .* (documents|reports|evidence)|"
    r"cannot determine from .* (documents|reports|evidence)|"
    r"can't determine from .* (documents|reports|evidence)|"
    r"none of the documents contain|"
    r")\b",
    re.IGNORECASE,
)


def get_tools_used(messages: list) -> list[str]:
    """Return unique tool names in the order they were called."""
    names = []

    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                name = call["name"]
                if name not in names:
                    names.append(name)

    return names


def numeric_values(answer: str) -> list[float]:
    """Extract ordinary numerical values, excluding years embedded in words
    and numeric citation markers.
    """
    values = []

    for match in NUMBER_PATTERN.finditer(answer):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue

    return values


def check_expected_numbers(question_id: str, answer: str) -> dict:
    expected = EXPECTED_NUMBERS.get(question_id, [])
    actual = numeric_values(answer)

    checks = []

    for target, tolerance in expected:
        matched = any(abs(value - target) <= tolerance for value in actual)

        checks.append(
            {
                "expected": target,
                "tolerance": tolerance,
                "matched": matched,
            }
        )

    return {
        "applicable": bool(expected),
        "passed": all(item["matched"] for item in checks),
        "checks": checks,
        "actual_numbers": actual,
    }


def evaluate_result(item: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    messages = result.get("messages", [])

    tool_trace = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                tool_trace.append(
                    {
                        "type": "tool_call",
                        "name": call["name"],
                        "args": call["args"],
                    }
                )

        elif isinstance(message, ToolMessage):
            tool_trace.append(
                {
                    "type": "tool_result",
                    "content": str(message.content)[:1000],
                }
            )

    expected_tools = set(item.get("expected_tools", []))
    actual_tools = set(get_tools_used(messages))

    tools_ok = actual_tools == expected_tools

    actual_source_pairs = {(source["source"], source["page"]) for source in sources}

    expected_source_pairs = {
        (expected["source"], page)
        for expected in item.get("expected_sources", [])
        for page in expected.get("pages", [])
    }

    has_expected_sources = bool(expected_source_pairs)

    source_hit = (
        any(pair in actual_source_pairs for pair in expected_source_pairs)
        if has_expected_sources
        else None
    )

    citation_ids = {int(match) for match in CITATION_PATTERN.findall(answer)}

    source_ids = {source["citation_id"] for source in sources}

    citations_valid = citation_ids.issubset(source_ids)

    citations_present_when_needed = bool(citation_ids) if has_expected_sources else True

    source_grounding_ok = (
        citations_valid and citations_present_when_needed and (source_hit is not False)
    )

    number_check = check_expected_numbers(item["id"], answer)

    category = item["category"]

    if category == "unsupported":
        acknowledges_limit = bool(ABSTENTION_PATTERN.search(answer))
        unsupported_ok = acknowledges_limit and not citation_ids
    else:
        acknowledges_limit = None
        unsupported_ok = True

    answer_nonempty = bool(answer.strip())

    checks = {
        "answer_nonempty": answer_nonempty,
        "tool_routing": tools_ok,
        "expected_source_retrieved": source_hit,
        "citation_mapping_valid": citations_valid,
        "source_grounding": source_grounding_ok,
        "numerical_facts": (
            number_check["passed"] if number_check["applicable"] else None
        ),
        "unsupported_question_handled": (
            unsupported_ok if category == "unsupported" else None
        ),
    }

    required_checks = [
        answer_nonempty,
        tools_ok,
        source_grounding_ok,
        (number_check["passed"] if number_check["applicable"] else True),
        (unsupported_ok if category == "unsupported" else True),
    ]

    return {
        "id": item["id"],
        "category": category,
        "question": item["question"],
        "expected_answer_points": item.get("expected_answer_points", []),
        "answer": answer,
        "expected_tools": sorted(expected_tools),
        "actual_tools": sorted(actual_tools),
        "tool_trace": tool_trace,
        "sources": sources,
        "numeric_check": number_check,
        "unsupported_phrase_detected": acknowledges_limit,
        "checks": checks,
        "automated_pass": all(required_checks),
        "manual_review_required": True,
    }


def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    selected_ids = {
        value.strip() for value in os.getenv("EVAL_IDS", "").split(",") if value.strip()
    }
    if selected_ids:
        questions = [item for item in questions if item["id"] in selected_ids]

    evaluations = []

    print("=" * 78)
    print("ANSWER QUALITY EVALUATION")
    print(f"Dataset questions: {len(questions)}")
    print("Uses the real LangGraph agent and Groq")
    print("=" * 78)

    for item in questions:
        print(f"\nRunning {item['id']}: {item['question']}")

        try:
            result = agent_graph.invoke(
                {
                    "question": item["question"],
                    "messages": [HumanMessage(content=item["question"])],
                }
            )

            evaluation = evaluate_result(item, result)

        except Exception as exc:  # noqa: BLE001
            evaluation = {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "error": str(exc),
                "automated_pass": False,
                "manual_review_required": True,
            }

        evaluations.append(evaluation)

        print("Answer:", evaluation.get("answer", "[Execution failed]"))
        print("Tools:", evaluation.get("actual_tools", []))
        print("Automated checks:", evaluation.get("checks", {}))
        print("Status:", "PASS" if evaluation["automated_pass"] else "REVIEW")

    passed = sum(item["automated_pass"] for item in evaluations)

    report = {
        "dataset_questions": len(questions),
        "automated_checks_passed": passed,
        "automated_checks_review": len(evaluations) - passed,
        "automated_pass_rate": passed / len(evaluations),
        "note": (
            "Automated checks are screening signals, not a substitute "
            "for manually reviewing answer correctness and evidence."
        ),
        "results": evaluations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("ANSWER EVALUATION SUMMARY")
    print(f"Questions: {len(evaluations)}")
    print(f"Automated passes: {passed}")
    print(f"Need review: {len(evaluations) - passed}")
    print(f"Pass rate: {passed / len(evaluations):.1%}")
    print(f"Report: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
