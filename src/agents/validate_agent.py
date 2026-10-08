from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import agent_graph

TEST_CASES = [
    (
        "Financial retrieval",
        "What was HDFC Bank's net profit in FY25?",
        ["financial_search"],
    ),
    (
        "Calculation",
        "What is 14% of 250?",
        ["calculator"],
    ),
    (
        "General response",
        "What can you help me with?",
        [],
    ),
    (
        "Multi-step financial calculation",
        (
            "What was HDFC Bank's FY25 net profit growth, and calculate the absolute increase in net profit from FY24 to FY25?"
        ),
        ["financial_search", "calculator"],
    ),
]


def run_test(name: str, question: str, expected_tools: list[str]):
    result = agent_graph.invoke(
        {
            "question": question,
            "messages": [
                HumanMessage(content=question),
            ],
        }
    )

    actual_tools = []

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                actual_tools.append(tool_call["name"])

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Question:\n{question}")

    print(f"\nExpected tools: {expected_tools}")
    print(f"Actual tools:   {actual_tools}")

    answer = result.get("answer", "")

    print(f"\nFinal answer:\n{answer}")

    sources = result.get("sources", [])

    if sources:
        print("\nApplication-owned sources:")

        for source in sources:
            print(
                f"[{source['citation_id']}] {source['source']} — Page {source['page']}"
            )
    else:
        print("\nApplication-owned sources: None")

    print("\nTool-routing check:")

    if all(tool in actual_tools for tool in expected_tools):
        print("PASS")
    else:
        print("FAIL")

    return {
        "name": name,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "answer": answer,
        "sources": sources,
    }


def main():
    results = []

    for name, question, expected_tools in TEST_CASES:
        results.append(
            run_test(
                name,
                question,
                expected_tools,
            )
        )

    print("\n" + "=" * 80)
    print("AGENT VALIDATION SUMMARY")
    print("=" * 80)

    all_passed = True

    for result in results:
        expected = result["expected_tools"]
        actual = result["actual_tools"]

        passed = all(tool in actual for tool in expected)

        status = "PASS" if passed else "FAIL"

        print(
            f"{status:4} | {result['name']} | Expected: {expected} | Actual: {actual}"
        )

        if not passed:
            all_passed = False

    print("\nOverall:", "PASS" if all_passed else "FAIL")


if __name__ == "__main__":
    main()
