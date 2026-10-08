from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.graph import agent_graph


def run_agent(question: str):
    """Run the LangGraph agent for a single user question."""

    result = agent_graph.invoke(
        {
            "question": question,
            "messages": [
                HumanMessage(content=question),
            ],
        }
    )

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    for message in result["messages"]:
        message_type = type(message).__name__

        print(f"\n[{message_type}]")

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"Tool called: {tool_call['name']}")
                print(f"Arguments: {tool_call['args']}")

        if isinstance(message, ToolMessage):
            print(f"Tool result:\n{message.content}")

        elif getattr(message, "content", None):
            print(f"Content:\n{message.content}")

    print("\n" + "-" * 70)
    print("APPLICATION-FORMATTED ANSWER")
    print("-" * 70)
    print(result.get("answer", ""))

    print("\n" + "-" * 70)
    print("APPLICATION-OWNED SOURCES")
    print("-" * 70)

    for source in result.get("sources", []):
        print(
            f"[{source['citation_id']}] "
            f"{source['source']} — "
            f"Page {source['page']} "
            f"({source['document_type']})"
        )

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    run_agent("What was HDFC Bank's net profit in FY25?")  # financial-search branch
    # run_agent("What is 14% of 250?")                          # calculator branch
    # run_agent("What can you help me with?")                   # direct-LMM branch
