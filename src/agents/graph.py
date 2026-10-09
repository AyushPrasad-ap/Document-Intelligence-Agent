from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.citations import normalize_citations
from src.agents.state import AgentState
from src.agents.tools import calculator, financial_search
from src.generation.llm import get_llm
from src.retrieval.vector_store import search

SYSTEM_PROMPT = """You are a financial document intelligence agent.

You have access to two tools:

1. financial_search
   Use this when the user asks about information contained in the
   financial and regulatory documents in the knowledge base.

2. calculator
   Use this when the user asks for arithmetic or numerical calculations.

Rules:
- Use financial_search for questions about the uploaded financial documents.
- Use calculator for mathematical calculations.
- You may use more than one tool when necessary.
- For general conversational questions that do not require either tool,
  answer directly.
- Do not invent financial facts.
- When using financial_search, cite factual claims using [1], [2], etc.
  corresponding to the numbered search results.
- Do not invent source names, page numbers, URLs, or citation mappings.
- Do not use markdown footnotes or external/system-style citation formats.
- Never perform arithmetic yourself when the calculator tool is available.
- For every numerical result produced by calculator, reproduce the
  calculator result exactly.
- Do not recalculate, approximate, or alter calculator results.
- Do not create citations such as 【calculator†result】.
- Only use numeric application citations such as [1], [2], [3].
- Avoid repeating the same search query or equivalent queries.
- For an unsupported question, search the document collection and
  then clearly state when the retrieved evidence does not establish
  the requested fact.
- Do not keep searching merely to force an answer when evidence is absent.
"""


TOOLS = [
    financial_search,
    calculator,
]


@lru_cache(maxsize=1)
def get_agent_model():
    """Return the Groq model configured with agent tools."""
    return get_llm().bind_tools(TOOLS)


def build_finalization_messages(state: AgentState) -> list:
    """Build a concise, tool-free prompt for final answer synthesis."""

    evidence_blocks = []
    remaining_chars = 5000

    # Include tool results, not the previous assistant tool-call messages.
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, ToolMessage):
            continue

        if remaining_chars <= 0:
            break

        content = str(message.content)
        content = content[:remaining_chars]

        tool_name = getattr(message, "name", None) or "tool"

        evidence_blocks.append(f"Tool result ({tool_name}):\n{content}")

        remaining_chars -= len(content)

    evidence_blocks.reverse()

    evidence = (
        "\n\n---\n\n".join(evidence_blocks)
        if evidence_blocks
        else "No tool results are available."
    )

    source_lines = []

    for source in state.get("sources", []):
        source_lines.append(
            f"[{source['citation_id']}] {source['source']}, page {source['page']}"
        )

    source_mapping = (
        "\n".join(source_lines)
        if source_lines
        else "No document sources were retrieved."
    )

    return [
        SystemMessage(
            content=(
                "You are the final-answer synthesizer for a financial "
                "document intelligence application. You have no tools "
                "available in this step. Do not request or invent tool "
                "calls. Answer the original question using only the "
                "provided tool results. If the evidence does not "
                "establish the requested fact, say so clearly. "
                "Use calculator results exactly as returned. Do not "
                "recalculate or change numerical results. Use only "
                "numeric citations that match the authoritative source "
                "mapping. Never invent citations, source names, or pages."
            )
        ),
        HumanMessage(
            content=(
                f"Original question:\n{state.get('question', '')}\n\n"
                f"Available tool results:\n{evidence}\n\n"
                f"Authoritative source mapping:\n{source_mapping}\n\n"
                "Produce the final answer only."
            )
        ),
    ]


def call_model(state: AgentState):
    """Choose tools when permitted, then synthesize a final answer."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get("messages", []),
    ]

    tool_rounds = sum(
        1
        for message in state.get("messages", [])
        if isinstance(message, AIMessage) and message.tool_calls
    )

    max_tool_rounds = 2

    if tool_rounds >= max_tool_rounds:
        response = get_llm().invoke(build_finalization_messages(state))
    else:
        response = get_agent_model().invoke(messages)

    result = {
        "messages": [response],
    }

    if isinstance(response, AIMessage) and not response.tool_calls:
        result["answer"] = normalize_citations(
            response.content,
            state.get("sources", []),
        )

    return result


def collect_financial_sources(state: AgentState):
    """
    Independently collect authoritative source metadata from the
    application retrieval layer.

    Existing sources are preserved across multi-tool agent execution.
    """

    sources = list(state.get("sources", []))
    retrieved_documents = list(state.get("retrieved_documents", []))

    existing_keys = {
        (
            source["source"],
            source["page"],
            source.get("chunk_id"),
        )
        for source in sources
    }

    messages = state.get("messages", [])

    # Inspect ALL AI messages because the agent may use multiple tools
    # during a single question.
    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        for tool_call in message.tool_calls:
            if tool_call["name"] != "financial_search":
                continue

            query = tool_call["args"].get("query", "").strip()

            if not query:
                continue

            results = search(query, top_k=5)

            for result in results:
                document = result["document"]
                metadata = document.metadata

                source_key = (
                    metadata["source"],
                    metadata["page"],
                    metadata.get("chunk_id"),
                )

                if source_key in existing_keys:
                    continue

                citation_id = len(sources) + 1

                source_record = {
                    "citation_id": citation_id,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "document_type": metadata["document_type"],
                    "score": result["score"],
                    "chunk_id": metadata.get("chunk_id"),
                }

                sources.append(source_record)

                retrieved_documents.append(
                    {
                        "citation_id": citation_id,
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "document_type": metadata["document_type"],
                        "score": result["score"],
                        "chunk_id": metadata.get("chunk_id"),
                        "content": document.page_content,
                    }
                )

                existing_keys.add(source_key)

    return {
        "sources": sources,
        "retrieved_documents": retrieved_documents,
    }


def build_agent_graph():
    """Build and compile the LangGraph agent."""

    builder = StateGraph(AgentState)

    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("collect_sources", collect_financial_sources)

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "collect_sources")
    builder.add_edge("collect_sources", "agent")

    return builder.compile()


agent_graph = build_agent_graph()


if __name__ == "__main__":
    print("LangGraph agent compiled successfully.")
