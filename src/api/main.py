import logging

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import agent_graph
from src.api.schemas import AskRequest, AskResponse, Source

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Intelligence Agent API",
    description=("REST API for the financial document intelligence agent."),
    version="1.0.0",
)


def extract_tools_used(messages) -> list[str]:
    """Extract tool names used during the agent execution."""

    tools_used = []

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]

            if tool_name not in tools_used:
                tools_used.append(tool_name)

    return tools_used


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post("/api/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = agent_graph.invoke(
            {
                "question": request.question,
                "messages": [HumanMessage(content=request.question)],
            }
        )

        sources = [
            Source(
                citation_id=source["citation_id"],
                source=source["source"],
                page=source["page"],
                document_type=source["document_type"],
                score=source.get("score"),
                chunk_id=source.get("chunk_id"),
            )
            for source in result.get("sources", [])
        ]

        tools_used = extract_tools_used(result.get("messages", []))

        return AskResponse(
            question=request.question,
            answer=result.get("answer", ""),
            sources=sources,
            tools_used=tools_used,
        )

    except Exception:
        logger.exception("Agent execution failed.")

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        )
