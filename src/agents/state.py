from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str
    sources: list[dict]
    retrieved_documents: list[dict]
    tools_used: list[str]
