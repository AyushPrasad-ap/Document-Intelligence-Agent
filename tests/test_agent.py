from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.graph import agent_graph


def make_fake_search_result():
    return {
        "score": 0.91,
        "vector_id": 1,
        "chunk_id": 1,
        "document": Document(
            page_content=(
                "HDFC Bank reported a net profit of ₹67,347.4 crore in FY25."
            ),
            metadata={
                "source": "HDFC_Bank_Annual_Report_2024_25.pdf",
                "page": 203,
                "document_type": "bank_annual_report",
                "chunk_id": 1,
            },
        ),
    }


def test_direct_response_does_not_use_tools():
    mock_model = MagicMock()

    mock_model.invoke.return_value = AIMessage(
        content="I can help with financial documents and calculations."
    )

    with patch(
        "src.agents.graph.get_agent_model",
        return_value=mock_model,
    ):
        result = agent_graph.invoke(
            {
                "question": "What can you help me with?",
                "messages": [HumanMessage(content="What can you help me with?")],
            }
        )

    assert result["answer"] == ("I can help with financial documents and calculations.")

    assert not any(isinstance(message, ToolMessage) for message in result["messages"])

    mock_model.invoke.assert_called_once()


def test_calculator_branch_uses_calculator_tool():
    mock_model = MagicMock()

    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {
                    "expression": "250 * 0.14",
                },
                "id": "call_calculator_test",
                "type": "tool_call",
            }
        ],
    )

    final_response = AIMessage(content="14% of 250 is 35.")

    mock_model.invoke.side_effect = [
        tool_call_response,
        final_response,
    ]

    with patch(
        "src.agents.graph.get_agent_model",
        return_value=mock_model,
    ):
        result = agent_graph.invoke(
            {
                "question": "What is 14% of 250?",
                "messages": [HumanMessage(content="What is 14% of 250?")],
            }
        )

    assert result["answer"] == "14% of 250 is 35."

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].content == "35"

    assert "sources" not in result or result["sources"] == []

    assert mock_model.invoke.call_count == 2


@patch("src.agents.graph.search")
@patch("src.agents.tools.search")
def test_financial_branch_uses_search_and_preserves_sources(
    mock_tool_search,
    mock_graph_search,
):
    fake_result = make_fake_search_result()

    mock_tool_search.return_value = [fake_result]
    mock_graph_search.return_value = [fake_result]

    mock_model = MagicMock()

    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "financial_search",
                "args": {
                    "query": "HDFC Bank net profit FY25",
                },
                "id": "call_financial_test",
                "type": "tool_call",
            }
        ],
    )

    final_response = AIMessage(
        content=(
            "HDFC Bank reported a net profit of "
            "₹67,347.4 crore in FY25."
            "【1†L13-L20】【99†L1-L4】"
        )
    )

    mock_model.invoke.side_effect = [
        tool_call_response,
        final_response,
    ]

    with patch(
        "src.agents.graph.get_agent_model",
        return_value=mock_model,
    ):
        result = agent_graph.invoke(
            {
                "question": "What was HDFC Bank's net profit in FY25?",
                "messages": [
                    HumanMessage(content=("What was HDFC Bank's net profit in FY25?"))
                ],
            }
        )

    assert (
        result["answer"] == "HDFC Bank reported a net profit of "
        "₹67,347.4 crore in FY25.[1]"
    )

    assert "【" not in result["answer"]
    assert "99" not in result["answer"]

    assert len(result["sources"]) == 1

    assert result["sources"][0]["citation_id"] == 1
    assert result["sources"][0]["source"] == "HDFC_Bank_Annual_Report_2024_25.pdf"
    assert result["sources"][0]["page"] == 203
    assert result["sources"][0]["document_type"] == "bank_annual_report"

    mock_tool_search.assert_called_once_with(
        "HDFC Bank net profit FY25",
        top_k=5,
    )

    mock_graph_search.assert_called_once_with(
        "HDFC Bank net profit FY25",
        top_k=5,
    )

    assert mock_model.invoke.call_count == 2


def test_agent_graph_is_compiled():
    assert agent_graph is not None
