from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_financial_question_endpoint():
    mock_result = {
        "answer": "HDFC Bank reported a net profit of ₹67,347 crore. [1]",
        "sources": [
            {
                "citation_id": 1,
                "source": "HDFC_Bank_Annual_Report_2024_25.pdf",
                "page": 5,
                "document_type": "bank_annual_report",
                "score": 0.79,
                "chunk_id": 9,
            }
        ],
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "financial_search",
                        "args": {"query": "HDFC Bank net profit FY25"},
                        "id": "test-search",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    }

    with patch(
        "src.api.main.agent_graph.invoke",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What was HDFC Bank's net profit in FY25?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What was HDFC Bank's net profit in FY25?"

    assert data["answer"] == "HDFC Bank reported a net profit of ₹67,347 crore. [1]"

    assert data["tools_used"] == ["financial_search"]

    assert len(data["sources"]) == 1
    assert data["sources"][0]["citation_id"] == 1
    assert data["sources"][0]["source"] == "HDFC_Bank_Annual_Report_2024_25.pdf"
    assert data["sources"][0]["page"] == 5


def test_calculator_question_endpoint():
    mock_result = {
        "answer": "14% of 250 is 35.",
        "sources": [],
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "250 * 0.14"},
                        "id": "test-calculator",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    }

    with patch(
        "src.api.main.agent_graph.invoke",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What is 14% of 250?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "14% of 250 is 35."
    assert data["tools_used"] == ["calculator"]
    assert data["sources"] == []


def test_direct_question_endpoint():
    mock_result = {
        "answer": "I can help with financial documents and calculations.",
        "sources": [],
        "messages": [
            AIMessage(content="I can help with financial documents and calculations.")
        ],
    }

    with patch(
        "src.api.main.agent_graph.invoke",
        return_value=mock_result,
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "What can you help me with?"},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "I can help with financial documents and calculations."

    assert data["tools_used"] == []
    assert data["sources"] == []


def test_empty_question_is_rejected():
    response = client.post(
        "/api/v1/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_question_too_long_is_rejected():
    response = client.post(
        "/api/v1/ask",
        json={"question": "a" * 2001},
    )

    assert response.status_code == 422


def test_agent_failure_returns_500():
    with patch(
        "src.api.main.agent_graph.invoke",
        side_effect=RuntimeError("Test agent failure"),
    ):
        response = client.post(
            "/api/v1/ask",
            json={"question": "Test question"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Agent execution failed."
