from unittest.mock import MagicMock, patch

import pytest
import requests

from src.api.client import ask_agent, check_health


def test_check_health_returns_true_when_api_is_healthy():
    mock_response = MagicMock()
    mock_response.ok = True

    with patch(
        "src.api.client.requests.get",
        return_value=mock_response,
    ):
        assert check_health() is True


def test_check_health_returns_false_when_api_is_unavailable():
    with patch(
        "src.api.client.requests.get",
        side_effect=requests.ConnectionError("Connection refused"),
    ):
        assert check_health() is False


def test_ask_agent_validates_successful_response():
    mock_response = MagicMock()

    mock_response.json.return_value = {
        "question": "What is 14% of 250?",
        "answer": "14% of 250 is 35.",
        "sources": [],
        "tools_used": ["calculator"],
    }

    with patch(
        "src.api.client.requests.post",
        return_value=mock_response,
    ):
        result = ask_agent("What is 14% of 250?")

    assert result.question == "What is 14% of 250?"
    assert result.answer == "14% of 250 is 35."
    assert result.sources == []
    assert result.tools_used == ["calculator"]


def test_ask_agent_raises_for_http_error():
    mock_response = MagicMock()

    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    with (
        patch(
            "src.api.client.requests.post",
            return_value=mock_response,
        ),
        pytest.raises(requests.HTTPError),
    ):
        ask_agent("Test question")


def test_ask_agent_rejects_invalid_response():
    mock_response = MagicMock()

    mock_response.json.return_value = {
        "question": "Test question",
        "answer": 123,
        "sources": "not-a-list",
        "tools_used": [],
    }

    with (
        patch(
            "src.api.client.requests.post",
            return_value=mock_response,
        ),
        pytest.raises(
            ValueError,
            match="invalid response",
        ),
    ):
        ask_agent("Test question")


def test_ask_agent_uses_expected_endpoint_and_payload():
    mock_response = MagicMock()

    mock_response.json.return_value = {
        "question": "Hello",
        "answer": "Hello!",
        "sources": [],
        "tools_used": [],
    }

    with patch(
        "src.api.client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        ask_agent("Hello", timeout=60)

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/api/v1/ask",
        json={"question": "Hello"},
        timeout=60,
    )
