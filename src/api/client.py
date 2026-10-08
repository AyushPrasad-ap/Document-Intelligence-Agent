import os

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from src.api.schemas import AskResponse

load_dotenv()


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    return os.getenv(
        "API_BASE_URL",
        DEFAULT_API_BASE_URL,
    ).rstrip("/")


def check_health(timeout: float = 5.0) -> bool:
    """Return True when the FastAPI backend is healthy."""

    try:
        response = requests.get(
            f"{get_api_base_url()}/health",
            timeout=timeout,
        )
        return response.ok
    except requests.RequestException:
        return False


def ask_agent(
    question: str,
    timeout: float = 120.0,
) -> AskResponse:
    """
    Send a question to the FastAPI backend and validate
    the returned response against the API schema.
    """

    response = requests.post(
        f"{get_api_base_url()}/api/v1/ask",
        json={"question": question},
        timeout=timeout,
    )

    response.raise_for_status()

    try:
        return AskResponse.model_validate(response.json())
    except ValidationError as exc:
        raise ValueError("The API returned an invalid response.") from exc
