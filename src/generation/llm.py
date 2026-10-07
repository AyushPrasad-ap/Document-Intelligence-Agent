import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Create and cache the Groq-hosted LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("Groq API key not found. Set GROQ_API_KEY in the .env file.")

    return ChatGroq(
        model=get_model_name(),
        temperature=0.2,
        max_tokens=1024,
        max_retries=3,
        timeout=60,
    )


if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke("Respond with exactly: Groq connection successful.")

    print(response.content)
