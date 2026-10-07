from typing import Any

from langchain_core.documents import Document

from src.generation.llm import get_llm
from src.generation.prompts import RAG_PROMPT
from src.retrieval.vector_store import search

DEFAULT_TOP_K = 5


def format_context(
    results: list[dict[str, Any]],
) -> str:
    """
    Convert retrieved FAISS results into citation-aware context.
    """
    context_blocks = []

    for citation_id, result in enumerate(results, start=1):
        document: Document = result["document"]

        source = document.metadata["source"]
        page = document.metadata["page"]
        document_type = document.metadata["document_type"]

        block = (
            f"[{citation_id}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Document type: {document_type}\n"
            f"Relevance score: {result['score']:.4f}\n\n"
            f"{document.page_content}"
        )

        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_sources(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create structured source information independently of the LLM.
    """
    sources = []

    for citation_id, result in enumerate(results, start=1):
        document: Document = result["document"]

        sources.append(
            {
                "citation_id": citation_id,
                "source": document.metadata["source"],
                "page": document.metadata["page"],
                "document_type": document.metadata["document_type"],
                "score": round(result["score"], 4),
                "chunk_id": document.metadata["chunk_id"],
            }
        )

    return sources


def ask_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    Execute the complete RAG pipeline:

    Question
        ↓
    FAISS retrieval
        ↓
    Context construction
        ↓
    Groq
        ↓
    Grounded answer + structured sources
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    results = search(
        query=question,
        top_k=top_k,
    )

    if not results:
        return {
            "question": question,
            "answer": (
                "I could not find relevant information in "
                "the indexed financial documents."
            ),
            "sources": [],
            "retrieved_documents": [],
        }

    context = format_context(results)

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=question,
    )

    llm = get_llm()

    response = llm.invoke(messages)

    answer = response.content.strip()

    return {
        "question": question,
        "answer": answer,
        "sources": build_sources(results),
        "retrieved_documents": results,
    }


def print_rag_response(response: dict[str, Any]) -> None:
    """
    Display a RAG response in a human-readable format.
    """
    print("\n" + "=" * 70)
    print("RAG RESPONSE")
    print("=" * 70)

    print(f"\nQuestion:\n{response['question']}")

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")

    for source in response["sources"]:
        print(
            f"[{source['citation_id']}] "
            f"{source['source']} — "
            f"Page {source['page']} "
            f"(score={source['score']})"
        )

    print("=" * 70)


if __name__ == "__main__":
    # question = "What was HDFC Bank's market share of banking deposits in the last financial year?"
    # question = "What artificial intelligence initiatives are mentioned in the ICICI Bank annual report?"
    question = (
        "What was HDFC Bank's total revenue in FY2010 according to these documents?"
    )

    response = ask_question(question)

    print_rag_response(response)
