from unittest.mock import MagicMock, patch

from src.generation.rag_chain import (
    build_sources,
    format_context,
)


def test_context_contains_source_and_page():
    results = [
        {
            "score": 0.85,
            "vector_id": 10,
            "chunk_id": 10,
            "document": MagicMock(
                page_content="Revenue increased significantly.",
                metadata={
                    "source": "HDFC_Bank_Annual_Report_2024_25.pdf",
                    "page": 47,
                    "document_type": "bank_annual_report",
                    "chunk_id": 10,
                },
            ),
        }
    ]

    context = format_context(results)

    assert "[1]" in context
    assert "HDFC_Bank_Annual_Report_2024_25.pdf" in context
    assert "Page: 47" in context
    assert "Revenue increased significantly." in context


def test_sources_are_built_without_llm():
    results = [
        {
            "score": 0.91,
            "vector_id": 25,
            "chunk_id": 25,
            "document": MagicMock(
                page_content="Some financial information.",
                metadata={
                    "source": "SEBI_Annual_Report_2024_25.pdf",
                    "page": 83,
                    "document_type": "regulatory_report",
                    "chunk_id": 25,
                },
            ),
        }
    ]

    sources = build_sources(results)

    assert len(sources) == 1
    assert sources[0]["citation_id"] == 1
    assert sources[0]["source"] == "SEBI_Annual_Report_2024_25.pdf"
    assert sources[0]["page"] == 83
    assert sources[0]["document_type"] == "regulatory_report"


@patch("src.generation.rag_chain.get_llm")
@patch("src.generation.rag_chain.search")
def test_rag_pipeline_returns_answer_and_sources(
    mock_search,
    mock_get_llm,
):
    mock_search.return_value = [
        {
            "score": 0.88,
            "vector_id": 1,
            "chunk_id": 1,
            "document": MagicMock(
                page_content="Revenue increased by 14%.",
                metadata={
                    "source": "Paytm_Annual_Report_2024_25.pdf",
                    "page": 100,
                    "document_type": "fintech_annual_report",
                    "chunk_id": 1,
                },
            ),
        }
    ]

    mock_llm = MagicMock()

    mock_response = MagicMock()
    mock_response.content = "Revenue increased by 14%. [1]"

    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    from src.generation.rag_chain import ask_question

    result = ask_question(
        "What happened to revenue?",
        top_k=1,
    )

    assert result["answer"] == "Revenue increased by 14%. [1]"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "Paytm_Annual_Report_2024_25.pdf"

    mock_search.assert_called_once()
    mock_llm.invoke.assert_called_once()
