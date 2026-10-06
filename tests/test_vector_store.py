from src.retrieval.vector_store import (
    EMBEDDING_DIMENSION,
    build_faiss_index,
    load_embeddings,
    load_faiss_store,
    search,
)


def test_faiss_index_has_expected_shape():
    embeddings = load_embeddings()

    index = build_faiss_index(embeddings)

    assert index.d == EMBEDDING_DIMENSION
    assert index.ntotal == embeddings.shape[0]


def test_faiss_store_metadata_alignment():
    index, documents = load_faiss_store()

    assert index.ntotal == len(documents)

    for vector_id in [0, 100, 1000]:
        assert vector_id < index.ntotal

        document = documents[vector_id]

        assert document.metadata["chunk_id"] == vector_id
        assert document.metadata["source"]
        assert document.metadata["page"] >= 1


def test_faiss_store_is_persistent():
    index, documents = load_faiss_store()

    assert index.ntotal == len(documents)
    assert index.d == EMBEDDING_DIMENSION


def test_semantic_search_returns_relevant_hdfc_document():
    query = "Banking today is more than managing money and fulfilling aspirations."

    results = search(query, top_k=5)

    assert len(results) == 5

    sources = [result["document"].metadata["source"] for result in results]

    assert "HDFC_Bank_Annual_Report_2024_25.pdf" in sources

    hdfc_results = [
        result
        for result in results
        if result["document"].metadata["source"]
        == "HDFC_Bank_Annual_Report_2024_25.pdf"
    ]

    assert len(hdfc_results) > 0

    for result in hdfc_results:
        document = result["document"]

        assert document.metadata["document_type"] == "bank_annual_report"
        assert document.metadata["page"] >= 1
        assert result["score"] > 0


def test_hdfc_specific_query_retrieves_hdfc():
    query = "What was HDFC Bank's market share of banking deposits and branches?"

    results = search(query, top_k=5)

    sources = [result["document"].metadata["source"] for result in results]

    assert "HDFC_Bank_Annual_Report_2024_25.pdf" in sources

    hdfc_result = next(
        result
        for result in results
        if result["document"].metadata["source"]
        == "HDFC_Bank_Annual_Report_2024_25.pdf"
    )

    assert hdfc_result["score"] > 0
