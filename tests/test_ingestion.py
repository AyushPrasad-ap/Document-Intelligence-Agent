from src.ingestion.loader import load_documents

EXPECTED_DOCUMENT_TYPES = {
    "HDFC_Bank_Annual_Report_2024_25.pdf": "bank_annual_report",
    "ICICI_Bank_Annual_Report_2024_25.pdf": "bank_annual_report",
    "Paytm_Annual_Report_2024_25.pdf": "fintech_annual_report",
    "RBI_Payments_Vision_2025.pdf": "rbi_policy",
    "SEBI_Annual_Report_2024_25.pdf": "regulatory_report",
}


def test_documents_are_loaded():
    documents = load_documents()

    assert len(documents) > 0

    for document in documents:
        assert document.page_content.strip()
        assert "source" in document.metadata
        assert "page" in document.metadata
        assert "document_type" in document.metadata
        assert "total_pages" in document.metadata


def test_expected_document_types():
    documents = load_documents()

    sources = {
        document.metadata["source"]: document.metadata["document_type"]
        for document in documents
    }

    for filename, expected_type in EXPECTED_DOCUMENT_TYPES.items():
        assert filename in sources
        assert sources[filename] == expected_type


def test_page_numbers_are_valid():
    documents = load_documents()

    for document in documents:
        page = document.metadata["page"]
        total_pages = document.metadata["total_pages"]

        assert isinstance(page, int)
        assert page >= 1
        assert page <= total_pages
