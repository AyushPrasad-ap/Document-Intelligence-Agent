from pathlib import Path

import pymupdf
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DOCUMENT_TYPES = {
    "RBI_Payments_Vision_2025.pdf": "rbi_policy",
    "SEBI_Annual_Report_2024_25.pdf": "regulatory_report",
    "HDFC_Bank_Annual_Report_2024_25.pdf": "bank_annual_report",
    "ICICI_Bank_Annual_Report_2024_25.pdf": "bank_annual_report",
    "Paytm_Annual_Report_2024_25.pdf": "fintech_annual_report",
}


def get_document_type(file_name: str) -> str:
    """
    Return the document type associated with a known PDF.
    Unknown files are classified as 'financial_document'.
    """
    return DOCUMENT_TYPES.get(file_name, "financial_document")


def load_pdf(file_path: Path) -> list[Document]:
    """
    Extract page-level text from a single PDF.

    Each PDF page becomes one LangChain Document so that
    source and page metadata remain available for later RAG citations.
    """
    documents = []

    pdf = pymupdf.open(file_path)

    try:
        total_pages = len(pdf)

        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            # Skip pages that contain no extractable text.
            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": page_index + 1,
                        "document_type": get_document_type(file_path.name),
                        "total_pages": total_pages,
                    },
                )
            )
    finally:
        pdf.close()

    return documents


def load_documents(data_dir: Path = RAW_DATA_DIR) -> list[Document]:
    """
    Load all PDF files from data/raw recursively.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    pdf_files = sorted(data_dir.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        documents = load_pdf(pdf_file)
        all_documents.extend(documents)

        print(f"  Extracted pages: {len(documents)}")

    return all_documents


def print_ingestion_summary(documents: list[Document]) -> None:
    """
    Print a simple ingestion summary for verification.
    """
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total extracted pages: {len(documents)}")

    sources = {}
    total_characters = 0

    for document in documents:
        source = document.metadata["source"]
        sources[source] = sources.get(source, 0) + 1
        total_characters += len(document.page_content)

    print(f"Total extracted characters: {total_characters:,}")
    print("\nDocuments:")

    for source, page_count in sources.items():
        print(f"  - {source}: {page_count} pages")

    print("=" * 60)


if __name__ == "__main__":
    documents = load_documents()
    print_ingestion_summary(documents)

    print("\nSample extracted document:")
    print("-" * 60)
    print(documents[0].page_content[:1000])

    print("\nMetadata:")
    print(documents[0].metadata)
