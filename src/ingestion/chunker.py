import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter used for financial documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
        add_start_index=True,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split page-level documents into smaller chunks while preserving
    the original metadata.
    """
    if not documents:
        raise ValueError("No documents were provided for chunking.")

    splitter = create_text_splitter()

    chunks = splitter.split_documents(documents)

    # Remove accidental empty chunks.
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    # Give every chunk a deterministic ID.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def save_chunks(chunks: list[Document], output_path: Path = CHUNKS_FILE) -> None:
    """
    Save chunk text + metadata as JSON for downstream processing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_chunks = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serialized_chunks, file, indent=2, ensure_ascii=False)


def print_chunking_summary(
    original_documents: list[Document],
    chunks: list[Document],
) -> None:
    """
    Print useful statistics for validating the chunking stage.
    """
    total_characters = sum(len(document.page_content) for document in chunks)

    chunk_lengths = [len(document.page_content) for document in chunks]

    average_chunk_length = total_characters / len(chunks) if chunks else 0

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)

    print(f"Original page documents : {len(original_documents):,}")
    print(f"Generated chunks        : {len(chunks):,}")
    print(f"Total chunk characters  : {total_characters:,}")
    print(f"Average chunk length    : {average_chunk_length:.1f}")
    print(f"Minimum chunk length    : {min(chunk_lengths)}")
    print(f"Maximum chunk length    : {max(chunk_lengths)}")

    print("\nConfiguration:")
    print(f"Chunk size              : {CHUNK_SIZE}")
    print(f"Chunk overlap           : {CHUNK_OVERLAP}")

    print("=" * 60)

    print("\nSample chunk:")
    print("-" * 60)
    print(chunks[0].page_content[:1000])

    print("\nSample metadata:")
    print(chunks[0].metadata)


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    save_chunks(chunks)

    print_chunking_summary(documents, chunks)

    print(f"\nSaved chunks to: {CHUNKS_FILE}")
