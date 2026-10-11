from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the local Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
        show_progress=True,
    )


def embed_documents(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> np.ndarray:
    """
    Generate embeddings for all document chunks.
    """
    if not documents:
        raise ValueError("No documents provided for embedding.")

    texts = [document.page_content for document in documents]

    embeddings = embedding_model.embed_documents(texts)

    return np.asarray(embeddings, dtype=np.float32)


def save_embeddings(
    embeddings: np.ndarray,
    output_path: Path = EMBEDDINGS_FILE,
) -> None:
    """
    Save the embedding matrix as a NumPy array.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, embeddings)


def print_embedding_summary(
    documents: list[Document],
    embeddings: np.ndarray,
) -> None:
    """
    Print embedding statistics for validation.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)

    print(f"Model                   : {EMBEDDING_MODEL_NAME}")
    print(f"Number of chunks        : {len(documents):,}")
    print(f"Embedding shape         : {embeddings.shape}")
    print(f"Embedding dtype         : {embeddings.dtype}")

    norms = np.linalg.norm(embeddings, axis=1)

    print(f"Minimum vector norm     : {norms.min():.6f}")
    print(f"Maximum vector norm     : {norms.max():.6f}")
    print(f"Average vector norm     : {norms.mean():.6f}")

    print("=" * 60)

    print("\nFirst embedding:")
    print(embeddings[0][:10])


def load_and_prepare_chunks() -> list[Document]:
    """
    Recreate the chunk list from the Phase 1 source documents.
    """
    documents = load_documents()
    return chunk_documents(documents)


if __name__ == "__main__":
    print("Preparing chunks...")

    chunks = load_and_prepare_chunks()

    print(f"Prepared {len(chunks):,} chunks.")

    print("\nLoading embedding model...")
    embedding_model = create_embedding_model()

    print("\nGenerating embeddings...")
    embeddings = embed_documents(
        chunks,
        embedding_model,
    )

    save_embeddings(embeddings)

    print_embedding_summary(
        chunks,
        embeddings,
    )

    print(f"\nSaved embeddings to: {EMBEDDINGS_FILE}")
