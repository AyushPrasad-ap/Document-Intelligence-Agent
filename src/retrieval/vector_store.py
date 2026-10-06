import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from src.retrieval.embeddings import (
    create_embedding_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "faiss_store"

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.json"
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.json"

EMBEDDING_DIMENSION = 384
DEFAULT_TOP_K = 5


def load_chunks() -> list[Document]:
    """
    Load the chunked Documents created in Phase 2.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in data
    ]

    if not documents:
        raise ValueError("No chunks found in chunks.json.")

    return documents


def load_embeddings() -> np.ndarray:
    """
    Load the embedding matrix generated in Phase 2.
    """
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embedding file not found: {EMBEDDINGS_FILE}")

    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}"
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
            f"got {embeddings.shape[1]}"
        )

    return embeddings.astype(np.float32)


def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product FAISS index.

    Because our embeddings are normalized, inner product corresponds
    to cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected dimension {EMBEDDING_DIMENSION}, got {embeddings.shape[1]}"
        )

    # Work on a copy so the original embeddings.npy remains unchanged.
    vectors = embeddings.copy()

    # Guarantee normalized vectors before inner-product search.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)

    return index


def save_metadata(
    documents: list[Document],
    output_path: Path = METADATA_FILE,
) -> None:
    """
    Save the mapping between FAISS row position and Document data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = []

    for vector_id, document in enumerate(documents):
        metadata.append(
            {
                "vector_id": vector_id,
                "chunk_id": document.metadata["chunk_id"],
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_metadata(
    input_path: Path = METADATA_FILE,
) -> list[Document]:
    """
    Load the Documents corresponding to FAISS vector positions.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    metadata.sort(key=lambda item: item["vector_id"])

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )
        for item in metadata
    ]

    return documents


def save_faiss_store(
    index: faiss.IndexFlatIP,
    documents: list[Document],
) -> None:
    """
    Persist the FAISS index and corresponding metadata.
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    save_metadata(documents, METADATA_FILE)


def load_faiss_store() -> tuple[faiss.Index, list[Document]]:
    """
    Load the persistent FAISS index and its Document metadata.
    """
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    index = faiss.read_index(str(INDEX_FILE))
    documents = load_metadata()

    if index.ntotal != len(documents):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but metadata contains {len(documents)} documents."
        )

    return index, documents


def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Perform semantic similarity search over the FAISS index.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    index, documents = load_faiss_store()

    embedding_model = create_embedding_model()

    query_embedding = embedding_model.embed_query(query)

    query_vector = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    # Guarantee normalized query vector.
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    for score, vector_id in zip(scores[0], indices[0]):
        if vector_id == -1:
            continue

        document = documents[int(vector_id)]

        results.append(
            {
                "score": float(score),
                "vector_id": int(vector_id),
                "chunk_id": document.metadata["chunk_id"],
                "document": document,
            }
        )

    return results


def print_search_results(
    query: str,
    results: list[dict],
) -> None:
    """
    Pretty-print semantic search results.
    """
    print("\n" + "=" * 70)
    print("FAISS SEARCH RESULTS")
    print("=" * 70)

    print(f"Query: {query}")

    for rank, result in enumerate(results, start=1):
        document = result["document"]

        print("\n" + "-" * 70)
        print(f"Rank          : {rank}")
        print(f"Score         : {result['score']:.4f}")
        print(f"Vector ID     : {result['vector_id']}")
        print(f"Chunk ID      : {result['chunk_id']}")
        print(f"Source        : {document.metadata['source']}")
        print(f"Page          : {document.metadata['page']}")
        print(f"Document Type : {document.metadata['document_type']}")
        print("\nText:")
        print(document.page_content[:500])

    print("=" * 70)


def build_and_save_vector_store() -> None:
    """
    Build the complete FAISS vector store from Phase 2 artifacts.
    """
    print("Loading chunks...")
    documents = load_chunks()

    print(f"Loaded {len(documents):,} chunks.")

    print("\nLoading embeddings...")
    embeddings = load_embeddings()

    print(f"Loaded embeddings: {embeddings.shape}")

    if len(documents) != embeddings.shape[0]:
        raise ValueError(
            f"Chunk count ({len(documents)}) does not match "
            f"embedding count ({embeddings.shape[0]})."
        )

    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)

    print("\nSaving vector store...")
    save_faiss_store(index, documents)

    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE SUMMARY")
    print("=" * 60)

    print(f"Embedding dimension : {index.d}")
    print(f"Vectors indexed     : {index.ntotal}")
    print(f"Index type          : {type(index).__name__}")
    print("Similarity metric   : Inner Product")
    print(f"Index file          : {INDEX_FILE}")
    print(f"Metadata file       : {METADATA_FILE}")

    print("=" * 60)


if __name__ == "__main__":
    build_and_save_vector_store()

    query = "Banking today is more than managing money and fulfilling aspirations."

    results = search(query, top_k=5)
    print_search_results(query, results)
