import numpy as np

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_documents
from src.retrieval.embeddings import (
    EMBEDDING_MODEL_NAME,
    create_embedding_model,
    embed_documents,
)


def test_chunking_produces_valid_chunks():
    documents = load_documents()
    chunks = chunk_documents(documents)

    assert len(chunks) > len(documents)

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert "document_type" in chunk.metadata
        assert "chunk_id" in chunk.metadata


def test_embedding_model_loads():
    model = create_embedding_model()

    assert model.model_name == EMBEDDING_MODEL_NAME


def test_embeddings_have_expected_shape():
    documents = load_documents()

    # Keep the test lightweight: use only two chunks.
    chunks = chunk_documents(documents)[:2]

    model = create_embedding_model()
    embeddings = embed_documents(chunks, model)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384


def test_embeddings_are_normalized():
    documents = load_documents()
    chunks = chunk_documents(documents)[:2]

    model = create_embedding_model()
    embeddings = embed_documents(chunks, model)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-4)
