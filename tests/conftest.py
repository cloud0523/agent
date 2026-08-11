import tempfile
from pathlib import Path

import pytest

from rag_agent.document.schemas import Document
from rag_agent.storage.doc_store import DocStore


SAMPLE_TEXT = """Artificial Intelligence (AI) is a branch of computer science
that aims to create intelligent machines. Machine learning is a subset of AI
that enables systems to learn from data. Deep learning uses neural networks
with many layers. Natural language processing (NLP) focuses on the interaction
between computers and human language. Transformer architecture has
revolutionized NLP since 2017."""


@pytest.fixture
def temp_dir():
    """Create a temporary directory and clean it up after each test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture(scope="session")
def sample_text():
    """A short paragraph about AI for testing."""
    return SAMPLE_TEXT


@pytest.fixture
def sample_document():
    """A pre-built Document with known values."""
    return Document(
        id="test-doc-001",
        filename="test.pdf",
        file_type="pdf",
        file_path="/tmp/test.pdf",
        num_chunks=5,
        status="indexed",
    )


@pytest.fixture
def dummy_embedding():
    """A minimal embedding stub returning a fixed two-dimensional vector."""

    class DummyEmbedding:
        def embed(self, texts):
            return [[0.1, 0.2] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2]

        @property
        def dimension(self):
            return 2

    return DummyEmbedding()


@pytest.fixture
def temp_db():
    """An in-memory DocStore, closed after each test."""
    store = DocStore(":memory:")
    yield store
    store.close()
