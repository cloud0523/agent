from __future__ import annotations

from rag_agent.retrieval.retriever import Retriever


class DummyEmbeddingProvider:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed(self, texts):
        return self._embeddings

    def embed_query(self, text):
        return [0.0, 0.0]

    @property
    def dimension(self):
        return 2


class DummyVectorStore:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def query(self, query_embedding, top_k=5):
        self.calls.append((query_embedding, top_k))
        return self._results


def test_retrieve_uses_first_embedding_and_returns_results():
    embedding_provider = DummyEmbeddingProvider([[0.1, 0.2]])
    vector_store = DummyVectorStore([
        {
            "chunk_id": "chunk-1",
            "text": "hello world",
            "doc_id": "doc-1",
            "score": 0.95,
            "distance": 0.05,
            "chunk_index": 0,
        }
    ])

    retriever = Retriever(embedding_provider=embedding_provider, vector_store=vector_store)
    results = retriever.retrieve("what is this", top_k=3)

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "hello world",
            "doc_id": "doc-1",
            "score": 0.95,
            "chunk_index": 0,
        }
    ]
    assert vector_store.calls == [([0.1, 0.2], 3)]


def test_retrieve_with_threshold_filters_low_score_results():
    embedding_provider = DummyEmbeddingProvider([[0.3, 0.4]])
    vector_store = DummyVectorStore([
        {
            "chunk_id": "chunk-1",
            "text": "high score",
            "doc_id": "doc-1",
            "score": 0.8,
            "distance": 0.2,
            "chunk_index": 1,
        },
        {
            "chunk_id": "chunk-2",
            "text": "low score",
            "doc_id": "doc-1",
            "score": 0.6,
            "distance": 0.4,
            "chunk_index": 2,
        },
    ])

    retriever = Retriever(embedding_provider=embedding_provider, vector_store=vector_store)
    results = retriever.retrieve_with_threshold("what is this", top_k=2, threshold=0.7)

    assert results == [
        {
            "chunk_id": "chunk-1",
            "text": "high score",
            "doc_id": "doc-1",
            "score": 0.8,
            "chunk_index": 1,
        }
    ]


def test_retrieve_with_threshold_returns_empty_when_all_filtered_out():
    embedding_provider = DummyEmbeddingProvider([[0.3, 0.4]])
    vector_store = DummyVectorStore([
        {
            "chunk_id": "chunk-1",
            "text": "low score",
            "doc_id": "doc-1",
            "score": 0.5,
            "distance": 0.5,
        }
    ])

    retriever = Retriever(embedding_provider=embedding_provider, vector_store=vector_store)
    results = retriever.retrieve_with_threshold("what is this", top_k=2, threshold=0.7)

    assert results == []
