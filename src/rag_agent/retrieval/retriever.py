from __future__ import annotations

from typing import Any

from rag_agent.config import settings

from rag_agent.embeddings.base import EmbeddingProvider

from rag_agent.storage.vector_store import VectorStore

class Retriever:
    """Retrieve the most relevant document chunks for a query."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Embed the query and return the top matching chunks."""
        effective_top_k = top_k if top_k is not None else settings.top_k
        embeddings = self.embedding_provider.embed([query])
        if not embeddings:
            return []

        query_embedding = embeddings[0]
        results = self.vector_store.query(query_embedding, top_k=effective_top_k)
        if not results:
            return []

        return [
            {
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text"),
                "doc_id": item.get("doc_id"),
                "score": item.get("score"),
                "chunk_index": item.get("chunk_index"),
            }
            for item in results
        ]

    def retrieve_with_threshold(self, query, top_k=None, threshold=None):
        effective_threshold = threshold if threshold is not None else settings.similarity_threshold
        results = self.retrieve(query, top_k=top_k)  # 直接传，让 retrieve 自己处理 None
        return [item for item in results if item["score"] >= effective_threshold]
