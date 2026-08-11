from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

from rag_agent.pipeline.query import query_document


class DummyEmbeddingProvider:
    def embed(self, texts):
        return [[0.1, 0.2]]

    def embed_query(self, text):
        return [0.1, 0.2]

    @property
    def dimension(self):
        return 2


class DummyVectorStore:
    def __init__(self, results):
        self._results = results

    def query(self, query_embedding, top_k=5):
        return self._results


class DummyDocStore:
    def __init__(self, docs):
        self._docs = docs

    def get_document(self, doc_id):
        return self._docs.get(doc_id)


class DummyDoc:
    def __init__(self, filename):
        self.filename = filename


class DummyLLM:
    def __init__(self):
        self.messages = None

    def generate(self, prompt, system_prompt=None):
        return "answer"

    def generate_with_messages(self, messages, system_prompt=None):
        self.messages = messages
        return "answer"

    def generate_stream(self, prompt, system_prompt=None):
        yield "a"
        yield "b"

    def generate_stream_with_messages(self, messages, system_prompt=None):
        self.messages = messages
        yield "a"
        yield "b"


def test_query_document_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr("rag_agent.pipeline.query.get_embedding_provider", lambda settings: DummyEmbeddingProvider())
    monkeypatch.setattr("rag_agent.pipeline.query.VectorStore", lambda persist_dir: DummyVectorStore([
        {"chunk_id": "chunk-1", "text": "hello", "doc_id": "doc-1", "score": 0.9, "chunk_index": 3}
    ]))
    monkeypatch.setattr("rag_agent.pipeline.query.DocStore", lambda db_path: DummyDocStore({"doc-1": DummyDoc("file.txt")}))
    dummy_llm = DummyLLM()
    monkeypatch.setattr("rag_agent.pipeline.query.LLMGenerator", lambda **kwargs: dummy_llm)

    answer, sources = query_document(
        "hi",
        top_k=1,
        stream=False,
        conversation_history=[{"role": "user", "content": "previous"}],
    )

    assert answer == "answer"
    assert sources == [{"filename": "file.txt", "chunk_id": "chunk-1", "score": 0.9, "chunk_index": 3}]
    assert dummy_llm.messages[-1]["role"] == "user"
    assert "参考文档" in dummy_llm.messages[-1]["content"]


def test_query_document_returns_empty_message_when_no_results(monkeypatch):
    monkeypatch.setattr("rag_agent.pipeline.query.get_embedding_provider", lambda settings: DummyEmbeddingProvider())
    monkeypatch.setattr("rag_agent.pipeline.query.VectorStore", lambda persist_dir: DummyVectorStore([]))
    monkeypatch.setattr("rag_agent.pipeline.query.DocStore", lambda db_path: DummyDocStore({}))
    monkeypatch.setattr("rag_agent.pipeline.query.LLMGenerator", lambda **kwargs: DummyLLM())

    result = list(query_document("hi", top_k=1, stream=True))

    assert result == [{"type": "token", "data": "文档中没有找到相关信息。"}]
