"""Tests for POST /api/query endpoint — sync & SSE streaming."""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_agent.api.server import app


client = TestClient(app)

QUERY_ROUTE = "/api/query"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_query_module(monkeypatch, *, stream_items=None, sync_result=None):
    """Mock rag_agent.api.routes.query.query_document.

    Returns the mock so callers can inspect it after the request.
    """
    mock = MagicMock()

    if stream_items is not None:

        def _stream(*args, **kwargs):
            for item in stream_items:
                yield item

        mock.side_effect = _stream
    elif sync_result is not None:
        mock.return_value = sync_result
    else:
        # Default: sync mode, returns simple answer
        mock.return_value = ("Some answer", [{"file": "test.txt", "text": "..."}])

    monkeypatch.setattr("rag_agent.api.routes.query.query_document", mock)
    return mock


# ============================================================================
#  POST /api/query  (non-stream)
# ============================================================================

class TestQuerySync:
    """Synchronous (non-streaming) query requests."""

    def test_returns_answer_and_sources(self, monkeypatch):
        _mock_query_module(monkeypatch, sync_result=("42", [{"source": "doc1"}]))

        resp = client.post(QUERY_ROUTE, json={"question": "What is the answer?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "42"
        assert data["sources"] == [{"source": "doc1"}]

    def test_passes_default_top_k(self, monkeypatch):
        mock = _mock_query_module(monkeypatch)

        client.post(QUERY_ROUTE, json={"question": "hi"})

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    def test_passes_custom_top_k(self, monkeypatch):
        mock = _mock_query_module(monkeypatch)

        client.post(QUERY_ROUTE, json={"question": "hi", "top_k": 3})

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    def test_passes_stream_false(self, monkeypatch):
        mock = _mock_query_module(monkeypatch)

        client.post(QUERY_ROUTE, json={"question": "hi"})

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["stream"] is False

    def test_passes_empty_conversation_history_by_default(self, monkeypatch):
        mock = _mock_query_module(monkeypatch)

        client.post(QUERY_ROUTE, json={"question": "hi"})

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["conversation_history"] is None

    def test_passes_conversation_history_when_provided(self, monkeypatch):
        mock = _mock_query_module(monkeypatch)
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "ok"}]

        client.post(QUERY_ROUTE, json={"question": "hi", "conversation_history": history})

        call_kwargs = mock.call_args.kwargs
        assert call_kwargs["conversation_history"] == history


# ============================================================================
#  POST /api/query  (SSE streaming)
# ============================================================================

class TestQueryStream:
    """Streaming (SSE) query requests."""

    def test_returns_event_stream_content_type(self, monkeypatch):
        _mock_query_module(monkeypatch, stream_items=[])

        resp = client.post(QUERY_ROUTE, json={"question": "hi", "stream": True})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_yields_sse_formatted_events(self, monkeypatch):
        stream_items = [
            {"type": "token", "data": "Hello"},
            {"type": "token", "data": " World"},
            {"type": "done", "data": ""},
        ]
        _mock_query_module(monkeypatch, stream_items=stream_items)

        resp = client.post(QUERY_ROUTE, json={"question": "hi", "stream": True})

        # Parse SSE lines — each item should be "data: <json>\n\n"
        body = resp.text
        lines = body.strip().split("\n\n")
        events = []
        for line in lines:
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) == 3
        assert events[0] == {"type": "token", "data": "Hello"}
        assert events[1] == {"type": "token", "data": " World"}
        assert events[2] == {"type": "done", "data": ""}


# ============================================================================
#  Error / validation
# ============================================================================

class TestQueryValidation:
    """Input validation for the query endpoint."""

    def test_empty_question_returns_400(self, monkeypatch):
        _mock_query_module(monkeypatch)

        resp = client.post(QUERY_ROUTE, json={"question": "   "})

        assert resp.status_code == 400
        assert "question is required" in resp.json()["detail"]

    def test_missing_question_field_returns_422(self):
        """Pydantic validation should reject a body without 'question'."""
        resp = client.post(QUERY_ROUTE, json={"top_k": 5})
        assert resp.status_code == 422
