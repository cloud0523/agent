"""Tests for GET /api/documents endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.server import app
from rag_agent.document.schemas import Document


client = TestClient(app)


def _sample_doc(**overrides) -> Document:
    defaults = {
        "id": "doc-1",
        "filename": "report.pdf",
        "file_type": "pdf",
        "file_path": "/tmp/report.pdf",
        "num_chunks": 3,
        "status": "indexed",
    }
    defaults.update(overrides)
    return Document(**defaults)


# ============================================================================
# Fixtures: mock DocStore and VectorStore per test via monkeypatch
# ============================================================================

@pytest.fixture
def mock_store(monkeypatch):
    """Replace DocStore in the documents router with a MagicMock."""
    mock_cls = MagicMock()
    mock_instance = mock_cls.return_value
    monkeypatch.setattr("rag_agent.api.routes.documents.DocStore", mock_cls)
    return mock_instance


@pytest.fixture
def mock_vs(monkeypatch):
    """Replace VectorStore in the documents router with a MagicMock."""
    mock_cls = MagicMock()
    mock_instance = mock_cls.return_value
    monkeypatch.setattr("rag_agent.api.routes.documents.VectorStore", mock_cls)
    return mock_instance


# ============================================================================
#  GET /api/documents
# ============================================================================

class TestListDocuments:
    def test_returns_empty_list_when_no_docs(self, mock_store):
        mock_store.list_documents.return_value = []

        resp = client.get("/api/documents")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_all_documents(self, mock_store):
        docs = [_sample_doc(id="d1"), _sample_doc(id="d2")]
        mock_store.list_documents.return_value = docs

        resp = client.get("/api/documents")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "d1"
        assert data[1]["id"] == "d2"


# ============================================================================
#  GET /api/documents/{document_id}
# ============================================================================

class TestGetDocument:
    def test_returns_document_when_found(self, mock_store):
        mock_store.get_document.return_value = _sample_doc(id="d99")

        resp = client.get("/api/documents/d99")

        assert resp.status_code == 200
        assert resp.json()["id"] == "d99"

    def test_returns_404_when_not_found(self, mock_store):
        mock_store.get_document.return_value = None

        resp = client.get("/api/documents/nonexistent")

        assert resp.status_code == 404
        assert "Document not found" in resp.json()["detail"]

    def test_passes_document_id_to_store(self, mock_store):
        mock_store.get_document.return_value = _sample_doc()

        client.get("/api/documents/abc-xyz")

        mock_store.get_document.assert_called_once_with("abc-xyz")


# ============================================================================
#  DELETE /api/documents/{document_id}
# ============================================================================

class TestDeleteDocument:
    def test_returns_204_when_deleted(self, mock_store, mock_vs):
        mock_store.get_document.return_value = _sample_doc(id="d99")
        mock_store.delete_document.return_value = True

        resp = client.delete("/api/documents/d99")

        assert resp.status_code == 204

    def test_returns_404_when_not_found(self, mock_store, mock_vs):
        mock_store.get_document.return_value = None

        resp = client.delete("/api/documents/nonexistent")

        assert resp.status_code == 404
        assert "Document not found" in resp.json()["detail"]

    def test_calls_vector_store_delete(self, mock_store, mock_vs):
        mock_store.get_document.return_value = _sample_doc(id="d1")

        client.delete("/api/documents/d1")

        mock_vs.delete_document.assert_called_once_with("d1")

    def test_calls_doc_store_delete(self, mock_store, mock_vs):
        mock_store.get_document.return_value = _sample_doc(id="d2")

        client.delete("/api/documents/d2")

        mock_store.delete_document.assert_called_once_with("d2")

    def test_does_not_delete_when_document_missing(self, mock_store, mock_vs):
        mock_store.get_document.return_value = None

        client.delete("/api/documents/ghost")

        mock_vs.delete_document.assert_not_called()
        mock_store.delete_document.assert_not_called()
