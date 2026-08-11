"""Tests for POST /api/ingest/* endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_agent.api.server import app
from rag_agent.document.schemas import Document


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_document(**overrides) -> Document:
    defaults = {
        "id": "abc-123",
        "filename": "test.pdf",
        "file_type": "pdf",
        "file_path": "/tmp/test.pdf",
        "num_chunks": 5,
        "status": "indexed",
    }
    defaults.update(overrides)
    return Document(**defaults)


# ============================================================================
#  POST /api/ingest/document
# ============================================================================

class TestIngestDocument:
    """POST /api/ingest/document — ingest by file path."""

    def test_returns_document_on_success(self, monkeypatch):
        doc = _sample_document()
        monkeypatch.setattr(
            "rag_agent.api.routes.ingest.ingest_document",
            lambda file_path, chunk_size=None, chunk_overlap=None: doc,
        )
        resp = client.post("/api/ingest/document", json={"file_path": "/tmp/test.pdf"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "abc-123"
        assert resp.json()["filename"] == "test.pdf"

    def test_passes_chunk_size_and_overlap(self, monkeypatch):
        captured = {}

        def _fake_ingest(file_path, chunk_size=None, chunk_overlap=None):
            captured.update(file_path=file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return _sample_document()

        monkeypatch.setattr("rag_agent.api.routes.ingest.ingest_document", _fake_ingest)

        client.post(
            "/api/ingest/document",
            json={"file_path": "/tmp/x.pdf", "chunk_size": 256, "chunk_overlap": 10},
        )

        assert captured["file_path"] == "/tmp/x.pdf"
        assert captured["chunk_size"] == 256
        assert captured["chunk_overlap"] == 10

    def test_file_not_found_returns_404(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("no such file")

        monkeypatch.setattr("rag_agent.api.routes.ingest.ingest_document", _raise)

        resp = client.post("/api/ingest/document", json={"file_path": "/tmp/missing.pdf"})
        assert resp.status_code == 404
        assert "no such file" in resp.json()["detail"]

    def test_generic_exception_returns_500(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("something broke")

        monkeypatch.setattr("rag_agent.api.routes.ingest.ingest_document", _raise)

        resp = client.post("/api/ingest/document", json={"file_path": "/tmp/x.pdf"})
        assert resp.status_code == 500
        assert "something broke" in resp.json()["detail"]


# ============================================================================
#  POST /api/ingest/upload
# ============================================================================

class TestIngestUpload:
    """POST /api/ingest/upload — file upload endpoint."""

    def test_upload_txt_file_succeeds(self, monkeypatch):
        doc = _sample_document(filename="notes.txt", file_type="txt")
        monkeypatch.setattr(
            "rag_agent.api.routes.ingest.ingest_document",
            lambda file_path, chunk_size=None, chunk_overlap=None: doc,
        )

        resp = client.post(
            "/api/ingest/upload",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )

        assert resp.status_code == 200
        assert resp.json()["filename"] == "notes.txt"

    def test_temp_file_is_cleaned_up(self, monkeypatch):
        """The temp file created during upload should be deleted after ingest."""
        import tempfile
        from pathlib import Path

        monkeypatch.setattr(
            "rag_agent.api.routes.ingest.ingest_document",
            lambda file_path, chunk_size=None, chunk_overlap=None: _sample_document(file_path=file_path),
        )

        resp = client.post(
            "/api/ingest/upload",
            files={"file": ("data.md", b"# Hello", "text/markdown")},
        )
        assert resp.status_code == 200
        temp_path = Path(resp.json()["file_path"])
        # Temp file must be gone after the request
        assert not temp_path.exists()

    def test_upload_without_file_returns_422(self):
        resp = client.post("/api/ingest/upload")
        assert resp.status_code == 422


# ============================================================================
#  POST /api/ingest/directory
# ============================================================================

class TestIngestDirectory:
    """POST /api/ingest/directory — batch ingest by directory path."""

    def test_returns_list_of_documents(self, monkeypatch):
        docs = [
            _sample_document(id="d1", filename="a.pdf"),
            _sample_document(id="d2", filename="b.pdf"),
        ]
        monkeypatch.setattr(
            "rag_agent.api.routes.ingest.ingest_directory",
            lambda directory, chunk_size=None, chunk_overlap=None: docs,
        )

        resp = client.post("/api/ingest/directory", json={"directory": "/tmp/docs"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "d1"
        assert data[1]["id"] == "d2"

    def test_not_a_directory_returns_404(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise NotADirectoryError("not a directory")

        monkeypatch.setattr("rag_agent.api.routes.ingest.ingest_directory", _raise)

        resp = client.post("/api/ingest/directory", json={"directory": "/tmp/notadir"})
        assert resp.status_code == 404

    def test_generic_exception_returns_500(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("rag_agent.api.routes.ingest.ingest_directory", _raise)

        resp = client.post("/api/ingest/directory", json={"directory": "/tmp/docs"})
        assert resp.status_code == 500
