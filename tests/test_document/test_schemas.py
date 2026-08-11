"""Tests for document schemas: Document, Chunk, ChunkWithScore."""

from datetime import datetime

from rag_agent.document.schemas import Chunk, ChunkWithScore, Document


class TestDocument:
    def test_create_with_required_fields(self):
        doc = Document(
            id="abc-123",
            filename="report.pdf",
            file_type="pdf",
            file_path="/data/report.pdf",
        )
        assert doc.id == "abc-123"
        assert doc.filename == "report.pdf"
        assert doc.file_type == "pdf"
        assert doc.file_path == "/data/report.pdf"

    def test_default_values(self):
        doc = Document(
            id="abc-123",
            filename="notes.txt",
            file_type="txt",
            file_path="/data/notes.txt",
        )
        assert doc.num_chunks == 0
        assert doc.size_bytes == 0
        assert doc.status == "processing"
        assert isinstance(doc.uploaded_at, datetime)

    def test_status_rejects_invalid_value(self):
        import pytest as pt

        with pt.raises(ValueError):  # Pydantic validation
            Document(
                id="abc",
                filename="x.txt",
                file_type="txt",
                file_path="/x.txt",
                status="invalid",  # type: ignore
            )

    def test_model_dump_includes_all_fields(self):
        doc = Document(
            id="abc",
            filename="x.txt",
            file_type="txt",
            file_path="/x.txt",
        )
        data = doc.model_dump()
        assert "id" in data
        assert "filename" in data
        assert "file_type" in data
        assert "status" in data
        assert "uploaded_at" in data

    def test_two_docs_with_same_fields_are_equal(self):
        now = datetime(2024, 1, 1, 12, 0, 0)
        d1 = Document(
            id="1", filename="a.txt", file_type="txt",
            file_path="/a.txt", uploaded_at=now,
        )
        d2 = Document(
            id="1", filename="a.txt", file_type="txt",
            file_path="/a.txt", uploaded_at=now,
        )
        assert d1 == d2


class TestChunk:
    def test_create_with_required_fields(self):
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            text="Hello world.",
            chunk_index=0,
        )
        assert chunk.id == "chunk-001"
        assert chunk.document_id == "doc-001"
        assert chunk.text == "Hello world."
        assert chunk.chunk_index == 0

    def test_default_metadata_is_empty_dict(self):
        chunk = Chunk(
            id="c1",
            document_id="d1",
            text="text",
            chunk_index=0,
        )
        assert chunk.metadata == {}

    def test_metadata_preserves_custom_keys(self):
        chunk = Chunk(
            id="c1",
            document_id="d1",
            text="text",
            chunk_index=3,
            metadata={"strategy": "recursive", "page": 2},
        )
        assert chunk.metadata["strategy"] == "recursive"
        assert chunk.metadata["page"] == 2


class TestChunkWithScore:
    def test_creates_with_chunk_and_score(self):
        chunk = Chunk(
            id="c1", document_id="d1", text="hello",
            chunk_index=0,
        )
        cws = ChunkWithScore(chunk=chunk, score=0.85)
        assert cws.score == 0.85
        assert cws.chunk == chunk

    def test_document_filename_is_optional(self):
        chunk = Chunk(
            id="c1", document_id="d1", text="hello",
            chunk_index=0,
        )
        cws = ChunkWithScore(chunk=chunk, score=0.5)
        assert cws.document_filename is None

    def test_document_filename_can_be_set(self):
        chunk = Chunk(
            id="c1", document_id="d1", text="hello",
            chunk_index=0,
        )
        cws = ChunkWithScore(chunk=chunk, score=0.5, document_filename="report.pdf")
        assert cws.document_filename == "report.pdf"
