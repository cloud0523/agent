"""Tests for reingest_document in the ingestion pipeline."""

import pytest

from rag_agent.document.schemas import Chunk, Document
from rag_agent.pipeline.ingestion import reingest_document
from rag_agent.utils.errors import DocumentNotFoundError


def _doc(**overrides) -> Document:
    defaults = {
        "id": "doc-1",
        "filename": "old.txt",
        "file_type": "txt",
        "file_path": "/data/documents/old.txt",
        "num_chunks": 3,
        "size_bytes": 10,
        "status": "indexed",
    }
    defaults.update(overrides)
    return Document(**defaults)


class DummyEmbed:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


def _patch_settings(monkeypatch, tmp_path):
    """Redirect data/chroma dirs to tmp_path so no real files are created."""
    import rag_agent.pipeline.ingestion as ing

    monkeypatch.setattr(ing.settings, "data_dir", tmp_path)
    monkeypatch.setattr(ing.settings, "chroma_persist_dir", tmp_path / "chroma")


def test_reingest_swaps_vectors_then_updates_metadata(monkeypatch, tmp_path):
    src = tmp_path / "new.md"
    src.write_text("# new content")

    old = _doc()
    updated = _doc(
        filename="new.md",
        file_type="md",
        file_path=str(tmp_path / "documents" / "new.md"),
        num_chunks=1,
        size_bytes=11,
    )

    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_document.side_effect = [old, updated]
    monkeypatch.setattr("rag_agent.pipeline.ingestion.DocStore", lambda path: store)

    vs = MagicMock()
    order = []
    vs.delete_document.side_effect = lambda doc_id: order.append("delete")
    vs.add_chunks.side_effect = lambda *a, **kw: order.append("add")
    monkeypatch.setattr("rag_agent.pipeline.ingestion.VectorStore", lambda persist_dir: vs)

    loaded = _doc(id="", filename="new.md", file_type="md", file_path=str(src), size_bytes=11)
    monkeypatch.setattr("rag_agent.pipeline.ingestion.load_document", lambda path: loaded)
    monkeypatch.setattr("rag_agent.pipeline.ingestion.get_text_content", lambda path: "hello world new content")
    monkeypatch.setattr(
        "rag_agent.pipeline.ingestion.chunk_text",
        lambda text, document_id, chunk_size=None, chunk_overlap=None, strategy=None: [
            Chunk(id=f"{document_id}_0", document_id=document_id, text="hello", chunk_index=0)
        ],
    )
    monkeypatch.setattr("rag_agent.pipeline.ingestion.get_embedding_provider", lambda s: DummyEmbed())

    _patch_settings(monkeypatch, tmp_path)

    result = reingest_document("doc-1", str(src))

    # 1. 安全交换顺序：先 delete 旧向量，再 add 新向量
    assert order == ["delete", "add"]
    vs.delete_document.assert_called_once_with("doc-1")
    vs.add_chunks.assert_called_once()

    # 2. 元数据必须包含新文件的 filename/type/size（Finding 1 的关键断言）
    kwargs = store.update_document.call_args.kwargs
    assert kwargs["filename"] == "new.md"
    assert kwargs["file_type"] == "md"
    assert kwargs["size_bytes"] == 11
    assert kwargs["num_chunks"] == 1
    assert kwargs["status"] == "indexed"

    # 3. 返回更新后的文档
    assert result == updated


def test_reingest_raises_when_document_missing(monkeypatch, tmp_path):
    src = tmp_path / "new.md"
    src.write_text("x")

    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_document.return_value = None
    monkeypatch.setattr("rag_agent.pipeline.ingestion.DocStore", lambda path: store)
    _patch_settings(monkeypatch, tmp_path)

    with pytest.raises(DocumentNotFoundError):
        reingest_document("ghost", str(src))


def test_reingest_raises_when_source_missing(monkeypatch, tmp_path):
    missing = tmp_path / "nope.txt"
    _patch_settings(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError):
        reingest_document("doc-1", str(missing))
