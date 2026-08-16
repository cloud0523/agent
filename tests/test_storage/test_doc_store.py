import re
from datetime import datetime, timedelta
import uuid

import pytest
import sqlite3

from rag_agent.document.schemas import Document
from rag_agent.storage.doc_store import DocStore


class TestDocStoreRegistration:
    def test_create_table_on_init(self, temp_db):
        cursor = temp_db._conn.execute("PRAGMA table_info('documents')")
        columns = {row['name'] for row in cursor.fetchall()}

        assert {
            'id',
            'filename',
            'file_type',
            'file_path',
            'num_chunks',
            'uploaded_at',
            'size_bytes',
            'status',
        } <= columns

    def test_register_document_returns_doc_with_id(self, temp_db, sample_document):
        document = temp_db.register_document(sample_document)

        assert document.id == sample_document.id
        assert document.filename == sample_document.filename

    def test_register_document_auto_generates_id(self, temp_db):
        document = Document(
            id='',
            filename='auto_id.txt',
            file_type='txt',
            file_path='/tmp/auto_id.txt',
        )

        registered = temp_db.register_document(document)

        assert isinstance(registered.id, str)
        assert len(registered.id) == 36
        assert registered.id.count('-') == 4
        assert re.fullmatch(r'[0-9a-fA-F-]{36}', registered.id)

    def test_register_duplicate_id_raises_error(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        with pytest.raises(sqlite3.IntegrityError):
            temp_db.register_document(sample_document)


class TestDocStoreQuery:
    def test_get_document_found(self, temp_db, sample_document):
        temp_db.register_document(sample_document)
        found = temp_db.get_document(sample_document.id)

        assert found is not None
        assert found.id == sample_document.id
        assert found.filename == sample_document.filename
        assert found.status == sample_document.status
        assert found.num_chunks == sample_document.num_chunks

    def test_get_document_not_found(self, temp_db):
        missing_id = str(uuid.uuid4())
        assert temp_db.get_document(missing_id) is None

    def test_list_documents_empty(self, temp_db):
        assert temp_db.list_documents() == []

    def test_list_documents_returns_newest_first(self, temp_db):
        older_doc = Document(
            id='old-doc',
            filename='old.txt',
            file_type='txt',
            file_path='/tmp/old.txt',
            uploaded_at=datetime.now() - timedelta(minutes=10),
        )
        newer_doc = Document(
            id='new-doc',
            filename='new.txt',
            file_type='txt',
            file_path='/tmp/new.txt',
            uploaded_at=datetime.now(),
        )

        temp_db.register_document(older_doc)
        temp_db.register_document(newer_doc)

        results = temp_db.list_documents()

        assert [doc.id for doc in results] == [newer_doc.id, older_doc.id]

    def test_update_status_changes_fields(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        temp_db.update_status(sample_document.id, 'indexed', num_chunks=12)
        updated = temp_db.get_document(sample_document.id)

        assert updated is not None
        assert updated.status == 'indexed'
        assert updated.num_chunks == 12

    def test_delete_document_removes_and_returns_true(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        assert temp_db.delete_document(sample_document.id) is True
        assert temp_db.get_document(sample_document.id) is None

    def test_delete_nonexistent_returns_false(self, temp_db):
        assert temp_db.delete_document(str(uuid.uuid4())) is False


class TestDocStoreUpdateDocument:
    def test_partial_update_status_only(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        assert temp_db.update_document(sample_document.id, status="error") is True

        updated = temp_db.get_document(sample_document.id)
        assert updated.status == "error"
        # 未传入的字段保持不变
        assert updated.filename == sample_document.filename
        assert updated.num_chunks == sample_document.num_chunks

    def test_full_update_via_kwargs(self, temp_db, sample_document):
        temp_db.register_document(sample_document)
        new_time = datetime.now()

        assert temp_db.update_document(
            sample_document.id,
            filename="v2.pdf",
            file_type="pdf",
            file_path="/tmp/v2.pdf",
            num_chunks=7,
            size_bytes=99,
            status="indexed",
            uploaded_at=new_time,
        ) is True

        updated = temp_db.get_document(sample_document.id)
        assert updated.filename == "v2.pdf"
        assert updated.file_path == "/tmp/v2.pdf"
        assert updated.num_chunks == 7
        assert updated.size_bytes == 99
        assert abs((updated.uploaded_at - new_time).total_seconds()) < 1

    def test_full_replace_via_document(self, temp_db, sample_document):
        temp_db.register_document(sample_document)
        replacement = Document(
            id=sample_document.id,
            filename="replaced.txt",
            file_type="txt",
            file_path="/tmp/replaced.txt",
            num_chunks=2,
            status="indexed",
        )

        assert temp_db.update_document(document=replacement) is True

        updated = temp_db.get_document(sample_document.id)
        assert updated.filename == "replaced.txt"
        assert updated.file_type == "txt"
        assert updated.num_chunks == 2

    def test_unknown_field_raises_value_error(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        with pytest.raises(ValueError):
            temp_db.update_document(sample_document.id, chunk_count=5)

    def test_invalid_status_raises(self, temp_db, sample_document):
        temp_db.register_document(sample_document)

        with pytest.raises(ValueError):
            temp_db.update_document(sample_document.id, status="bogus")

    def test_missing_doc_id_raises(self, temp_db):
        with pytest.raises(ValueError):
            temp_db.update_document()

    def test_no_fields_returns_false(self, temp_db, sample_document):
        temp_db.register_document(sample_document)
        assert temp_db.update_document(sample_document.id) is False

    def test_nonexistent_returns_false(self, temp_db):
        assert temp_db.update_document(str(uuid.uuid4()), status="indexed") is False


class TestDocStoreInit:
    """DocStore must create its parent directory (fixes fresh-install 500s)."""

    def test_creates_parent_directory(self, temp_dir):
        db_path = temp_dir / "nested" / "subdir" / "metadata.db"
        assert not db_path.parent.exists()  # pre-condition: dir is missing

        store = DocStore(str(db_path))
        try:
            assert db_path.parent.exists()  # mkdir happened
            assert db_path.exists()  # sqlite file actually created
        finally:
            store.close()

    def test_usable_after_parent_created(self, temp_dir):
        db_path = temp_dir / "data" / "metadata.db"
        store = DocStore(str(db_path))
        try:
            doc = Document(
                id="fresh-1",
                filename="a.txt",
                file_type="txt",
                file_path="/tmp/a.txt",
            )
            store.register_document(doc)
            assert [d.id for d in store.list_documents()] == ["fresh-1"]
        finally:
            store.close()

