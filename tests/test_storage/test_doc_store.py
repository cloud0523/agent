import re
from datetime import datetime, timedelta
import uuid

import pytest
import sqlite3

from rag_agent.document.schemas import Document


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

