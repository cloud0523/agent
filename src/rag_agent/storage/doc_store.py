"""SQLite-based document metadata store.

Tracks what documents have been ingested, their status, and chunk counts.
Uses Python's built-in sqlite3 — no extra dependencies needed.
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from rag_agent.document.schemas import Document


class DocStore:
    """Manages document metadata in a local SQLite database.

    This is separate from the vector store (ChromaDB) so we can
    answer "what documents do I have?" without touching vectors.

    Usage:
        store = DocStore("data/metadata.db")
        store.register_document(doc)
        docs = store.list_documents()
        store.delete_document(doc_id)
    """

    def __init__(self, db_path: str):
        """Initialize the document store.

        Args:
            db_path: Path to the SQLite database file.
                     Created automatically if it doesn't exist.
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_table()

    def _create_table(self) -> None:
        """Create the documents table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                num_chunks INTEGER DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                status TEXT DEFAULT 'processing'
            )
        """)
        self._conn.commit()

    def register_document(self, document: Document) -> Document:
        """Insert a new document record. Assigns an ID if empty.

        Args:
            document: The Document to register.

        Returns:
            The Document with ID assigned.
        """
        if not document.id:
            document.id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT INTO documents (id, filename, file_type, file_path,
                                   num_chunks, uploaded_at, size_bytes, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.id,
                document.filename,
                document.file_type,
                document.file_path,
                document.num_chunks,
                document.uploaded_at.isoformat(),
                document.size_bytes,
                document.status,
            ),
        )
        self._conn.commit()
        return document

    def update_status(self, doc_id: str, status: str, num_chunks: int = 0) -> None:
        """Update a document's processing status and chunk count.

        Args:
            doc_id: The document ID.
            status: New status: 'processing', 'indexed', or 'error'.
            num_chunks: Number of chunks created (set when status='indexed').
        """
        self._conn.execute(
            "UPDATE documents SET status = ?, num_chunks = ? WHERE id = ?",
            (status, num_chunks, doc_id),
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by ID.

        Args:
            doc_id: The document ID.

        Returns:
            A Document object, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()

        if row is None:
            return None

        return Document(
            id=row["id"],
            filename=row["filename"],
            file_type=row["file_type"],
            file_path=row["file_path"],
            num_chunks=row["num_chunks"],
            uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
            size_bytes=row["size_bytes"],
            status=row["status"],
        )

    def list_documents(self) -> list[Document]:
        """List all documents, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()

        return [
            Document(
                id=row["id"],
                filename=row["filename"],
                file_type=row["file_type"],
                file_path=row["file_path"],
                num_chunks=row["num_chunks"],
                uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
                size_bytes=row["size_bytes"],
                status=row["status"],
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document record.

        Args:
            doc_id: The document ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        cursor = self._conn.execute(
            "DELETE FROM documents WHERE id = ?", (doc_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # Allowed columns that can be updated via update_document
    _ALLOWED_FIELDS = {
        "filename",
        "file_type",
        "file_path",
        "num_chunks",
        "uploaded_at",
        "size_bytes",
        "status",
    }

    def update_document(self, doc_id: str | None = None, document: Optional[Document] = None, **fields) -> bool:
        """Update document fields in the metadata store.

        Use either `document` (full replace) or `doc_id` + kwargs for partial update.

        Args:
            doc_id: ID of the document to update. Required if `document` not provided.
            document: Optional `Document` instance to update from (preferred for full updates).
            **fields: Column-value pairs to update. Keys must be in `_ALLOWED_FIELDS`.

        Returns:
            True if a row was updated, False otherwise.

        Raises:
            ValueError: If no updatable fields are provided, if unknown field keys are passed,
                        or if required parameters are missing.
        """
        # If a Document instance is provided, extract its fields (full-replace style)
        if document is not None:
            doc_id = document.id or doc_id
            fields = {
                "filename": document.filename,
                "file_type": document.file_type,
                "file_path": document.file_path,
                "num_chunks": document.num_chunks,
                "uploaded_at": document.uploaded_at,
                "size_bytes": document.size_bytes,
                "status": document.status,
            }

        if not doc_id:
            raise ValueError("doc_id is required")

        # No-op guard: nothing to update
        if not fields:
            return False

        # Validate keys against whitelist and build SET clause
        updates = []
        params = []
        for k, v in fields.items():
            if k not in self._ALLOWED_FIELDS:
                raise ValueError(f"unknown updatable field: {k}")

            # Skip explicit None values to avoid unintentionally setting NULL.
            if v is None:
                continue

            # Normalize types
            if k == "uploaded_at":
                v = v.isoformat() if isinstance(v, datetime) else v
            if k in {"num_chunks", "size_bytes"}:
                v = int(v)
            if k == "status":
                if v not in {"processing", "indexed", "error"}:
                    raise ValueError(f"invalid status value: {v}")

            updates.append(f"{k} = ?")
            params.append(v)

        if not updates:
            # e.g. all provided fields were None and therefore skipped
            return False

        sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"
        params.append(doc_id)
        cursor = self._conn.execute(sql, tuple(params))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
