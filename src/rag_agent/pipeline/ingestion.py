from pathlib import Path
import shutil
import uuid
from datetime import datetime

from rag_agent.config import settings
from rag_agent.document.chunker import chunk_text
from rag_agent.document.loader import SUPPORTED_EXTENSIONS, get_text_content, load_document
from rag_agent.embeddings.factory import get_embedding_provider
from rag_agent.storage.doc_store import DocStore
from rag_agent.storage.vector_store import VectorStore
from rag_agent.document.schemas import Document
from rag_agent.utils.errors import DocumentNotFoundError


def ingest_document(file_path, chunk_size=None, chunk_overlap=None):
    """Ingest a single document into the RAG knowledge base.

    Steps:
        1. Initialize embedding provider
        2. Initialize metadata and vector stores
        3. Load document metadata
        4. Register document in DocStore (status="processing")
        5. Chunk the text
        6. Generate embeddings
        7. Save chunks to VectorStore
        8. Update DocStore (status="indexed", num_chunks)
        9. Return the registered Document
    """
    settings.ensure_directories()

    source_path = Path(file_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Document not found: {source_path}")

    dest_dir = settings.data_dir / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source_path.name
    if dest_path.exists():
        dest_path = dest_dir / f"{uuid.uuid4().hex}_{source_path.name}"

    shutil.copy2(source_path, dest_path)

    document = load_document(dest_path)
    document.file_path = str(dest_path)
    document.status = "processing"

    store = DocStore(str(settings.doc_store_path))
    document = store.register_document(document)

    try:
        text = get_text_content(dest_path)
        chunks = chunk_text(
            text,
            document_id=document.id,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            strategy=settings.chunking_strategy,
        )

        chunk_texts = [chunk.text for chunk in chunks]
        embedding_provider = get_embedding_provider(settings)
        embeddings = embedding_provider.embed(chunk_texts)

        vector_store = VectorStore(str(settings.chroma_persist_dir))
        vector_store.add_chunks(document.id, chunk_texts, embeddings)

        store.update_status(document.id, "indexed", num_chunks=len(chunks))
        document.num_chunks = len(chunks)
        document.status = "indexed"
        return document

    except Exception:
        store.update_status(document.id, "error", num_chunks=0)
        raise


def ingest_directory(directory, chunk_size=None, chunk_overlap=None):
    """Batch ingest every supported document in a directory."""
    settings.ensure_directories()
    results: list[Document] = []

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    for file_path in sorted(dir_path.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            document = ingest_document(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:
            document = Document(
                id=str(uuid.uuid4()),
                filename=file_path.name,
                file_type=file_path.suffix.lstrip("."),
                file_path=str(file_path),
                num_chunks=0,
                status="error",
            )
        results.append(document)
        continue

    return results


def reingest_document(document_id: str, file_path: str, chunk_size=None, chunk_overlap=None):
    """Reingest an existing document ID with a new source file.

    Behavior decisions:
      1) Refresh `uploaded_at` to `datetime.now()` at the final swap.
      2) Best-effort delete of the old file if its path differs from the new one.
      3) Keep the old vectors available while building embeddings; only delete/add during the final swap.
    """
    settings.ensure_directories()

    source_path = Path(file_path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Document not found: {source_path}")

    dest_dir = settings.data_dir / "documents"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / source_path.name
    if dest_path.exists():
        dest_path = dest_dir / f"{uuid.uuid4().hex}_{source_path.name}"

    store = DocStore(str(settings.doc_store_path))
    existing = store.get_document(document_id)
    if existing is None:
        raise DocumentNotFoundError(document_id)

    # Copy new file after verifying the document exists (dedup behavior mirrors ingest_document)
    shutil.copy2(source_path, dest_path)

    # Build new chunks and embeddings using the copied file, without changing the store yet.
    document = load_document(dest_path)
    document.file_path = str(dest_path)

    text = get_text_content(dest_path)
    chunks = chunk_text(
        text,
        document_id=document_id,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        strategy=settings.chunking_strategy,
    )

    chunk_texts = [chunk.text for chunk in chunks]
    embedding_provider = get_embedding_provider(settings)
    embeddings = embedding_provider.embed(chunk_texts)

    vector_store = VectorStore(str(settings.chroma_persist_dir))

    # Now perform the fast swap: delete old vectors, add new vectors, then update metadata atomically.
    try:
        vector_store.delete_document(document_id)
        vector_store.add_chunks(document_id, chunk_texts, embeddings)

        # Refresh uploaded_at and update metadata (use fields from the loaded document)
        store.update_document(
            document_id,
            filename=document.filename,
            file_type=document.file_type,
            file_path=str(dest_path),
            size_bytes=document.size_bytes,
            num_chunks=len(chunks),
            uploaded_at=datetime.now(),
            status="indexed",
        )

    except Exception:
        # If anything goes wrong during swap, mark as error so the doc is discoverable as failed.
        store.update_document(document_id, status="error")
        raise

    # Best-effort: remove old file if it's no longer referenced
    try:
        if existing.file_path and existing.file_path != str(dest_path):
            Path(existing.file_path).unlink(missing_ok=True)
    except Exception:
        # ignore filesystem cleanup errors
        pass

    return store.get_document(document_id)
