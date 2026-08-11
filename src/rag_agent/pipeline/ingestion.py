from pathlib import Path
import shutil
import uuid

from rag_agent.config import settings
from rag_agent.document.chunker import chunk_text
from rag_agent.document.loader import SUPPORTED_EXTENSIONS, get_text_content, load_document
from rag_agent.embeddings.factory import get_embedding_provider
from rag_agent.storage.doc_store import DocStore
from rag_agent.storage.vector_store import VectorStore
from rag_agent.document.schemas import Document


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
