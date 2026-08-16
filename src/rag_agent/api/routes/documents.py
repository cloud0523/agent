from typing import List, Optional

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_agent.config import settings
from rag_agent.document.schemas import Document
from rag_agent.storage.doc_store import DocStore
from rag_agent.storage.vector_store import VectorStore
from rag_agent.pipeline.ingestion import reingest_document
from rag_agent.utils.errors import DocumentNotFoundError


router = APIRouter()


@router.get("/documents", response_model=List[Document])
def list_documents():
    store = DocStore(str(settings.doc_store_path))
    return store.list_documents()


@router.get("/documents/{document_id}", response_model=Document)
def get_document(document_id: str):
    store = DocStore(str(settings.doc_store_path))
    document = store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete a document and all its vectors from the knowledge base."""
    store = DocStore(str(settings.doc_store_path))
    document = store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    vs = VectorStore(str(settings.chroma_persist_dir))
    vs.delete_document(document_id)
    store.delete_document(document_id)

    return JSONResponse(status_code=204, content=None)


class ReingestDocumentRequest(BaseModel):
    file_path: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@router.post("/documents/{document_id}/reingest", response_model=Document)
def reingest_document_endpoint(document_id: str, req: ReingestDocumentRequest):
    try:
        doc = reingest_document(
            document_id,
            req.file_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        return doc
    except DocumentNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source file not found")
    except Exception as exc:  # pragma: no cover - generic mapping
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/documents/{document_id}/reingest/upload", response_model=Document)
def reingest_document_upload(document_id: str, file: UploadFile = File(...)):
    # write upload into a temporary directory and call pipeline synchronously
    content = file.file.read()
    safe_name = Path(file.filename).name
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / safe_name
        tmp_path.write_bytes(content)
        try:
            return reingest_document(document_id, str(tmp_path))
        except DocumentNotFoundError:
            raise HTTPException(status_code=404, detail="Document not found")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Source file not found")
        except Exception as exc:  # pragma: no cover - generic mapping
            raise HTTPException(status_code=500, detail=str(exc))
