from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from rag_agent.config import settings
from rag_agent.document.schemas import Document
from rag_agent.storage.doc_store import DocStore
from rag_agent.storage.vector_store import VectorStore


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
