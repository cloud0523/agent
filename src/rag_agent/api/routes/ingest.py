import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag_agent.document.schemas import Document
from rag_agent.pipeline.ingestion import ingest_directory, ingest_document


class IngestDocumentRequest(BaseModel):
    file_path: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class IngestDirectoryRequest(BaseModel):
    directory: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


router = APIRouter()


@router.post("/ingest/document", response_model=Document)
def ingest_document_endpoint(request: IngestDocumentRequest):
    try:
        return ingest_document(
            request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest/upload", response_model=Document)
async def ingest_upload_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return ingest_document(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/ingest/directory", response_model=List[Document])
def ingest_directory_endpoint(request: IngestDirectoryRequest):
    try:
        return ingest_directory(
            request.directory,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except NotADirectoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
