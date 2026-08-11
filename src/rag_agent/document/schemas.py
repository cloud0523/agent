"""Pydantic models for documents and chunks."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an ingested document."""

    id: str = Field(..., description="Unique document ID (UUID)")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension: pdf, docx, txt, md")
    file_path: str = Field(..., description="Path to the stored copy of the file")
    num_chunks: int = Field(default=0, description="Number of chunks this document was split into")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    size_bytes: int = Field(default=0, description="File size in bytes")
    status: Literal["processing", "indexed", "error"] = Field(
        default="processing", description="Current processing status"
    )


class Chunk(BaseModel):
    """Represents a single text chunk from a document."""

    id: str = Field(..., description="Unique chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk within the document")
    metadata: dict = Field(
        default_factory=dict,
        description="Extra metadata: page_number, section_title, etc.",
    )


class ChunkWithScore(BaseModel):
    """A chunk returned from a vector search, with its similarity score."""

    chunk: Chunk
    score: float = Field(..., description="Similarity score (0-1)")
    document_filename: Optional[str] = Field(
        default=None, description="Source document filename"
    )
