"""Unified document loader supporting PDF, DOCX, TXT, and Markdown.

Uses LlamaIndex SimpleDirectoryReader under the hood, with custom
fallbacks for unsupported formats.
"""

from pathlib import Path
from typing import Optional

from rag_agent.document.schemas import Document
from rag_agent.utils.errors import DocumentLoadError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def load_document(file_path: Path) -> Document:
    """Load a single document and extract its text content.

    Args:
        file_path: Path to the document file.

    Returns:
        A Document object with extracted text and metadata.

    Raises:
        UnsupportedFileTypeError: If the file extension is not supported.
        DocumentLoadError: If the document cannot be parsed.
    """
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(str(file_path), ext)

    try:
        from llama_index.core import SimpleDirectoryReader

        # LlamaIndex SimpleDirectoryReader auto-detects file type
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        docs = reader.load_data()

        if not docs:
            raise DocumentLoadError(str(file_path), "No content extracted")

        # Extract full text from all pages
        full_text = "\n\n".join(doc.text for doc in docs if doc.text)

        return Document(
            id="",  # Assigned by the pipeline
            filename=file_path.name,
            file_type=ext.lstrip("."),
            file_path=str(file_path),
            size_bytes=file_path.stat().st_size,
        )

    except DocumentLoadError:
        raise
    except Exception as e:
        raise DocumentLoadError(str(file_path), str(e)) from e


def get_text_content(file_path: Path) -> str:
    """Extract raw text content from a document.

    Args:
        file_path: Path to the document file.

    Returns:
        The full text content as a string.
    """
    try:
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        docs = reader.load_data()
        return "\n\n".join(doc.text for doc in docs if doc.text)

    except Exception as e:
        raise DocumentLoadError(str(file_path), str(e)) from e
