"""Text chunking strategies for document splitting."""

from typing import Literal

from rag_agent.document.schemas import Chunk


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: Literal["fixed", "sentence", "recursive"] = "recursive",
) -> list[Chunk]:
    """Split text into overlapping chunks using the specified strategy.

    Args:
        text: The full text content to split.
        document_id: The parent document ID.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        strategy: Chunking method to use.

    Returns:
        A list of Chunk objects.
    """
    if strategy == "fixed":
        chunks_text = _fixed_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        chunks_text = _sentence_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        chunks_text = _recursive_chunk(text, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [
        Chunk(
            id="",  # Assigned by pipeline
            document_id=document_id,
            text=chunk_text.strip(),
            chunk_index=i,
            metadata={"strategy": strategy},
        )
        for i, chunk_text in enumerate(chunks_text)
        if chunk_text.strip()
    ]


def _fixed_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size character chunking with overlap."""
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)  # Prevent infinite loop when overlap >= chunk_size
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def _sentence_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sentence-aware chunking: split on sentence boundaries, merge up to chunk_size."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Keep overlap: retain last few chars from previous chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _recursive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive splitting using LlamaIndex SentenceSplitter.

    Tries separators in order: paragraph, newline, sentence, word, character.
    """
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    doc = LlamaDocument(text=text)
    nodes = splitter.get_nodes_from_documents([doc])

    return [node.text for node in nodes if node.text]
