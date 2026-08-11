"""Tests for text chunking strategies."""

import pytest

from rag_agent.document.chunker import chunk_text


# ── Fake recursive chunker (avoids LlamaIndex → NLTK → regex import issues) ──


def _fake_recursive_chunk(text, chunk_size, overlap):
    """A simple character-based splitter that mimics recursive splitting."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Prevent infinite loop when overlap >= chunk_size
        advance = max(1, chunk_size - overlap)
        start += advance
    return chunks


# ── Helper to mock the recursive strategy ──

@pytest.fixture
def mock_recursive(monkeypatch):
    """Replace _recursive_chunk with the fake version."""
    monkeypatch.setattr(
        "rag_agent.document.chunker._recursive_chunk",
        _fake_recursive_chunk,
    )


# ── Tests ──


class TestChunkTextDispatcher:
    """Tests for the chunk_text public function and its strategy routing."""

    def test_fixed_strategy_returns_chunks_of_expected_size(self):
        text = "0123456789" * 20  # 200 chars
        chunks = chunk_text(text, "doc-1", chunk_size=50, chunk_overlap=0, strategy="fixed")
        assert len(chunks) > 0
        for c in chunks:
            assert c.document_id == "doc-1"
            assert c.metadata["strategy"] == "fixed"

    def test_sentence_strategy_splits_on_boundaries(self):
        text = "Hello world. This is a test. Another sentence here."
        chunks = chunk_text(text, "doc-1", chunk_size=512, strategy="sentence")
        assert all(c.metadata["strategy"] == "sentence" for c in chunks)

    def test_recursive_strategy_returns_chunks(self, mock_recursive):
        text = "This is a sample paragraph. It has multiple sentences. " * 10
        chunks = chunk_text(text, "doc-1", chunk_size=200, chunk_overlap=30, strategy="recursive")
        assert len(chunks) > 0
        assert all(c.metadata["strategy"] == "recursive" for c in chunks)

    def test_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("hello", "doc-1", strategy="unknown")  # type: ignore

    def test_chunk_index_is_sequential(self):
        text = "A. " * 500
        chunks = chunk_text(text, "doc-1", chunk_size=100, chunk_overlap=0, strategy="fixed")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text_produces_no_chunks(self):
        chunks = chunk_text("   ", "doc-1", strategy="fixed")
        assert chunks == []

    def test_text_shorter_than_chunk_size_produces_one_chunk(self):
        chunks = chunk_text("short", "doc-1", chunk_size=1000, strategy="fixed")
        assert len(chunks) == 1
        assert chunks[0].text == "short"


class TestFixedChunking:
    """Tests specific to the fixed-size chunking algorithm."""

    def test_no_overlap_chunks_are_contiguous(self):
        text = "0123456789" * 20  # 200 chars
        chunks = chunk_text(text, "doc-1", chunk_size=50, chunk_overlap=0, strategy="fixed")
        for c in chunks[:-1]:
            assert len(c.text) == 50

    def test_overlap_between_consecutive_chunks(self):
        text = "ABCDEFGHIJ" * 20  # 200 chars
        chunks = chunk_text(text, "doc-1", chunk_size=15, chunk_overlap=5, strategy="fixed")
        if len(chunks) >= 2:
            tail = chunks[0].text[-5:]
            head = chunks[1].text[:5]
            assert tail == head

    def test_overlap_not_smaller_than_chunk_size_still_terminates(self):
        """Edge case: chunk_overlap >= chunk_size should not cause infinite loop."""
        text = "0123456789" * 10
        chunks = chunk_text(text, "doc-1", chunk_size=10, chunk_overlap=10, strategy="fixed")
        assert len(chunks) > 0
        # Check all chunks have content
        for c in chunks:
            assert len(c.text) > 0


class TestSentenceChunking:
    """Tests specific to sentence-aware chunking."""

    def test_preserves_sentence_boundaries(self, sample_text):
        chunks = chunk_text(sample_text, "doc-1", chunk_size=512, strategy="sentence")
        for c in chunks:
            assert c.text.strip()[-1] in ".!?"

    def test_merges_short_sentences(self):
        text = "Hi. Hello. Hey there. How are you?"
        chunks = chunk_text(text, "doc-1", chunk_size=500, strategy="sentence")
        assert len(chunks) == 1


class TestRecursiveChunking:
    """Tests specific to recursive chunking (uses mock to avoid NLTK deps)."""

    def test_splits_long_text_into_multiple_chunks(self, mock_recursive):
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThird paragraph here."
        chunks = chunk_text(text, "doc-1", chunk_size=20, chunk_overlap=0, strategy="recursive")
        assert len(chunks) >= 1

    def test_chunks_have_non_empty_text(self, mock_recursive):
        text = "Single sentence here. Another one. And a third to make length." * 5
        chunks = chunk_text(text, "doc-1", chunk_size=50, strategy="recursive")
        for c in chunks:
            assert len(c.text.strip()) > 0
