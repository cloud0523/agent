"""Tests for the RAG Agent custom exception hierarchy."""

from rag_agent.utils.errors import (
    ConfigurationError,
    ContextOverflowError,
    DocumentError,
    DocumentLoadError,
    DocumentNotFoundError,
    EmbeddingError,
    EmbeddingModelError,
    LLMConnectionError,
    MissingAPIKeyError,
    RAGAgentError,
    StorageError,
    UnsupportedFileTypeError,
    VectorStoreError,
    GenerationError,
)


def test_rag_agent_error_basics():
    err = RAGAgentError("base failure")
    assert isinstance(err, Exception)
    assert str(err) == "base failure"


def test_document_error_basics():
    err = DocumentError("document failure")
    assert isinstance(err, RAGAgentError)
    assert str(err) == "document failure"


def test_unsupported_file_type_error():
    err = UnsupportedFileTypeError("/tmp/file.xyz", "xyz")
    assert isinstance(err, DocumentError)
    assert isinstance(err, RAGAgentError)
    assert err.file_path == "/tmp/file.xyz"
    assert err.extension == "xyz"
    assert "xyz" in str(err)
    assert "Unsupported file type" in str(err)


def test_document_load_error():
    err = DocumentLoadError("/tmp/doc.txt", "parse failure")
    assert isinstance(err, DocumentError)
    assert isinstance(err, RAGAgentError)
    assert err.file_path == "/tmp/doc.txt"
    assert err.reason == "parse failure"
    assert "Failed to load document" in str(err)
    assert "parse failure" in str(err)


def test_document_not_found_error():
    err = DocumentNotFoundError("doc-123")
    assert isinstance(err, DocumentError)
    assert err.doc_id == "doc-123"
    assert "Document not found" in str(err)


def test_embedding_error_basics():
    err = EmbeddingError("embed failure")
    assert isinstance(err, RAGAgentError)
    assert str(err) == "embed failure"


def test_embedding_model_error():
    err = EmbeddingModelError("test-model", "timeout")
    assert isinstance(err, EmbeddingError)
    assert err.model_name == "test-model"
    assert err.reason == "timeout"
    assert "Embedding model 'test-model' error" in str(err)


def test_storage_error_basics():
    err = StorageError("storage failure")
    assert isinstance(err, RAGAgentError)
    assert str(err) == "storage failure"


def test_vector_store_error():
    err = VectorStoreError("upsert", "connection lost")
    assert isinstance(err, StorageError)
    assert err.operation == "upsert"
    assert err.reason == "connection lost"
    assert "Vector store 'upsert' failed" in str(err)


def test_generation_error_basics():
    err = GenerationError("generation failure")
    assert isinstance(err, RAGAgentError)
    assert str(err) == "generation failure"


def test_llm_connection_error():
    err = LLMConnectionError("openai", "timeout")
    assert isinstance(err, GenerationError)
    assert err.provider == "openai"
    assert err.reason == "timeout"
    assert "LLM provider 'openai' connection failed" in str(err)


def test_context_overflow_error():
    err = ContextOverflowError(5000, 4000)
    assert isinstance(err, GenerationError)
    assert err.tokens == 5000
    assert err.limit == 4000
    assert "Context too large" in str(err)
    assert "5000" in str(err)
    assert "4000" in str(err)


def test_configuration_error_basics():
    err = ConfigurationError("bad config")
    assert isinstance(err, RAGAgentError)
    assert str(err) == "bad config"


def test_missing_api_key_error():
    err = MissingAPIKeyError("openai", "OPENAI_API_KEY")
    assert isinstance(err, ConfigurationError)
    assert err.provider == "openai"
    assert err.key_name == "OPENAI_API_KEY"
    assert "OPENAI_API_KEY" in str(err)
    assert "provider 'openai'" in str(err)
