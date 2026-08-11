"""Custom exception hierarchy for the RAG Agent."""


class RAGAgentError(Exception):
    """Base exception for all RAG Agent errors."""


# --- Document Errors ---

class DocumentError(RAGAgentError):
    """Base exception for document processing errors."""


class UnsupportedFileTypeError(DocumentError):
    """Raised when a file type is not supported."""

    def __init__(self, file_path: str, extension: str):
        self.file_path = file_path
        self.extension = extension
        super().__init__(
            f"Unsupported file type '{extension}' for file: {file_path}. "
            f"Supported types: PDF, DOCX, TXT, MD"
        )


class DocumentLoadError(DocumentError):
    """Raised when a document cannot be loaded/parsed."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to load document '{file_path}': {reason}")


class DocumentNotFoundError(DocumentError):
    """Raised when a document ID is not found in the store."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


# --- Embedding Errors ---

class EmbeddingError(RAGAgentError):
    """Base exception for embedding-related errors."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the embedding model fails to load or encode."""

    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Embedding model '{model_name}' error: {reason}")


# --- Storage Errors ---

class StorageError(RAGAgentError):
    """Base exception for storage-related errors."""


class VectorStoreError(StorageError):
    """Raised when a vector store operation fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store '{operation}' failed: {reason}")


# --- Generation Errors ---

class GenerationError(RAGAgentError):
    """Base exception for LLM generation errors."""


class LLMConnectionError(GenerationError):
    """Raised when connection to the LLM provider fails."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLM provider '{provider}' connection failed: {reason}")


class ContextOverflowError(GenerationError):
    """Raised when the prompt context exceeds the model's limit."""

    def __init__(self, tokens: int, limit: int):
        self.tokens = tokens
        self.limit = limit
        super().__init__(f"Context too large: {tokens} tokens exceeds limit of {limit}")


# --- Configuration Errors ---

class ConfigurationError(RAGAgentError):
    """Raised when configuration is invalid."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is not set."""

    def __init__(self, provider: str, key_name: str):
        self.provider = provider
        self.key_name = key_name
        super().__init__(
            f"API key '{key_name}' is required for provider '{provider}'. "
            f"Set it in .env or as an environment variable."
        )
