"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings.

    All embedding providers must implement this interface so they can
    be swapped transparently via configuration.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            A list of embedding vectors, each being a list of floats.
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: The query string to embed.

        Returns:
            An embedding vector as a list of floats.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors produced."""
        ...
