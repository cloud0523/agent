"""Local embedding provider using sentence-transformers."""

from rag_agent.embeddings.base import EmbeddingProvider
from rag_agent.utils.errors import EmbeddingModelError


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using a local sentence-transformers model.

    No API key required. Runs entirely on the local machine.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the local embedding provider.

        Args:
            model_name: The sentence-transformers model to use.

        Raises:
            EmbeddingModelError: If the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer

            self._model_name = model_name
            self._model: SentenceTransformer = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingModelError(model_name, str(e)) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text."""
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._model.get_sentence_embedding_dimension()
