"""OpenAI embedding provider."""

from rag_agent.embeddings.base import EmbeddingProvider
from rag_agent.utils.errors import EmbeddingModelError, MissingAPIKeyError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using the OpenAI API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """Initialize the OpenAI embedding provider.

        Args:
            api_key: OpenAI API key.
            model: OpenAI embedding model name.

        Raises:
            MissingAPIKeyError: If no API key is provided.
            EmbeddingModelError: If the client cannot be initialized.
        """
        if not api_key:
            raise MissingAPIKeyError("openai", "OPENAI_API_KEY")

        try:
            from openai import OpenAI

            self._model_name = model
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingModelError(model, str(e)) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [d.embedding for d in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self._model_name, 1536)
