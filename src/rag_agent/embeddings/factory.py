"""Factory for creating embedding providers based on configuration."""

from rag_agent.config import Settings
from rag_agent.embeddings.base import EmbeddingProvider
from rag_agent.embeddings.local import LocalEmbeddingProvider
from rag_agent.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_agent.utils.errors import ConfigurationError


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create an embedding provider instance from configuration.

    Args:
        settings: Application settings.

    Returns:
        An EmbeddingProvider instance.

    Raises:
        ConfigurationError: If the configured provider is unknown.
    """
    provider_name = settings.embedding_provider

    if provider_name == "local":
        return LocalEmbeddingProvider(model_name=settings.embedding_model)

    elif provider_name == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    else:
        raise ConfigurationError(f"Unknown embedding provider: {provider_name}")
