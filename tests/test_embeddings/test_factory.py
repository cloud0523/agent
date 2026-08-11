from unittest.mock import patch

import pytest

from rag_agent.config import Settings
from rag_agent.embeddings.factory import get_embedding_provider
from rag_agent.embeddings.local import LocalEmbeddingProvider
from rag_agent.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_agent.utils.errors import ConfigurationError


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider factory function."""

    def test_openai_provider(self):
        """Returns an OpenAIEmbeddingProvider with the configured API key and model."""
        settings = Settings(
            embedding_provider="openai",
            openai_api_key="sk-test",
            openai_embedding_model="text-embedding-3-large",
        )
        with patch("openai.OpenAI"):
            provider = get_embedding_provider(settings)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider._model_name == "text-embedding-3-large"

    def test_openai_missing_api_key_raises_configuration_error(self):
        """openai provider with no API key raises ConfigurationError."""
        settings = Settings(
            embedding_provider="openai",
            openai_api_key=None,
            openai_embedding_model="text-embedding-3-large",
        )
        with pytest.raises(ConfigurationError):
            get_embedding_provider(settings)

    def test_unknown_provider_raises_configuration_error(self):
        """An unknown provider name raises ConfigurationError."""

        class UnknownSettings:
            embedding_provider = "unknown"
            embedding_model = "ignored"
            openai_api_key = None
            openai_embedding_model = "ignored"

        with pytest.raises(ConfigurationError):
            get_embedding_provider(UnknownSettings())
