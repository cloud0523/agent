from unittest.mock import MagicMock, patch

import pytest

from rag_agent.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_agent.utils.errors import MissingAPIKeyError


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider using a mocked OpenAI client."""

    @pytest.fixture
    def fake_client(self):
        """A mock OpenAI client."""
        client = MagicMock()

        def fake_create(model, input):
            response = MagicMock()
            response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in input]
            return response

        client.embeddings.create.side_effect = fake_create
        return client

    @pytest.fixture
    def provider(self, fake_client):
        """Create an OpenAIEmbeddingProvider with mocked client."""
        with patch(
            "openai.OpenAI",
            return_value=fake_client,
        ):
            return OpenAIEmbeddingProvider(api_key="sk-test")

    def test_missing_api_key_raises_missing_api_key_error(self):
        with pytest.raises(MissingAPIKeyError) as exc_info:
            OpenAIEmbeddingProvider(api_key="")

        assert exc_info.value.provider == "openai"
        assert exc_info.value.key_name == "OPENAI_API_KEY"

    def test_default_model_name(self, provider):
        assert provider._model_name == "text-embedding-3-small"

    def test_custom_model_name(self, fake_client):
        with patch("openai.OpenAI", return_value=fake_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-large")

        assert provider._model_name == "text-embedding-3-large"

    def test_embed_returns_two_embeddings(self, provider):
        result = provider.embed(["first", "second"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

    def test_embed_calls_api_correctly(self, provider, fake_client):
        texts = ["first", "second"]
        provider.embed(texts)
        fake_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts,
        )

    def test_embed_query_delegates_to_embed(self, provider, fake_client):
        result = provider.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]
        fake_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello"],
        )

    def test_dimension_text_embedding_3_small(self, provider):
        provider._model_name = "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_dimension_text_embedding_3_large(self, provider):
        provider._model_name = "text-embedding-3-large"
        assert provider.dimension == 3072

    def test_dimension_text_embedding_ada_002(self, provider):
        provider._model_name = "text-embedding-ada-002"
        assert provider.dimension == 1536

    def test_dimension_unknown_model_defaults_1536(self, provider):
        provider._model_name = "unknown-model"
        assert provider.dimension == 1536
