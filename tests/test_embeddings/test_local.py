from unittest.mock import MagicMock, patch

import pytest

from rag_agent.embeddings.local import LocalEmbeddingProvider
from rag_agent.utils.errors import EmbeddingModelError


class TestLocalEmbeddingProvider:
    """Tests for LocalEmbeddingProvider using a mocked SentenceTransformer."""

    @pytest.fixture
    def fake_st_model(self):
        """A mock SentenceTransformer returning known values."""
        model = MagicMock()

        def fake_encode(payload, convert_to_numpy=True, show_progress_bar=False):
            if isinstance(payload, list):
                return MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2, 0.3] for _ in payload]))
            return MagicMock(tolist=MagicMock(return_value=[0.1, 0.2, 0.3]))

        model.encode.side_effect = fake_encode
        model.get_sentence_embedding_dimension.return_value = 3
        return model

    @pytest.fixture
    def provider(self, fake_st_model):
        """Create a LocalEmbeddingProvider with mocked model."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            return LocalEmbeddingProvider()

    def test_default_model_name(self, provider):
        assert provider._model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self, fake_st_model):
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_st_model,
        ):
            provider = LocalEmbeddingProvider(model_name="custom-model")
        assert provider._model_name == "custom-model"

    def test_embed_returns_list_of_lists(self, provider):
        embeddings = provider.embed(["a", "b"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(item, list) for item in embeddings)

    def test_embed_calls_encode_correctly(self, provider, fake_st_model):
        provider.embed(["a", "b"])
        fake_st_model.encode.assert_called_once_with(
            ["a", "b"], convert_to_numpy=True, show_progress_bar=False
        )

    def test_embed_query_returns_flat_list(self, provider):
        result = provider.embed_query("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_dimension(self, provider):
        assert provider.dimension == 3

    def test_embedding_dimension_matches_model(self, provider):
        first_embedding = provider.embed(["x"])[0]
        assert len(first_embedding) == provider.dimension

    def test_init_failure_raises_embedding_model_error(self):
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=RuntimeError("failed to load model"),
        ):
            with pytest.raises(EmbeddingModelError):
                LocalEmbeddingProvider()