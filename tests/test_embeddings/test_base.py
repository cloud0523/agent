import pytest

from rag_agent.embeddings.base import EmbeddingProvider


class TestEmbeddingProviderABC:
    """Verify that EmbeddingProvider enforces its interface."""

    def test_cannot_instantiate_abc_directly(self):
        """EmbeddingProvider is abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_subclass_missing_embed_cannot_instantiate(self):
        """A subclass without embed() is still abstract."""
        with pytest.raises(TypeError):

            class MissingEmbed(EmbeddingProvider):
                def embed_query(self, text):
                    return [0.0]

                @property
                def dimension(self):
                    return 1

            MissingEmbed()  # type: ignore[abstract]

    def test_subclass_missing_embed_query_cannot_instantiate(self):
        """A subclass without embed_query() is still abstract."""
        with pytest.raises(TypeError):

            class MissingEmbedQuery(EmbeddingProvider):
                def embed(self, texts):
                    return [[0.0]]

                @property
                def dimension(self):
                    return 1

            MissingEmbedQuery()  # type: ignore[abstract]

    def test_subclass_missing_dimension_cannot_instantiate(self):
        """A subclass without dimension property is still abstract."""
        with pytest.raises(TypeError):

            class MissingDimension(EmbeddingProvider):
                def embed(self, texts):
                    return [[0.0]]

                def embed_query(self, text):
                    return [0.0]

            MissingDimension()  # type: ignore[abstract]

    def test_full_subclass_can_instantiate(self):
        """A subclass implementing all three abstract methods works."""

        class FullProvider(EmbeddingProvider):
            def embed(self, texts):
                return [[0.1, 0.2] for _ in texts]

            def embed_query(self, text):
                return [0.1, 0.2]

            @property
            def dimension(self):
                return 2

        provider = FullProvider()
        assert provider.dimension == 2
        assert provider.embed_query("hello") == [0.1, 0.2]
        assert provider.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]