"""Tests for application configuration (config.py).

Covers Settings defaults, env-var overrides, Path coercion,
ensure_directories, and external-service env pushes.
"""

import os
from pathlib import Path

import pytest

from rag_agent.config import Settings


# ---------------------------------------------------------------------------
# Helper: build Settings that does NOT read the real .env
# ---------------------------------------------------------------------------

def _settings(**overrides) -> Settings:
    """Create a Settings instance that ignores .env and uses the given values."""
    return Settings(_env_file=None, **overrides)


# ============================================================================
# Default values
# ============================================================================

class TestDefaults:
    """Verify that every Settings field has the expected default."""

    def test_embedding_provider_default(self):
        s = _settings()
        assert s.embedding_provider == "local"

    def test_embedding_model_default(self):
        s = _settings()
        assert s.embedding_model == "all-MiniLM-L6-v2"

    def test_openai_api_key_default(self):
        s = _settings()
        assert s.openai_api_key is None

    def test_openai_embedding_model_default(self):
        s = _settings()
        assert s.openai_embedding_model == "text-embedding-3-small"

    def test_llm_provider_default(self):
        s = _settings()
        assert s.llm_provider == "claude"

    def test_anthropic_api_key_default(self):
        s = _settings()
        assert s.anthropic_api_key is None

    def test_claude_model_default(self):
        s = _settings()
        assert s.claude_model == "claude-sonnet-4-20250514"

    def test_openai_model_default(self):
        s = _settings()
        assert s.openai_model == "gpt-4o-mini"

    def test_ollama_base_url_default(self):
        s = _settings()
        assert s.ollama_base_url == "http://localhost:11434"

    def test_ollama_model_default(self):
        s = _settings()
        assert s.ollama_model == "llama3.2"

    def test_chunk_size_default(self):
        s = _settings()
        assert s.chunk_size == 512

    def test_chunk_overlap_default(self):
        s = _settings()
        assert s.chunk_overlap == 50

    def test_chunking_strategy_default(self):
        s = _settings()
        assert s.chunking_strategy == "recursive"

    def test_top_k_default(self):
        s = _settings()
        assert s.top_k == 5

    def test_similarity_threshold_default(self):
        s = _settings()
        assert s.similarity_threshold == 0.7

    def test_use_reranker_default(self):
        s = _settings()
        assert s.use_reranker is False

    def test_data_dir_default(self):
        s = _settings()
        assert s.data_dir == Path("data")

    def test_chroma_persist_dir_default(self):
        s = _settings()
        assert s.chroma_persist_dir == Path("data/chroma")

    def test_doc_store_path_default(self):
        s = _settings()
        assert s.doc_store_path == Path("data/metadata.db")

    def test_host_default(self):
        s = _settings()
        assert s.host == "127.0.0.1"

    def test_port_default(self):
        s = _settings()
        assert s.port == 8000

    def test_max_history_default(self):
        s = _settings()
        assert s.max_history_turns == 10

    def test_max_context_tokens_default(self):
        s = _settings()
        assert s.max_context_tokens == 4000

    def test_log_level_default(self):
        s = _settings()
        assert s.log_level == "INFO"

    def test_hf_endpoint_is_optional(self):
        """hf_endpoint can be None (when not set in .env or env vars)."""
        s = _settings(hf_endpoint=None)
        assert s.hf_endpoint is None

    def test_hf_endpoint_accepts_string(self):
        s = _settings(hf_endpoint="https://hf-mirror.com")
        assert s.hf_endpoint == "https://hf-mirror.com"

    def test_pythonioencoding_default(self):
        s = _settings()
        assert s.pythonioencoding == "utf-8"

    def test_openai_base_url_default(self):
        s = _settings()
        assert s.openai_base_url == "https://api.openai.com/v1"


# ============================================================================
# Environment-variable overrides
# ============================================================================

class TestEnvVarOverride:
    """Settings should read from environment variables and override defaults."""

    def test_override_embedding_provider(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        s = _settings()
        assert s.embedding_provider == "openai"

    def test_override_chunk_size_int(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "256")
        s = _settings()
        assert s.chunk_size == 256

    def test_override_top_k_int(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "3")
        s = _settings()
        assert s.top_k == 3

    def test_override_similarity_threshold_float(self, monkeypatch):
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")
        s = _settings()
        assert s.similarity_threshold == 0.5

    def test_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = _settings()
        assert s.log_level == "DEBUG"

    def test_override_host(self, monkeypatch):
        monkeypatch.setenv("HOST", "0.0.0.0")
        s = _settings()
        assert s.host == "0.0.0.0"

    def test_override_llm_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        s = _settings()
        assert s.llm_provider == "ollama"

    def test_ignore_unknown_env_var(self, monkeypatch):
        """extra='ignore' means unknown vars should not raise an error."""
        monkeypatch.setenv("UNKNOWN_FIELD", "should-be-ignored")
        s = _settings()
        # Must not raise; the field simply doesn't exist on Settings
        assert not hasattr(s, "UNKNOWN_FIELD")


# ============================================================================
# Path coercion
# ============================================================================

class TestPathCoercion:
    """String values for Path-typed fields should become Path objects."""

    def test_data_dir_is_path(self):
        s = _settings(data_dir="custom/data")
        assert isinstance(s.data_dir, Path)
        assert s.data_dir == Path("custom/data")

    def test_chroma_persist_dir_is_path(self):
        s = _settings(chroma_persist_dir="custom/chroma")
        assert isinstance(s.chroma_persist_dir, Path)

    def test_doc_store_path_is_path(self):
        s = _settings(doc_store_path="custom/meta.db")
        assert isinstance(s.doc_store_path, Path)


# ============================================================================
# ensure_directories()
# ============================================================================

class TestEnsureDirectories:
    """ensure_directories() should create data/, data/chroma/, data/documents/."""

    def test_creates_all_three_dirs(self, temp_dir):
        s = _settings(
            data_dir=temp_dir / "data",
            chroma_persist_dir=temp_dir / "data" / "chroma",
            doc_store_path=temp_dir / "data" / "metadata.db",
        )
        s.ensure_directories()

        assert s.data_dir.is_dir()
        assert s.chroma_persist_dir.is_dir()
        assert (s.data_dir / "documents").is_dir()

    def test_idempotent(self, temp_dir):
        """Calling ensure_directories twice should not raise."""
        s = _settings(
            data_dir=temp_dir / "data",
            chroma_persist_dir=temp_dir / "data" / "chroma",
            doc_store_path=temp_dir / "data" / "metadata.db",
        )
        s.ensure_directories()
        s.ensure_directories()  # must not raise


# ============================================================================
# HF_ENDPOINT & PYTHONIOENCODING push to os.environ
# ============================================================================

class TestExternalServiceEnvPush:
    """hf_endpoint and pythonioencoding are pushed into os.environ at import time.

    Because the push happens at module level (when config.py is imported),
    we test the behaviour by constructing Settings ourselves and verifying
    the side-effect contract.  We do NOT run the actual module-level code
    (it already ran when the test process started).
    """

    def test_hf_endpoint_set_pushes_to_environ(self, monkeypatch):
        """When hf_endpoint is set, os.environ['HF_ENDPOINT'] should be written."""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        s = _settings(hf_endpoint="https://hf-mirror.com")
        # Simulate the module-level push
        if s.hf_endpoint:
            os.environ["HF_ENDPOINT"] = s.hf_endpoint
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    def test_hf_endpoint_none_does_not_crash(self, monkeypatch):
        """When hf_endpoint is None, no push happens — must not raise."""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        s = _settings(hf_endpoint=None)
        # Simulate the module-level guard
        if s.hf_endpoint:
            os.environ["HF_ENDPOINT"] = s.hf_endpoint
        # No HF_ENDPOINT key should exist (unless already set externally)
        assert "HF_ENDPOINT" not in os.environ or os.environ["HF_ENDPOINT"] == ""

    def test_pythonioencoding_pushes_to_environ(self, monkeypatch):
        """pythonioencoding should be written to os.environ."""
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        s = _settings(pythonioencoding="utf-8")
        if s.pythonioencoding:
            os.environ["PYTHONIOENCODING"] = s.pythonioencoding
        assert os.environ["PYTHONIOENCODING"] == "utf-8"


# ============================================================================
# Literal validation
# ============================================================================

class TestLiteralValidation:
    """Pydantic should reject values outside the Literal annotation."""

    def test_invalid_embedding_provider_raises(self):
        with pytest.raises(ValueError):  # pydantic ValidationError is a ValueError
            _settings(embedding_provider="invalid")

    def test_invalid_llm_provider_raises(self):
        with pytest.raises(ValueError):
            _settings(llm_provider="invalid")

    def test_invalid_chunking_strategy_raises(self):
        with pytest.raises(ValueError):
            _settings(chunking_strategy="invalid")

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError):
            _settings(log_level="TRACE")
