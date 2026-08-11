"""Application configuration via pydantic-settings.

All tunable parameters are defined here. Values are loaded from
environment variables or a .env file.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Embedding ---
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- LLM ---
    llm_provider: Literal["claude", "openai", "ollama"] = "claude"
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_strategy: Literal["fixed", "sentence", "recursive"] = "recursive"

    # --- Retrieval ---
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranker: bool = False

    # --- Storage ---
    data_dir: Path = Path("data")
    chroma_persist_dir: Path = Path("data/chroma")
    doc_store_path: Path = Path("data/metadata.db")

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Conversation ---
    max_history_turns: int = 10
    max_context_tokens: int = 4000

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- External services ---
    hf_endpoint: Optional[str] = None
    pythonioencoding: str = "utf-8"
    openai_base_url: str = "https://api.openai.com/v1"

    def ensure_directories(self) -> None:
        """Create runtime data directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        documents_dir = self.data_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance — the single source of truth
settings = Settings()

if settings.hf_endpoint:
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint

if settings.pythonioencoding:
    os.environ["PYTHONIOENCODING"] = settings.pythonioencoding
