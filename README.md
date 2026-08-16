# DocQuery

[![CI](https://github.com/cloud0523/agent/actions/workflows/ci.yml/badge.svg)](https://github.com/cloud0523/agent/acti
  ons/workflows/ci.yml)

A production-quality Retrieval-Augmented Generation (RAG) pipeline for document question-answering. Ingest your PDF, DOCX, TXT, and Markdown files, then ask natural-language questions — answers are grounded in your documents with cited sources. Comes with a **CLI** for power users and a **React web app** with streaming chat.

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **Flexible chunking** — fixed-size, sentence-boundary, or recursive (LlamaIndex)
- **Local embeddings** — runs offline with `all-MiniLM-L6-v2` (Sentence Transformers)
- **Pluggable LLM** — Claude, OpenAI, or local Ollama
- **Streaming** — SSE-powered token-by-token output in CLI and web UI
- **Conversation memory** — multi-turn chat with context-aware follow-up
- **Hybrid retrieval** — vector search (ChromaDB) + optional BM25 reranker
- **Source citations** — every answer linked back to the original document chunks
- **178 tests** — full coverage of core logic, API, and configuration

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  CLI (Typer) │────▶│           RAG Pipeline               │
│  or          │     │                                      │
│  Web (React) │◀───▶│  Query ─▶ Retrieve ─▶ Generate       │
└─────────────┘     │    │           │            │         │
                    │    ▼           ▼            ▼         │
                    │  Embed    ChromaDB      LLM API       │
                    │  (local   (cosine       (Claude/      │
                    │   or API)  distance)    OpenAI/Ollama) │
                    └──────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the web frontend)

### 1. Clone & Install

```bash
cd rag-agent
python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set your LLM credentials:

```env
LLM_PROVIDER=claude              # or "openai" / "ollama"
ANTHROPIC_API_KEY=sk-ant-...     # your API key
```

For OpenAI-compatible providers (DeepSeek, etc.), also set:

```env
OPENAI_BASE_URL=https://api.deepseek.com
```

### 3. Ingest Documents

```bash
rag-cli ingest docs/report.pdf
rag-cli ingest_dir docs/          # batch ingest a folder
```

### 4. Ask Questions

```bash
rag-cli ask "What is the main conclusion of the report?"
rag-cli chat                      # interactive multi-turn session
```

### 5. Launch Web App

```bash
# Terminal 1 — API server
rag-cli serve --reload

# Terminal 2 — React frontend
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — upload documents and chat through the browser.

## CLI Reference

| Command | Description |
|---------|-------------|
| `rag-cli ingest <file>` | Ingest a single document |
| `rag-cli ingest_dir <dir>` | Batch ingest all documents in a directory |
| `rag-cli ask <question>` | Ask a question (streaming output) |
| `rag-cli chat` | Start an interactive conversation |
| `rag-cli serve` | Start the FastAPI + web server |
| `rag-cli list` | List all indexed documents |
| `rag-cli delete <id>` | Remove a document and its vectors |
| `rag-cli status` | Show configuration and system status |

## Configuration

All settings are defined in `src/rag_agent/config.py` and can be overridden via `.env` or environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `local` | `local` (all-MiniLM) or `openai` |
| `LLM_PROVIDER` | `claude` | `claude`, `openai`, or `ollama` |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `CHUNKING_STRATEGY` | `recursive` | `fixed`, `sentence`, or `recursive` |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum cosine similarity for retrieval |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns kept in memory |
| `DATA_DIR` | `data` | Root data directory |
| `LOG_LEVEL` | `INFO` | Logging level |

> [!TIP]
> On Windows, set `HF_ENDPOINT=https://hf-mirror.com` to download embedding models from a mirror.

## API

The FastAPI server exposes a REST API at `http://localhost:8000`. Interactive docs are available at **/docs** (Swagger UI).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Ask a question (sync or SSE stream) |
| `POST` | `/api/ingest/document` | Ingest by file path |
| `POST` | `/api/ingest/upload` | Upload a file directly |
| `POST` | `/api/ingest/directory` | Ingest all files in a directory |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/documents/{id}` | Get document metadata |

See [docs/api.md](docs/api.md) for request/response schemas and SSE streaming details.

## Project Structure

```
rag-agent/
├── src/rag_agent/
│   ├── main.py                  # CLI entry point (Typer)
│   ├── config.py                # pydantic-settings config
│   ├── document/
│   │   ├── schemas.py           # Document, Chunk models
│   │   ├── loader.py            # PDF/DOCX/TXT/MD loader
│   │   └── chunker.py           # fixed/sentence/recursive chunking
│   ├── embeddings/
│   │   ├── base.py              # Abstract embedding provider
│   │   ├── local.py             # Sentence Transformers (all-MiniLM)
│   │   ├── openai_provider.py   # OpenAI embeddings API
│   │   └── factory.py           # Provider factory
│   ├── storage/
│   │   ├── vector_store.py      # ChromaDB wrapper
│   │   └── doc_store.py         # SQLite metadata store
│   ├── retrieval/
│   │   └── retriever.py         # Vector + optional BM25 hybrid
│   ├── generation/
│   │   └── llm.py               # LLM generator (Claude/OpenAI/Ollama)
│   ├── pipeline/
│   │   ├── ingestion.py         # Ingest orchestration
│   │   └── query.py             # Query orchestration
│   ├── conversation/
│   │   └── memory.py            # Conversation history + token budget
│   ├── api/
│   │   ├── server.py            # FastAPI app factory + CORS
│   │   └── routes/
│   │       ├── query.py         # POST /api/query
│   │       ├── ingest.py        # POST /api/ingest/*
│   │       └── documents.py     # GET /api/documents
│   └── utils/
│       ├── errors.py            # Custom exception hierarchy
│       └── logging.py           # loguru setup
├── frontend/                    # React (Vite) SPA
├── tests/                       # 178 pytest tests
├── data/                        # Runtime data (created on first run)
├── docs/                        # Documentation
└── pyproject.toml
```

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=rag_agent --cov-report=term-missing

# Skip slow/integration tests
pytest -m "not slow and not integration"
```

**178 tests** covering:
- Document schemas, loaders, chunkers
- Embedding providers (local, OpenAI, factory)
- Storage (ChromaDB vector store, SQLite doc store)
- Retrieval, generation, query pipeline
- Conversation memory
- CLI + API endpoints (FastAPI TestClient)
- Configuration (pydantic-settings)

## License

MIT
