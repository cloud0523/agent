# DocQuery API

Base URL: `http://localhost:8000`

Interactive docs (Swagger UI): http://localhost:8000/docs

---

## POST /api/query

Ask a question against your ingested documents. Supports both synchronous and SSE streaming responses.

### Request

```json
{
  "question": "What is the main finding?",
  "top_k": 5,
  "stream": false,
  "conversation_history": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | `string` | ✅ | — | The question to ask |
| `top_k` | `int` | | `5` | Number of chunks to retrieve |
| `stream` | `bool` | | `false` | Enable SSE streaming |
| `conversation_history` | `array` \| `null` | | `null` | Previous messages for context |

### Response (sync, `stream: false`)

**200 OK**

```json
{
  "answer": "The main finding is that transformer models outperform RNNs...",
  "sources": [
    {
      "filename": "report.pdf",
      "chunk_index": 3,
      "text": "Transformer models achieve state-of-the-art..."
    }
  ]
}
```

### Response (streaming, `stream: true`)

**200 OK** — `Content-Type: text/event-stream`

```
data: {"type": "token", "data": "The"}
data: {"type": "token", "data": " main"}
data: {"type": "citation", "data": {"filename": "report.pdf", "chunk_index": 3, "text": "..."}}
data: {"type": "done", "data": ""}
```

Each SSE event contains a JSON object:

| `type` | `data` content | Meaning |
|--------|---------------|---------|
| `"token"` | `string` | A piece of generated text |
| `"citation"` | `object` | Source metadata for a retrieved chunk |
| `"done"` | `""` | Stream finished |

### Errors

| Status | Detail | Condition |
|--------|--------|-----------|
| `400` | `"question is required"` | `question` is empty or whitespace |
| `422` | Validation error | Missing `question` field or invalid types |

---

## POST /api/ingest/document

Ingest a document by its file path on the server.

### Request

```json
{
  "file_path": "/path/to/report.pdf",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_path` | `string` | ✅ | — | Absolute path to the document |
| `chunk_size` | `int` \| `null` | | `null` | Override chunk size (uses config default) |
| `chunk_overlap` | `int` \| `null` | | `null` | Override chunk overlap (uses config default) |

### Response

**200 OK** — `Document` object:

```json
{
  "id": "f47ac10b-...",
  "filename": "report.pdf",
  "file_type": "pdf",
  "file_path": "/path/to/report.pdf",
  "num_chunks": 12,
  "size_bytes": 45234,
  "status": "indexed",
  "uploaded_at": "2026-08-11T12:00:00Z"
}
```

### Errors

| Status | Detail | Condition |
|--------|--------|-----------|
| `404` | `"File not found: ..."` | `file_path` does not exist |
| `500` | Error message | Internal processing error |

---

## POST /api/ingest/upload

Upload a file directly via multipart form data.

### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `file` | ✅ | The document file (PDF, DOCX, TXT, MD) |

### Response

**200 OK** — `Document` object (same schema as above).

The uploaded file is saved to a temporary location during processing and automatically deleted afterward.

### Errors

| Status | Detail | Condition |
|--------|--------|-----------|
| `422` | Validation error | No file provided |

---

## POST /api/ingest/directory

Batch ingest all supported documents from a directory.

### Request

```json
{
  "directory": "/path/to/docs",
  "chunk_size": null,
  "chunk_overlap": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `directory` | `string` | ✅ | — | Path to the directory |
| `chunk_size` | `int` \| `null` | | `null` | Override chunk size |
| `chunk_overlap` | `int` \| `null` | | `null` | Override chunk overlap |

### Response

**200 OK** — Array of `Document` objects:

```json
[
  { "id": "abc-1", "filename": "a.pdf", "num_chunks": 5, "status": "indexed", ... },
  { "id": "abc-2", "filename": "b.docx", "num_chunks": 8, "status": "indexed", ... }
]
```

### Errors

| Status | Detail | Condition |
|--------|--------|-----------|
| `404` | `"Not a directory: ..."` | `directory` is not a directory |
| `500` | Error message | Internal processing error |

---

## GET /api/documents

List all indexed documents.

### Response

**200 OK** — Array of `Document` objects (sorted newest-first):

```json
[
  {
    "id": "f47ac10b-...",
    "filename": "report.pdf",
    "file_type": "pdf",
    "file_path": "/path/to/report.pdf",
    "num_chunks": 12,
    "size_bytes": 45234,
    "status": "indexed",
    "uploaded_at": "2026-08-11T12:00:00Z"
  }
]
```

Returns an empty array `[]` if no documents have been ingested.

---

## GET /api/documents/{document_id}

Get metadata for a specific document.

### Response

**200 OK** — `Document` object.

### Errors

| Status | Detail | Condition |
|--------|--------|-----------|
| `404` | `"Document not found"` | `document_id` does not exist |

---

## Document Schema

All endpoints that return documents use this schema:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | UUID |
| `filename` | `string` | Original filename |
| `file_type` | `string` | `"pdf"`, `"docx"`, `"txt"`, or `"md"` |
| `file_path` | `string` | Absolute path to the source file |
| `num_chunks` | `int` | Number of chunks this document was split into |
| `size_bytes` | `int` | File size in bytes |
| `status` | `string` | `"indexed"` or `"error"` |
| `uploaded_at` | `string` | ISO 8601 timestamp |
