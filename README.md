# BookChat

RAG API for ingesting books and documents (PDF, TXT, MD) and asking questions about them.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Copy the environment template and add your Hugging Face token:

```bash
cp .env.example .env
```

Set `HF_TOKEN` in `.env`. You need this for the `/query` endpoint. Ingestion works without it.

## Run the API

```bash
uv run bookchat
```

Or with uvicorn directly:

```bash
uv run uvicorn bookchat.api.app:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## CLI chat (optional)

Run a terminal chat session against the local vector store:

```bash
uv run python -m bookchat.core.generate
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload PDF/TXT/MD files |
| POST | `/query` | Ask a question (`question`, optional `k`) |
| DELETE | `/reset` | Clear the vector store |

### Example

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@book.pdf"

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme?", "k": 4}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token (required for queries) |
| `CHROMA_DIR` | `./chroma` | Vector store directory |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Max upload size per file |

## Notes

- The `chroma/` directory stores the local vector database and is gitignored.
- Re-ingesting a file with the same name replaces its existing chunks (upsert by filename).
- The `/reset` endpoint is unauthenticated — do not expose it publicly without adding auth.
