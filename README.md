# BookChat

Multilingual RAG API and UI for ingesting books and documents (PDF, TXT, MD) and chatting with them in **English**, **Kannada (ಕನ್ನಡ)**, and **Punjabi (ਪੰਜਾਬੀ)** with Tesseract OCR support.

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
| POST | `/query` | Ask a question (`question`, optional `k`, `lang`) |
| DELETE | `/reset` | Clear the vector store |

### Example

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@book.pdf"

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme?", "k": 4, "lang": "en"}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token (required for queries) |
| `QDRANT_PATH` | `./local_qdrant_db` | Local Qdrant vector store directory |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `CHUNK_SIZE` | `1200` | Characters per chunk for text splitting |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `OCR_LANGUAGES` | `kan+pan+eng` | Tesseract OCR language packs |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | Max upload size per file |

## Notes

- The `local_qdrant_db/` directory stores the local Qdrant vector database and is gitignored.
- Re-ingesting a file with the same name replaces its existing chunks (upsert by filename).
- The `/reset` endpoint is unauthenticated — do not expose it publicly without adding auth.
