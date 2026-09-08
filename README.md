# BookChat

Multilingual RAG API and UI for ingesting books and documents (PDF, TXT, MD) in **English**, **Kannada (ಕನ್ನಡ)**, and **Punjabi (ਪੰਜਾਬੀ)**, then asking English questions answered from multilingual context with Tesseract OCR support.

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
| POST | `/query` | Ask an English question across all ingested languages (`question`, optional `k`) |
| DELETE | `/reset` | Clear the vector store |

### Example

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@book.pdf"

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main theme?", "k": 4, "answer_language": "en", "query_language": "auto"}'
```

The query endpoint embeds the English question with the multilingual `BAAI/bge-m3` model, retrieves a broad candidate set, and reranks candidates with the multilingual `BAAI/bge-reranker-v2-m3` cross-encoder before generating an English answer from the best chunks. It searches the shared index containing English, Kannada, and Punjabi chunks. The response also includes structured source references with filename, page, chunk ID, and retrieval score when available.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token (required for queries) |
| `QDRANT_PATH` | `./local_qdrant_db` | Local Qdrant vector store directory |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Multilingual embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder reranker |
| `RERANKER_CANDIDATE_K` | `20` | Dense candidates passed to the reranker |
| `CHUNK_SIZE` | `2500` | Characters per chunk for text splitting |
| `CHUNK_OVERLAP` | `400` | Overlap between consecutive chunks |
| `OCR_LANGUAGES` | `kan+pan+eng` | Tesseract OCR language packs |
| `MAX_UPLOAD_BYTES` | `524288000` (500 MB) | Max upload size per file |

## Notes

- The `local_qdrant_db/` directory stores the local Qdrant vector database and is gitignored.
- The first-release query workflow searches all supported document languages and answers in English.
- Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires re-ingesting documents because existing vectors use the previous chunk boundaries.
- Re-ingesting a file with the same name replaces its existing chunks (upsert by filename).
- The `/reset` endpoint is unauthenticated — do not expose it publicly without adding auth.
