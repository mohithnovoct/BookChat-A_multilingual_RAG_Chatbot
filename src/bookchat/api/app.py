import logging
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from bookchat.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetResponse,
)
from bookchat.config import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES
from bookchat.core.generate import get_rag_chain
from bookchat.core.ingestion import ingest, init_qdrant_store, reset_store

logger = logging.getLogger(__name__)

app = FastAPI(title="BookChat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────── Constants ────────

STREAM_CHUNK = 1024 * 1024  # 1 MB chunks for streaming uploads to disk


# ──────── Helpers ────────

def _safe_filename(filename: str | None, suffix: str, used_names: set[str]) -> str:
    base_name = os.path.basename(filename or f"file{suffix}")
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    stem, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


async def _stream_to_disk(upload: UploadFile, dest: str, max_bytes: int) -> int:
    """Streams an uploaded file to disk in chunks, avoiding loading it entirely into RAM."""
    total = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(STREAM_CHUNK):
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File '{upload.filename}' exceeds the "
                        f"{max_bytes // (1024 * 1024)} MB upload limit."
                    ),
                )
            f.write(chunk)
    return total


# ──────── Endpoints ────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    tmp_dir = tempfile.mkdtemp()
    processed: List[str] = []
    used_names: set[str] = set()

    try:
        for upload in files:
            suffix = os.path.splitext(upload.filename or "")[1].lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type '{suffix}'. "
                        f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
                    ),
                )

            safe_name = _safe_filename(upload.filename, suffix, used_names)
            tmp_path = os.path.join(tmp_dir, safe_name)

            # Stream file to disk in chunks instead of reading entirely into memory
            await _stream_to_disk(upload, tmp_path, MAX_UPLOAD_BYTES)
            processed.append(safe_name)

        result = ingest(tmp_dir)

        if result.chunk_count == 0:
            detail = "No documents could be ingested."
            if result.warnings:
                detail = f"{detail} {'; '.join(result.warnings)}"
            raise HTTPException(status_code=400, detail=detail)

        message = f"Successfully ingested {len(processed)} file(s) into {result.chunk_count} chunks."
        if result.files_replaced:
            message = (
                f"{message} Replaced existing data for: "
                f"{', '.join(result.files_replaced)}."
            )
        if result.warnings:
            message = f"{message} {len(result.warnings)} file(s) failed to load."

        return IngestResponse(
            message=message,
            files_processed=processed,
            chunks_created=result.chunk_count,
            files_replaced=result.files_replaced,
            warnings=result.warnings,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
async def query_documents(body: QueryRequest):
    try:
        store = init_qdrant_store()
        chain = get_rag_chain(store=store, k=body.k, lang=body.lang)
        answer = chain.invoke(body.question)
        return QueryResponse(answer=answer, question=body.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=500, detail="Failed to process the query."
        ) from None


@app.delete("/reset", response_model=ResetResponse)
async def reset_vectorstore():
    try:
        reset_store()
        return ResetResponse(message="Vectorstore has been reset successfully.")
    except Exception:
        logger.exception("Reset failed")
        raise HTTPException(
            status_code=500, detail="Failed to reset the vectorstore."
        ) from None
