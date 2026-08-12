import logging
import os
import shutil
import tempfile
import threading
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
from bookchat.core.ingestion import ingest, load_store, reset_store

logger = logging.getLogger(__name__)

_store = None
_store_lock = threading.Lock()

app = FastAPI(title="BookChat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store():
    global _store
    with _store_lock:
        if _store is None:
            _store = load_store()
        return _store


def invalidate_store() -> None:
    global _store
    with _store_lock:
        _store = None


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

            content = await upload.read()
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File '{upload.filename}' exceeds the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
                    ),
                )

            safe_name = _safe_filename(upload.filename, suffix, used_names)
            tmp_path = os.path.join(tmp_dir, safe_name)
            with open(tmp_path, "wb") as file_handle:
                file_handle.write(content)
            processed.append(safe_name)

        result = ingest(tmp_dir)
        invalidate_store()

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
        store = get_store()
        chain = get_rag_chain(store=store, k=body.k)
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
        invalidate_store()
        return ResetResponse(message="Vectorstore has been reset successfully.")
    except Exception:
        logger.exception("Reset failed")
        raise HTTPException(
            status_code=500, detail="Failed to reset the vectorstore."
        ) from None
