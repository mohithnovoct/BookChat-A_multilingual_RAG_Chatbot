import tempfile
from typing import List
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from .schemas import QueryRequest, QueryResponse, IngestResponse, ResetResponse

from bookchat.core.ingestion import ingest, load_store


from dotenv import load_dotenv

load_dotenv()

_store = None

def get_store():
    global _store
    if _store is None:
        _store = load_store()
    return _store

def invalidate_store():
    global _store
    _store = None
    
app = FastAPI(title="BookChat")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):

    ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
    tmp_dir = tempfile.mkdtemp()
    processed: List[str] = []

    try:
        for upload in files:
            suffix = os.path.splitext(upload.filename or "")[1].lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{suffix}'. Allowed: .pdf, .txt, .md",
                )
            tmp_path = os.path.join(tmp_dir, upload.filename or f"file{suffix}")
            with open(tmp_path, "wb") as f:
                content = await upload.read()
                f.write(content)
            processed.append(upload.filename or tmp_path)

        
        ingest(tmp_dir, reset=False)
        invalidate_store()

        return IngestResponse(
            message=f"Successfully ingested {len(processed)} file(s).",
            files_processed=processed,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
