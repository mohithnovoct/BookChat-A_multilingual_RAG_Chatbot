import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

QDRANT_PATH = os.environ.get("QDRANT_PATH", "./local_qdrant_db")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "BAAI/bge-m3"
)
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
)
RERANKER_CANDIDATE_K = int(os.environ.get("RERANKER_CANDIDATE_K", 40))
HYBRID_CANDIDATE_K = int(os.environ.get("HYBRID_CANDIDATE_K", 40))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "kan+pan+eng")
OCR_PSM = int(os.environ.get("OCR_PSM", 4))
OCR_OEM = int(os.environ.get("OCR_OEM", 1))

LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", 400))
MIN_RERANK_SCORE = float(os.environ.get("MIN_RERANK_SCORE", "-5.0"))

TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    str(BASE_DIR / "Tesseract-OCR" / "tesseract.exe")
    if (BASE_DIR / "Tesseract-OCR" / "tesseract.exe").exists()
    else None,
)

POPPLER_PATH = os.environ.get(
    "POPPLER_PATH",
    str(BASE_DIR / "poppler-26.02.0" / "Library" / "bin")
    if (BASE_DIR / "poppler-26.02.0" / "Library" / "bin").exists()
    else None,
)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))
FALLBACK_ANSWER = "Information not found in the source documents."


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for querying. "
            "Set it in a .env file or your environment."
        )
    return token
