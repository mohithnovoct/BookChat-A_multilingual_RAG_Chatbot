import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

QDRANT_PATH = os.environ.get("QDRANT_PATH", "./local_qdrant_db")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "BAAI/bge-m3"
)

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "kan+pan+eng")

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


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for querying. "
            "Set it in a .env file or your environment."
        )
    return token
