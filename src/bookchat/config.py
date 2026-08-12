import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", BASE_DIR / "chroma"))
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "BAAI/bge-m3"
)

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
