import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", Path.cwd() / "chroma"))
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 50 * 1024 * 1024))


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is required for querying. "
            "Set it in a .env file or your environment."
        )
    return token
