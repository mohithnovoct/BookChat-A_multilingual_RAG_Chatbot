import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import unicodedata
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

from bookchat.config import (
    ALLOWED_SUFFIXES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OCR_LANGUAGES,
    POPPLER_PATH,
    QDRANT_PATH,
    TESSERACT_CMD,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_count: int
    chunk_count: int
    warnings: List[str] = field(default_factory=list)
    files_replaced: List[str] = field(default_factory=list)


# ──────── Lazy singletons ────────

_embeddings: HuggingFaceEmbeddings | None = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-loads the embedding model on first use to avoid blocking server startup."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"batch_size": 64},
            )
            logger.info("Embedding model loaded.")
        return _embeddings


_qdrant_client: QdrantClient | None = None
_qdrant_lock = threading.Lock()


def _get_client() -> QdrantClient:
    """Returns a shared Qdrant client, creating it on first call."""
    global _qdrant_client
    with _qdrant_lock:
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(path=QDRANT_PATH, timeout=60)
        return _qdrant_client


if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ──────── OCR helpers ────────

def _ocr_page(args: tuple[int, Any]) -> Document | None:
    page_idx, image = args
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
    if text.strip():
        return Document(
            page_content=text,
            metadata={"page": page_idx},
        )
    return None


def _ocr_pdf(file_path: str) -> List[Document]:
    """Performs batched parallel OCR to avoid loading all pages into memory at once."""
    logger.info("Performing batched OCR on '%s' using Tesseract (%s)...", file_path, OCR_LANGUAGES)

    OCR_BATCH_SIZE = 10
    poppler_kwargs: dict[str, Any] = {}
    if POPPLER_PATH:
        poppler_kwargs["poppler_path"] = POPPLER_PATH

    # Get total page count without loading images
    info = pdfinfo_from_path(file_path, **poppler_kwargs)
    total_pages = info["Pages"]

    documents: List[Document] = []
    max_workers = min(os.cpu_count() or 4, 8)

    for batch_start in range(1, total_pages + 1, OCR_BATCH_SIZE):
        batch_end = min(batch_start + OCR_BATCH_SIZE - 1, total_pages)
        images = convert_from_path(
            file_path,
            first_page=batch_start,
            last_page=batch_end,
            dpi=150,
            **poppler_kwargs,
        )

        page_args = [(batch_start - 1 + i, img) for i, img in enumerate(images)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_ocr_page, page_args))

        for result in results:
            if result:
                result.metadata["source"] = file_path
                documents.append(result)

        # Explicitly free image memory before loading next batch
        del images

    return documents


# ──────── Document loading ────────

def _normalize_docs(docs: List[Document]) -> List[Document]:
    for doc in docs:
        if doc.page_content:
            doc.page_content = unicodedata.normalize("NFC", doc.page_content)
    return docs


def _load_single_file(file_path: str) -> List[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    docs: List[Document] = []

    if ext == ".pdf":
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            total_text = "".join(d.page_content.strip() for d in docs)
            if len(total_text) < 50:
                logger.info("PDF text extraction resulted in minimal text; attempting OCR fallback.")
                ocr_docs = _ocr_pdf(file_path)
                if ocr_docs:
                    docs = ocr_docs
        except Exception as exc:
            logger.warning("Standard PDF load failed (%s); attempting OCR fallback.", exc)
            docs = _ocr_pdf(file_path)
    elif ext in {".txt", ".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError(
            f"Unsupported file format {ext}. Supported file formats: .pdf, .txt, .md"
        )

    return _normalize_docs(docs)


def load_documents(path: str) -> tuple[List[Document], List[str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    documents: List[Document] = []
    warnings: List[str] = []

    if target.is_file():
        documents.extend(_load_single_file(str(target)))
        return documents, warnings
    
    if target.is_dir():
        file_paths = []
        for root, _, files in os.walk(target):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in ALLOWED_SUFFIXES:
                    file_paths.append(str(file_path))

        max_workers = min(os.cpu_count() or 4, 16)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_load_single_file, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as exc:
                    msg = f"Failed to load '{fp}': {exc}"
                    logger.warning(msg)
                    warnings.append(msg)

        return documents, warnings

    raise ValueError(f"Invalid path type: {path}")


# ──────── Chunking ────────

def get_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(documents)

    per_file_chunk_id: dict[str, int] = {}
    for chunk in chunks:
        source_path = chunk.metadata.get("source", "")
        filename = os.path.basename(source_path) if source_path else "junknown"
        chunk.metadata["filename"] = filename
        chunk_index = per_file_chunk_id.get(filename, 0)
        chunk.metadata["chunk_id"] = chunk_index
        per_file_chunk_id[filename] = chunk_index + 1

    return chunks


# ──────── Qdrant store ────────

def init_qdrant_store() -> QdrantVectorStore:
    """Initializes Qdrant collection with optimized config and payload indexes."""
    client = _get_client()
    
    if not client.collection_exists(collection_name="test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
        # Create payload index for fast filtered deletes/lookups by filename
        client.create_payload_index(
            collection_name="test",
            field_name="metadata.filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection 'test' with payload index on metadata.filename")
    
    return QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=_get_embeddings(),
    )


def reset_store() -> None:
    """Removes the entire local Qdrant directory tree and resets the shared client."""
    global _qdrant_client
    with _qdrant_lock:
        _qdrant_client = None

    directory = Path(QDRANT_PATH)
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Reset local Qdrant vectorstore directory at '%s'", directory)


def delete_by_filenames(store: QdrantVectorStore, filenames: Set[str]) -> List[str]:
    """Deletes entries by filename, only reporting filenames that actually had data."""
    replaced: List[str] = []
    if not filenames:
        return replaced

    for filename in sorted(filenames):
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="metadata.filename",
                        match=MatchValue(value=filename)
                    )
                ]
            )

            # Check if any points actually exist for this filename
            scroll_result = store.client.scroll(
                collection_name=store.collection_name,
                scroll_filter=filter_condition,
                limit=1,
            )
            if not scroll_result[0]:
                continue

            store.client.delete(
                collection_name=store.collection_name,
                points_selector=FilterSelector(filter=filter_condition)
            )
            replaced.append(filename)
            logger.info("Replaced existing data for '%s'", filename)
        except Exception as e:
            logger.error("Failed to delete records for %s: %s", filename, e)
            
    return replaced


def build_index(store: QdrantVectorStore, chunks: List[Document]) -> Tuple[QdrantVectorStore, List[str]]:
    """Batches document additions to prevent network overhead bottlenecks."""
    filenames = {chunk.metadata.get("filename") for chunk in chunks if chunk.metadata.get("filename")}
    replaced = delete_by_filenames(store, filenames)

    if chunks:
        # Micro-batching chunk uploads minimizes server network overhead locks
        batch_size = 256
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            store.add_documents(batch)
            
    return store, replaced


def ingest(docs_path: str) -> IngestResult:
    """Main orchestrator: load → chunk → index."""
    store = init_qdrant_store()

    docs, warnings = load_documents(docs_path)
    chunks = get_chunks(docs)

    _, replaced = build_index(store, chunks)
    logger.info("Ingested %d documents into %d chunks", len(docs), len(chunks))

    return IngestResult(
        document_count=len(docs),
        chunk_count=len(chunks),
        warnings=warnings,
        files_replaced=replaced,
    )
