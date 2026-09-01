from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import unicodedata

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pdf2image import convert_from_path, pdfinfo_from_path
from pypdf import PdfReader
import pytesseract
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
    warnings: list[str] = field(default_factory=list)
    files_replaced: list[str] = field(default_factory=list)


# ──────── Lazy singletons ────────

_embeddings: HuggingFaceEmbedding | None = None
_embeddings_lock = threading.Lock()


def _get_embeddings() -> HuggingFaceEmbedding:
    """Lazy-loads the embedding model on first use to avoid blocking server startup."""
    global _embeddings
    with _embeddings_lock:
        if _embeddings is None:
            logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
            _embeddings = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL,
                embed_batch_size=64,
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


def _clean_multilingual_text(text: str) -> str:
    """Normalizes Unicode (NFC) and handles sentence boundaries across English, Punjabi, and Kannada."""
    if not text:
        return ""

    # 1. Standard Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Fix spacing around Gurmukhi/Kannada/English sentence delimiters (., !, ?, ।, ॥)
    text = re.sub(r"\s*([।॥\.\!\?])\s*", r"\1 ", text)

    # 3. Collapse multiple whitespace while retaining paragraphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _is_valid_multilingual_content(text: str) -> bool:
    """
    Validates content for English (Latin), Punjabi (Gurmukhi), and Kannada.
    Unicode Ranges:
    - English / Latin: U+0020 - U+007F
    - Gurmukhi (Punjabi): U+0A00 - U+0A7F
    - Kannada: U+0C80 - U+0CFF
    """
    if not text or len(text.strip()) < 30:
        return False

    # Matches Latin, Gurmukhi, and Kannada scripts
    valid_script_chars = len(
        re.findall(r"[\u0020-\u007F\u0A00-\u0A7F\u0C80-\u0CFF]", text)
    )
    total_chars = len(text.strip())

    return (valid_script_chars / total_chars) > 0.50


# ──────── OCR helpers ────────


def _ocr_single_page(file_path: str, page_num: int, filename: str) -> Document | None:
    """Runs OCR on a single PDF page if text extraction yields minimal content."""
    poppler_kwargs: dict[str, any] = {}
    if POPPLER_PATH:
        poppler_kwargs["poppler_path"] = POPPLER_PATH
    try:
        images = convert_from_path(
            file_path,
            first_page=page_num,
            last_page=page_num,
            dpi=300,
            **poppler_kwargs,
        )
        if images:
            custom_config = f"-l {OCR_LANGUAGES} --psm 3"
            text = pytesseract.image_to_string(
                images[0], lang=OCR_LANGUAGES, config=custom_config
            )
            if text.strip():
                return Document(
                    text=text,
                    metadata={
                        "source": file_path,
                        "filename": filename,
                        "page": page_num,
                    },
                )
    except Exception as exc:
        logger.warning(
            "Single page OCR failed for page %d of '%s': %s", page_num, file_path, exc
        )
    return None


def _ocr_page(args: tuple[int, any]) -> Document | None:
    page_idx, image = args
    text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
    if text.strip():
        return Document(
            text=text,
            metadata={"page": page_idx},
        )
    return None


def _ocr_pdf(file_path: str) -> list[Document]:
    """Performs batched parallel OCR to avoid loading all pages into memory at once."""
    logger.info(
        "Performing batched OCR on '%s' using Tesseract (%s)...",
        file_path,
        OCR_LANGUAGES,
    )

    OCR_BATCH_SIZE = 10
    poppler_kwargs: dict[str, any] = {}
    if POPPLER_PATH:
        poppler_kwargs["poppler_path"] = POPPLER_PATH

    info = pdfinfo_from_path(file_path, **poppler_kwargs)
    total_pages = info["Pages"]

    documents: list[Document] = []
    max_workers = min(os.cpu_count() or 4, 8)
    filename = Path(file_path).name

    for batch_start in range(1, total_pages + 1, OCR_BATCH_SIZE):
        batch_end = min(batch_start + OCR_BATCH_SIZE - 1, total_pages)
        images = convert_from_path(
            file_path,
            first_page=batch_start,
            last_page=batch_end,
            dpi=300,
            **poppler_kwargs,
        )

        page_args = [(batch_start - 1 + i, img) for i, img in enumerate(images)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_ocr_page, page_args))

        for result in results:
            if result:
                result.metadata["source"] = file_path
                result.metadata["filename"] = filename
                documents.append(result)

        del images

    return documents


# ──────── Document loading ────────


def _normalize_docs(docs: list[Document]) -> list[Document]:
    for doc in docs:
        content = doc.get_content()
        if content:
            doc.set_content(unicodedata.normalize("NFC", content))
    return docs


def _load_single_file(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    filename = path.name
    docs: list[Document] = []

    if ext == ".pdf":
        logger.info("Loading PDF document: '%s'", filename)
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if len(text) >= 150:
                    docs.append(
                        Document(
                            text=text,
                            metadata={
                                "source": file_path,
                                "filename": filename,
                                "page": i + 1,
                            },
                        )
                    )
                else:
                    logger.info(
                        "Page %d of '%s' has minimal text; running single-page OCR fallback.",
                        i + 1,
                        filename,
                    )
                    ocr_doc = _ocr_single_page(
                        file_path, page_num=i + 1, filename=filename
                    )
                    if ocr_doc:
                        docs.append(ocr_doc)

            if not docs and total_pages > 0:
                logger.info(
                    "Standard PDF extraction found no text across all pages; falling back to full OCR."
                )
                docs = _ocr_pdf(file_path)
        except Exception as exc:
            logger.warning(
                "Standard PDF load failed (%s); attempting full OCR fallback.", exc
            )
            docs = _ocr_pdf(file_path)
    elif ext in {".txt", ".md"}:
        logger.info("Loading text document: '%s'", filename)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        docs = [
            Document(
                text=content,
                metadata={"source": file_path, "filename": filename},
            )
        ]
    else:
        raise ValueError(
            f"Unsupported file format {ext}. Supported file formats: .pdf, .txt, .md"
        )

    return _normalize_docs(docs)


def load_documents(path: str) -> tuple[list[Document], list[str]]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    documents: list[Document] = []
    warnings: list[str] = []

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
            future_to_path = {
                executor.submit(_load_single_file, fp): fp for fp in file_paths
            }

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
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[any]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"[^।॥\.\!\?]+[।॥\.\!\?]",
    )

    nodes = splitter.get_nodes_from_documents(documents)

    per_file_chunk_id: dict[str, int] = {}
    for node in nodes:
        source_path = node.metadata.get("source", "")
        filename = node.metadata.get("filename") or (
            os.path.basename(source_path) if source_path else "unknown"
        )
        node.metadata["filename"] = filename
        chunk_index = per_file_chunk_id.get(filename, 0)
        node.metadata["chunk_id"] = chunk_index
        per_file_chunk_id[filename] = chunk_index + 1

    return nodes


# ──────── Qdrant store ────────


def init_qdrant_store() -> QdrantVectorStore:
    """Initializes Qdrant collection with payload indexes for LlamaIndex."""
    client = _get_client()

    if not client.collection_exists(collection_name="test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name="test",
            field_name="filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name="test",
            field_name="metadata.filename",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection 'test' with payload index on filename")

    return QdrantVectorStore(
        client=client,
        collection_name="test",
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


def delete_by_filenames(store: QdrantVectorStore, filenames: set[str]) -> list[str]:
    """Deletes entries by filename, only reporting filenames that actually had data."""
    replaced: list[str] = []
    if not filenames:
        return replaced

    client = store.client
    for filename in sorted(filenames):
        try:
            filter_condition = Filter(
                should=[
                    FieldCondition(key="filename", match=MatchValue(value=filename)),
                    FieldCondition(
                        key="metadata.filename", match=MatchValue(value=filename)
                    ),
                ]
            )

            scroll_result = client.scroll(
                collection_name=store.collection_name,
                scroll_filter=filter_condition,
                limit=1,
            )
            if not scroll_result[0]:
                continue

            client.delete(
                collection_name=store.collection_name,
                points_selector=FilterSelector(filter=filter_condition),
            )
            replaced.append(filename)
            logger.info("Replaced existing data for '%s'", filename)
        except Exception as e:
            logger.error("Failed to delete records for %s: %s", filename, e)

    return replaced


def build_index(
    store: QdrantVectorStore, nodes: list[any]
) -> tuple[QdrantVectorStore, list[str]]:
    """Batches document node additions to Qdrant vector store."""
    filenames = {
        node.metadata.get("filename") for node in nodes if node.metadata.get("filename")
    }
    replaced = delete_by_filenames(store, filenames)

    if nodes:
        logger.info("Generating embeddings for %d text chunks...", len(nodes))
        embed_model = _get_embeddings()

        texts = [node.get_content(metadata_mode="embed") for node in nodes]
        batch_size = 64
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            embeddings = embed_model.get_text_embedding_batch(batch_texts)
            for node, emb in zip(batch_nodes, embeddings):
                node.embedding = emb
            logger.info(
                "Embedded chunks %d to %d of %d",
                i + 1,
                min(i + batch_size, len(nodes)),
                len(nodes),
            )

        logger.info("Indexing %d chunks into Qdrant vector store...", len(nodes))
        add_batch_size = 256
        for i in range(0, len(nodes), add_batch_size):
            store.add(nodes[i : i + add_batch_size])
            logger.info(
                "Indexed chunks %d to %d into Qdrant",
                i + 1,
                min(i + add_batch_size, len(nodes)),
            )

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
