import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Set

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, FilterSelector

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import unicodedata
import pytesseract
from pdf2image import convert_from_path

from bookchat.config import (
    ALLOWED_SUFFIXES,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    POPPLER_PATH,
    TESSERACT_CMD,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_count: int
    chunk_count: int
    warnings: List[str] = field(default_factory=list)
    files_replaced: List[str] = field(default_factory=list)


embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"batch_size": 64},
    )



if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _ocr_page(args: tuple[int, Any]) -> Document | None:
    page_idx, image = args
    text = pytesseract.image_to_string(image, lang="kan+eng")
    if text.strip():
        return Document(
            page_content=text,
            metadata={"page": page_idx},
        )
    return None


def _ocr_pdf(file_path: str) -> List[Document]:
    logger.info("Performing parallel OCR on '%s' using Tesseract (kan+eng)...", file_path)
    kwargs = {"dpi": 150}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    images = convert_from_path(file_path, **kwargs)
    documents: List[Document] = []

    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_ocr_page, enumerate(images)))

    for result in results:
        if result:
            result.metadata["source"] = file_path
            documents.append(result)

    return documents


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


def get_chunks(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 128) -> List[Document]:
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
        filename = os.path.basename(source_path) if source_path else "unknown"
        chunk.metadata["filename"] = filename
        chunk_index = per_file_chunk_id.get(filename, 0)
        chunk.metadata["chunk_id"] = chunk_index
        per_file_chunk_id[filename] = chunk_index + 1

    return chunks


def init_qdrant_store() -> QdrantVectorStore:
    """Initializes Qdrant Client with optimized collection configs."""
    client = QdrantClient(path="./local_qdrant_db", timeout=60)
    
    if not client.collection_exists(collection_name="test"):
        client.create_collection(
            collection_name="test",
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    
    return QdrantVectorStore(
        client=client,
        collection_name="test",
        embedding=embeddings
    )


def reset_store(db_path: str = "./local_qdrant_db") -> None:
    """Removes the entire local Qdrant directory tree safely."""
    directory = Path(db_path)
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Reset local Qdrant vectorstore directory at '%s'", directory)


def _filenames_from_chunks(chunks: List[Document]) -> set[str]:
    return {
        filename
        for chunk in chunks
        if (filename := chunk.metadata.get("filename"))
    }


def _filename_exists(store: QdrantVectorStore, filename: str) -> bool:
    results = store._collection.get(where={"filename": filename}, limit=1)
    return bool(results.get("ids"))


def delete_by_filenames(store: QdrantVectorStore, filenames: Set[str]) -> List[str]:
    """Uses Qdrant's fast single-stage filtering engine to erase entries instantly."""
    replaced: List[str] = []
    if not filenames:
        return replaced

    for filename in sorted(filenames):
        # Instead of searching if it exists first, Qdrant allows a safe conditional delete sweep
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="metadata.filename",
                        match=MatchValue(value=filename)
                    )
                ]
            )
            store.client.delete(
                collection_name=store.collection_name,
                points_selector=FilterSelector(filter=filter_condition)
            )
            replaced.append(filename)
            logger.info("Executed flush sweep for tracking context: '%s'", filename)
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
    """Refactored main orchestrator using low-latency components."""
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
