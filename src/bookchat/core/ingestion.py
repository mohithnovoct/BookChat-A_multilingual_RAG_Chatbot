import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
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


from concurrent.futures import ThreadPoolExecutor
from typing import Any

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
    elif target.is_dir():
        for root, _, files in os.walk(target):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in ALLOWED_SUFFIXES:
                    try:
                        docs = _load_single_file(str(file_path))
                        documents.extend(docs)
                    except Exception as exc:
                        message = f"Failed to load '{file_path}': {exc}"
                        logger.warning(message)
                        warnings.append(message)
    else:
        raise ValueError(f"Invalid path type: {path}")

    return documents, warnings


def get_chunks(
    documents: List[Document],
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    embedding_model: str = EMBEDDING_MODEL,
) -> List[Document]:
   
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"batch_size": 64},
    )

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
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


def load_store(
    persist_dir: Path | str | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    directory = str(persist_dir or CHROMA_DIR)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"batch_size": 64},
    )
    return Chroma(persist_directory=directory, embedding_function=embeddings)


def reset_store(persist_dir: Path | str | None = None) -> None:
    directory = Path(persist_dir or CHROMA_DIR)
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Reset vectorstore at '%s'", directory)


def _filenames_from_chunks(chunks: List[Document]) -> set[str]:
    return {
        filename
        for chunk in chunks
        if (filename := chunk.metadata.get("filename"))
    }


def _filename_exists(store: Chroma, filename: str) -> bool:
    results = store._collection.get(where={"filename": filename}, limit=1)
    return bool(results.get("ids"))


def delete_by_filenames(store: Chroma, filenames: set[str]) -> list[str]:
    replaced: list[str] = []
    for filename in sorted(filenames):
        if _filename_exists(store, filename):
            store.delete(where={"filename": filename})
            replaced.append(filename)
            logger.info("Removed existing chunks for '%s'", filename)
    return replaced


def build_index(store: Chroma, chunks: List[Document]) -> tuple[Chroma, list[str]]:
    filenames = _filenames_from_chunks(chunks)
    replaced = delete_by_filenames(store, filenames)

    if chunks:
        store.add_documents(chunks)

    return store, replaced


def ingest(
    docs_path: str, persist_dir: Path | str | None = None
) -> IngestResult:
    store = load_store(persist_dir=persist_dir)

    docs, warnings = load_documents(docs_path)
    chunks = get_chunks(docs)

    _, replaced = build_index(store, chunks)
    logger.info(
        "Ingested %d documents into %d chunks", len(docs), len(chunks)
    )

    return IngestResult(
        document_count=len(docs),
        chunk_count=len(chunks),
        warnings=warnings,
        files_replaced=replaced,
    )
