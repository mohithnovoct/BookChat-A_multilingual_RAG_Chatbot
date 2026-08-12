import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bookchat.config import ALLOWED_SUFFIXES, CHROMA_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_count: int
    chunk_count: int
    warnings: List[str] = field(default_factory=list)
    files_replaced: List[str] = field(default_factory=list)


def _load_single_file(file_path: str) -> List[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in {".txt", ".md"}:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file format {ext}. Supported file formats: .pdf, .txt, .md"
        )

    return loader.load()


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
    documents: List[Document], chunk_size: int = 1500, chunk_overlap: int = 200
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
    embeddings = HuggingFaceEmbeddings(model=embedding_model)
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
