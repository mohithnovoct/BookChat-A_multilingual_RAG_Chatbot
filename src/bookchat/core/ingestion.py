import os
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



def _load_single_file(file_path: str) -> List[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format {ext}. Supported file formats: .pdf, .txt, .md")

    return loader.load()


def load_documents(path: str) -> List[Document]:

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    documents: List[Document] = []

    if target.is_file():
        documents.extend(_load_single_file(str(target)))
    elif target.is_dir():
        supported_extensions = {".pdf", ".txt", ".md"}
        for root, _, files in os.walk(target):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in supported_extensions:
                    try:
                        docs = _load_single_file(str(file_path))
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Warning: Failed to load '{file_path}': {e}")
    else:
        raise ValueError(f"Invalid path type: {path}")
    return documents

def load_document(path: str) -> List[Document]:
    return load_documents()


def get_chunks(documents: List[Document], chunk_size: int=1500, chunk_overlap: int=200) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        source_path = chunk.metadata.get("source", "")
        if source_path:
            chunk.metadata["filename"] = os.path.basename(source_path)

    return chunks

def load_store(persist_dir: str="./chroma", embedding_model: str="sentence-transformers/all-MiniLM-L6-v2"):

    embeddings = HuggingFaceEmbeddings(model=embedding_model)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

def reset_store(persist_dir: str="./chroma"):

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Successfully reset vectorstore at '{persist_dir}'.")

def build_index(store: Chroma, chunks: List[Document]):

    if chunks:
        store.add_documents(chunks)
    return store

def ingest(docs_path: str, persist_dir: str="./chroma"):

    store = load_store()

    docs = load_documents(docs_path)
    chunks = get_chunks(docs)

    build_index(chunks)
    print(f"Ingested {len(docs)} documents into  {len(chunks)} chunks successfully.")
    return store

if __name__=="__main__":
    pass