from bookchat.core.ingestion import ingest, load_store
from bookchat.core.generate import get_rag_chain

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="BookChat")


