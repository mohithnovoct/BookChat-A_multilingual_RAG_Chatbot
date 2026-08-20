import os
import threading
from dataclasses import dataclass
from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from bookchat.config import require_hf_token
from bookchat.core.ingestion import init_qdrant_store


_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "You MUST respond in English only.",
    "kn": "You MUST respond in Kannada (ಕನ್ನಡ) only.",
    "pa": "You MUST respond in Punjabi (ਪੰਜਾਬੀ) in Gurmukhi script only.",
}
_DEFAULT_LANG = "en"

DEFAULT_SYSTEM_PROMPT = """You are a helpful multilingual assistant with support for Kannada (ಕನ್ನಡ), Punjabi (ਪੰਜਾਬੀ), and English. Answer the user's question accurately using only the provided context. Every answer must explicitly name the source book where the information was found. If the context does not contain the answer, say 'Information not found in the source documents.' (in Kannada: 'ಮೂಲ ದಾಖಲೆಗಳಲ್ಲಿ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ.', in Punjabi: 'ਸਰੋਤ ਦਸਤਾਵੇਜ਼ਾਂ ਵਿੱਚ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।')."""


def build_system_prompt(lang: str = _DEFAULT_LANG) -> str:
    lang_instruction = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS[_DEFAULT_LANG])
    return f"{lang_instruction}\n\n{DEFAULT_SYSTEM_PROMPT}"


@dataclass
class ModelParams:
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    task: str = "text-generation"
    max_new_tokens: int = 800
    temperature: float = 0.3
    repetition_penalty: float = 1.15


# ──────── Cached LLM singleton ────────

_llm: ChatHuggingFace | None = None
_llm_lock = threading.Lock()


def _get_llm() -> ChatHuggingFace:
    """Returns a cached ChatHuggingFace instance, creating it on first call."""
    global _llm
    with _llm_lock:
        if _llm is None:
            params = ModelParams()
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=params.name,
                task=params.task,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature,
                huggingfacehub_api_token=require_hf_token(),
                repetition_penalty=params.repetition_penalty,
            )
            _llm = ChatHuggingFace(llm=llm_endpoint)
        return _llm


def format_metadata(docs: List[Document]) -> str:
    formatted_chunks = []

    for doc in docs:
        source = doc.metadata.get("filename") or os.path.basename(
            doc.metadata.get("source", "Unknown")
        )
        header = f"[Source: {source}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def get_rag_chain(
    store: Optional[QdrantVectorStore] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    k: int = 4,
    lang: str = _DEFAULT_LANG,
):
    if store is None:
        store = init_qdrant_store()

    # Use MMR to retrieve diverse chunks instead of near-duplicates
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7},
    )

    # Use ChatPromptTemplate so ChatHuggingFace can apply the Llama chat template
    # with proper system/human message roles
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])

    llm = _get_llm()

    resolved_prompt = build_system_prompt(lang)

    chain = (
        {
            "context": retriever | RunnableLambda(format_metadata),
            "query": RunnablePassthrough(),
            "system_prompt": lambda _: resolved_prompt,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    store = init_qdrant_store()
    rag_chain = get_rag_chain(store)

    while True:
        try:
            query = input("User: ").strip()
            if not query:
                continue
            if query.lower() in {"q", "exit", "quit"}:
                break

            print("\nAns: ")
            print(rag_chain.invoke(query))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break
