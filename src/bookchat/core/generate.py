import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

from huggingface_hub import InferenceClient
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr

from bookchat.config import require_hf_token
from bookchat.core.ingestion import _get_embeddings, init_qdrant_store

# Configure global application logging to show progress in standard output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "You MUST respond in English only.",
    "kn": "You MUST respond in Kannada (ಕನ್ನಡ) only.",
    "pa": "You MUST respond in Punjabi (ਪੰਜਾਬੀ) in Gurmukhi script only.",
}
_DEFAULT_LANG = "en"

DEFAULT_SYSTEM_PROMPT = """You are a strict Retrieval-Augmented Generation (RAG) assistant. 
Your ONLY task is to answer the query in the exact same language it was asked, using ONLY the provided text context.

CRITICAL RULES:
1. CONTEXT ISOLATION: Rely strictly on the facts directly mentioned in the context. Do NOT use your pre-trained external historical knowledge, assumptions, or later chronological events. If a fact is not explicitly in the context, treat it as entirely untrue.
2. CITATION REQUIREMENT: You must explicitly mention the source book name (e.g., 'ਸਾਚੀ ਸਾਖੀ') at the beginning or end of your answer. Do NOT include page numbers or chunk IDs.
3. NO HALLUCINATION / FALLBACK: If the provided context does not contain the definitive answer to the question, you must output exactly this phrase and nothing else:
   - In Punjabi: 'ਸਰੋਤ ਦਸਤਾਵੇਜ਼ਾਂ ਵਿੱਚ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।'
   - In English: 'Information not found in the source documents.'
   - In Kannada: 'ಮೂಲ ದಾಖಲೆಗಳಲ್ಲಿ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ.'
4. OUTPUT FORMATTING: Be concise. Write in complete, grammatically correct sentences. If making a list, ensure each point introduces completely unique information without repeating words or concepts from previous points."""


def build_system_prompt(lang: str = _DEFAULT_LANG) -> str:
    lang_instruction = _LANG_INSTRUCTIONS.get(lang, _LANG_INSTRUCTIONS[_DEFAULT_LANG])
    return f"{lang_instruction}\n\n{DEFAULT_SYSTEM_PROMPT}"


@dataclass
class ModelParams:
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_new_tokens: int = 1500
    temperature: float = 0.0
    repetition_penalty: float = 1.20


class HuggingFaceInferenceLLM(CustomLLM):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    token: str = ""
    max_new_tokens: int = 1500
    temperature: float = 0.0
    repetition_penalty: float = 1.20
    system_prompt: str = ""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = InferenceClient(model=self.model_name, token=self.token)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Wrap the raw string prompt into a chat structure
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = self._client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )
        # Extract text from the chat response object
        response_text = res.choices[0].message.content
        return CompletionResponse(text=response_text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = [{"role": "user", "content": prompt}]
        response = ""

        # Switch from text_generation to chat_completion for streaming
        stream = self._client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:  # Filter out empty or None tokens
                response += token
                yield CompletionResponse(text=response, delta=token)


# ──────── Cached LLM singleton ────────

_llm: HuggingFaceInferenceLLM | None = None
_llm_lock = threading.Lock()


def _get_llm() -> HuggingFaceInferenceLLM:
    """Returns a cached HuggingFaceInferenceLLM instance, creating it on first call."""
    global _llm
    with _llm_lock:
        if _llm is None:
            params = ModelParams()
            token = require_hf_token()
            _llm = HuggingFaceInferenceLLM(
                model_name=params.name,
                token=token,
                temperature=params.temperature,
                max_new_tokens=params.max_new_tokens,
                repetition_penalty=params.repetition_penalty,
            )
        return _llm


def get_query_engine(
    store: Optional[QdrantVectorStore] = None,
    k: int = 8,
    lang: str = _DEFAULT_LANG,
):
    if store is None:
        store = init_qdrant_store()

    embed_model = _get_embeddings()
    index = VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=embed_model,
    )

    system_prompt = build_system_prompt(lang)

    qa_template = PromptTemplate(
        template=(
            f"{system_prompt}\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    llm = _get_llm()

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=k,
        text_qa_template=qa_template,
    )

    return query_engine


def get_rag_chain(
    store: Optional[QdrantVectorStore] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    k: int = 8,
    lang: str = _DEFAULT_LANG,
):
    """Backwards-compatible wrapper returning a runnable with an .invoke(query) interface."""
    query_engine = get_query_engine(store=store, k=k, lang=lang)

    class RAGWrapper:
        def __init__(self, engine):
            self.engine = engine

        def invoke(self, query: str) -> str:
            response = self.engine.query(query)
            return str(response)

    return RAGWrapper(query_engine)


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
