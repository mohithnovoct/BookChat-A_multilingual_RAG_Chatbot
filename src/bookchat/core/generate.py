import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from bookchat.config import require_hf_token
from bookchat.core.ingestion import load_store

DEFAULT_SYSTEM_PROMPT = """You are a helpful multilingual assistant with support for Kannada (ಕನ್ನಡ) and English. Answer the user's question accurately using only the provided context. Answer in the same language as the user's question (e.g. reply in Kannada if the question is in Kannada). Every answer must explicitly name the source book where the information was found. If the context does not contain the answer, say 'Information not found in the source documents.' (or in Kannada: 'ಮೂಲ ದಾಖಲೆಗಳಲ್ಲಿ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ.')."""


@dataclass
class ModelParams:
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    task: str = "text-generation"
    max_new_tokens: int = 800
    temperature: float = 0.5


def model(params: ModelParams) -> ChatHuggingFace:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=params.name,
        task=params.task,
        max_new_tokens=params.max_new_tokens,
        temperature=params.temperature,
        huggingfacehub_api_token=require_hf_token(),
    )

    return ChatHuggingFace(llm=llm_endpoint)


def format_metadata(docs: List[Document]) -> str:
    formatted_chunks = []

    for doc in docs:
        source = doc.metadata.get("filename") or os.path.basename(
            doc.metadata.get("source", "Unknown")
        )
        page = doc.metadata.get("page")
        page_info = f", Page {page + 1}" if page is not None else ""
        header = f"[Source: {source}{page_info}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)


def get_rag_chain(
    store: Optional[Chroma] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    k: int = 4,
):
    if store is None:
        store = load_store()

    retriever = store.as_retriever(search_kwargs={"k": k})

    prompt_template = PromptTemplate(
        template="{system_prompt} \n\n Context:{context} \n\n Question:{query}",
        input_variables=["system_prompt", "context", "query"],
    )

    params = ModelParams()
    llm = model(params)

    chain = (
        {
            "context": retriever | RunnableLambda(format_metadata),
            "query": RunnablePassthrough(),
            "system_prompt": lambda _: system_prompt,
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    store = load_store()
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
