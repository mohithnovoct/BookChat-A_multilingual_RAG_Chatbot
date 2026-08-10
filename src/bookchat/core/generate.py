import os
from typing import List, Optional

from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from bookchat.core.ingestion import ingest, load_store

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = "You are an assistant. Answer the given questions using the context only."

@dataclass
class ModelParams():
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    task: str = "text-generation"
    max_new_tokens: int= 512
    temperature: float = 0.3


def model(params: ModelParams):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=params.name,
        task=params.task,
        max_new_tokens=params.max_new_tokens,
        temperature=params.temperature,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )

    return ChatHuggingFace(llm=llm_endpoint)

def format_metadata(docs: List[Document]) -> str:

    formatted_chunks = []

    for doc in docs:
        source = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page")
        page_info = f", Page {page+1}" if page is not None else ""
        header = f"[Source: {source}{page_info}]"
        formatted_chunks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted_chunks)

def retrieve(store: Chroma, query: str, k: int=4, persist_dir: str="./chroma") -> str:

    docs = store.similarity_search(query, k)
    return format_metadata(docs)


def get_rag_chain(store: Optional[Chroma]=None, system_prompt: str=DEFAULT_SYSTEM_PROMPT, k: int=4):

    if store is None:
        store = load_store()

    retriever = store.as_retriever(search_kwargs={"k": k})

    prompt_template = PromptTemplate(
        template="{system_prompt} \n\n Context:{context} \n\n Question:{query}",
        input_variables=["system_prompt", "context", "query"]
    )

    params = ModelParams()
    llm = model(params)

    chain = (
        {
            "context": retriever | RunnableLambda(format_metadata),
            "query": RunnablePassthrough(),
            "system_prompt": lambda _: system_prompt
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain


if __name__=="__main__":
    store = load_store()
    rag_chain = get_rag_chain(store)

    while True:
        try:
            query = input("User: ").strip()
            if not query:
                continue
            if query.lower() in ["q", "exit", "quit"]:
                break

            print("\nAns: ")
            print(rag_chain.invoke(query))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break