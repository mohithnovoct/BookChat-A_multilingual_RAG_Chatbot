import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from huggingface_hub import InferenceClient
from llama_index.core import PromptTemplate, QueryBundle, VectorStoreIndex
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr

from bookchat.config import (
    FALLBACK_ANSWER,
    HYBRID_CANDIDATE_K,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    MIN_RERANK_SCORE,
    RERANKER_CANDIDATE_K,
    RERANKER_MODEL,
    require_hf_token,
)
from bookchat.core.ingestion import (
    _get_embeddings,
    collection_point_count,
    get_index_generation,
    init_qdrant_store,
    load_text_nodes_from_store,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

INDIC_CHAR_RE = re.compile(r"[\u0A00-\u0A7F\u0C80-\u0CFF]")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_ENGLISH_INSTRUCTION = "You MUST respond in English only."

DEFAULT_SYSTEM_PROMPT = """You are a RAG assistant. Answer in English only.
Use only the passages inside <context>. Context may contain English, Kannada, Punjabi, or a mixture of these languages.
If the passages do not contain the answer, reply with exactly:
Information not found in the source documents.
Do not use outside knowledge. Do not switch language.
Put the final answer only inside <answer></answer>.
Mention the source book name when answering. Structured source references are returned separately by the application."""


def build_system_prompt() -> str:
    return f"{_ENGLISH_INSTRUCTION}\n\n{DEFAULT_SYSTEM_PROMPT}"


QA_USER_TEMPLATE = """<context>
{context_str}
</context>
<query>{query_str}</query>"""

GOLDEN_RETRIEVAL_CASES = [
    {
        "id": "en",
        "query": "What is the main theme of the book?",
        "relevant_chunk_ids": ["en-theme"],
    },
    {
        "id": "kn",
        "query": "ಕಥೆಯ ಮುಖ್ಯ ವಿಷಯ ಏನು?",
        "relevant_chunk_ids": ["kn-theme"],
    },
    {
        "id": "pa",
        "query": "ਕਹਾਣੀ ਦਾ ਮੁੱਖ ਵਿਸ਼ਾ ਕੀ ਹੈ?",
        "relevant_chunk_ids": ["pa-theme"],
    },
]


@dataclass
class SourceReference:
    filename: str
    page: int | None = None
    chunk_id: int | None = None
    score: float | None = None


@dataclass
class RAGResult:
    answer: str
    sources: list[SourceReference]


@dataclass
class ModelParams:
    name: str = LLM_MODEL
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    temperature: float = 0.0


class HuggingFaceInferenceLLM(CustomLLM):
    model_name: str = LLM_MODEL
    token: str = ""
    max_new_tokens: int = LLM_MAX_NEW_TOKENS
    temperature: float = 0.0
    system_prompt: str = ""
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = InferenceClient(model=self.model_name, token=self.token)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = self._client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        response_text = res.choices[0].message.content or ""
        return CompletionResponse(text=response_text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = ""

        stream = self._client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                response += token
                yield CompletionResponse(text=response, delta=token)


_llm: HuggingFaceInferenceLLM | None = None
_llm_lock = threading.Lock()

_reranker: SentenceTransformerRerank | None = None
_reranker_lock = threading.Lock()

_bm25_lock = threading.Lock()
_bm25_cache: dict[str, Any] = {
    "count": None,
    "generation": None,
    "nodes": None,
    "retrievers": {},
}


def parse_model_answer(text: str) -> str:
    if not text or not text.strip():
        return FALLBACK_ANSWER
    match = ANSWER_TAG_RE.search(text)
    answer = match.group(1).strip() if match else text.strip()
    if not answer:
        return FALLBACK_ANSWER
    if answer.strip() == FALLBACK_ANSWER:
        return FALLBACK_ANSWER
    indic = len(INDIC_CHAR_RE.findall(answer))
    if indic / max(len(answer), 1) > 0.30:
        return FALLBACK_ANSWER
    return answer


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for item in relevant_ids if item in top)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    dcg = 0.0
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _get_llm(system_prompt: str | None = None) -> HuggingFaceInferenceLLM:
    """Returns a cached HuggingFaceInferenceLLM instance, creating it on first call."""
    prompt = system_prompt if system_prompt is not None else build_system_prompt()
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
                system_prompt=prompt,
            )
        else:
            _llm.system_prompt = prompt
        return _llm


def _get_reranker(top_n: int = 8) -> SentenceTransformerRerank:
    """Returns a cached SentenceTransformerRerank, loading the model only once."""
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            t0 = time.perf_counter()
            logger.info("Loading cross-encoder reranker '%s'...", RERANKER_MODEL)
            _reranker = SentenceTransformerRerank(
                model=RERANKER_MODEL,
                top_n=top_n,
                keep_retrieval_score=True,
            )
            logger.info(
                "Reranker loaded in %.1fs.", time.perf_counter() - t0
            )
        else:
            _reranker.top_n = top_n
        return _reranker


def _metadata_filters(query_language: str) -> MetadataFilters | None:
    if query_language in {"en", "kn", "pa"}:
        return MetadataFilters(
            filters=[MetadataFilter(key="lang", value=query_language)]
        )
    return None


def _filter_nodes_by_lang(nodes: list[TextNode], query_language: str) -> list[TextNode]:
    if query_language not in {"en", "kn", "pa"}:
        return nodes
    filtered = [node for node in nodes if node.metadata.get("lang") == query_language]
    return filtered or nodes


def _get_bm25_retriever(
    store: QdrantVectorStore,
    query_language: str,
    candidate_k: int,
):
    """Returns a cached BM25 retriever, rebuilding only when the store changes."""
    count = collection_point_count(store)
    generation = get_index_generation()
    with _bm25_lock:
        cache_hit = (
            _bm25_cache["nodes"] is not None
            and _bm25_cache["count"] == count
            and _bm25_cache["generation"] == generation
        )

        if not cache_hit:
            t0 = time.perf_counter()
            nodes = load_text_nodes_from_store(store)
            _bm25_cache["count"] = count
            _bm25_cache["generation"] = generation
            _bm25_cache["nodes"] = nodes
            _bm25_cache["retrievers"] = {}
            logger.info(
                "Loaded %d BM25 nodes from store in %.1fs.",
                len(nodes),
                time.perf_counter() - t0,
            )

        retriever_key = (query_language, candidate_k)
        if retriever_key in _bm25_cache["retrievers"]:
            logger.debug("Using cached BM25 retriever for lang=%s, k=%d.", query_language, candidate_k)
            return _bm25_cache["retrievers"][retriever_key]

        filtered_nodes = _filter_nodes_by_lang(_bm25_cache["nodes"], query_language)
        if not filtered_nodes:
            return None

        try:
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError:
            logger.warning("BM25 retriever is not installed; using dense retrieval only.")
            return None

        t0 = time.perf_counter()
        retriever = BM25Retriever.from_defaults(
            nodes=filtered_nodes,
            similarity_top_k=candidate_k,
            skip_stemming=True,
        )
        _bm25_cache["retrievers"][retriever_key] = retriever
        logger.info(
            "Built BM25 retriever (lang=%s, k=%d, %d nodes) in %.1fs.",
            query_language,
            candidate_k,
            len(filtered_nodes),
            time.perf_counter() - t0,
        )
        return retriever


def _sources_from_nodes(nodes: list[NodeWithScore]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for source_node in nodes:
        metadata = source_node.node.metadata
        source = SourceReference(
            filename=str(metadata.get("filename", "unknown")),
            page=metadata.get("page"),
            chunk_id=metadata.get("chunk_id"),
            score=source_node.score,
        )
        key = (source.filename, source.page, source.chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


def warmup_models() -> None:
    """Pre-loads expensive ML models in background so the first query is fast."""
    logger.info("Background model warmup starting...")
    t0 = time.perf_counter()
    _get_embeddings()
    _get_reranker()
    logger.info("Background model warmup finished in %.1fs.", time.perf_counter() - t0)


def get_query_engine(
    store: Optional[QdrantVectorStore] = None,
    k: int = 8,
    system_prompt: str | None = None,
    query_language: str = "auto",
):
    t_start = time.perf_counter()

    if store is None:
        store = init_qdrant_store()

    embed_model = _get_embeddings()
    index = VectorStoreIndex.from_vector_store(
        vector_store=store,
        embed_model=embed_model,
    )

    prompt = system_prompt if system_prompt is not None else build_system_prompt()
    qa_template = PromptTemplate(
        template=QA_USER_TEMPLATE,
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    llm = _get_llm(system_prompt=prompt)
    candidate_k = max(k, RERANKER_CANDIDATE_K, HYBRID_CANDIDATE_K)
    reranker = _get_reranker(top_n=k)

    lang_filters = _metadata_filters(query_language)
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=candidate_k,
        filters=lang_filters,
    )

    bm25_retriever = _get_bm25_retriever(store, query_language, candidate_k)

    if bm25_retriever is not None:
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=candidate_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            llm=llm,
        )
    else:
        retriever = vector_retriever

    logger.info("Query engine built in %.2fs.", time.perf_counter() - t_start)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker],
        text_qa_template=qa_template,
    )


def get_rag_chain(
    store: Optional[QdrantVectorStore] = None,
    system_prompt: str | None = None,
    k: int = 8,
    query_language: str = "auto",
):
    """Backwards-compatible wrapper returning a runnable with an .invoke(query) interface."""
    resolved_prompt = (
        system_prompt if system_prompt is not None else build_system_prompt()
    )
    query_engine = get_query_engine(
        store=store,
        k=k,
        system_prompt=resolved_prompt,
        query_language=query_language,
    )

    class RAGWrapper:
        def __init__(self, engine, language: str):
            self.engine = engine
            self.query_language = language

        def invoke(self, query: str) -> RAGResult:
            t0 = time.perf_counter()
            query_bundle = QueryBundle(query_str=query)

            t_ret = time.perf_counter()
            nodes = self.engine.retrieve(query_bundle)
            logger.info("Retrieval + rerank: %.2fs.", time.perf_counter() - t_ret)

            sources = _sources_from_nodes(nodes)
            max_score = max(
                (node.score for node in nodes if node.score is not None),
                default=None,
            )
            if not nodes or max_score is None or max_score < MIN_RERANK_SCORE:
                logger.info("Total query: %.2fs (below score threshold).", time.perf_counter() - t0)
                return RAGResult(answer=FALLBACK_ANSWER, sources=sources)

            t_llm = time.perf_counter()
            response = self.engine.synthesize(query_bundle, nodes=nodes)
            logger.info("LLM synthesis: %.2fs.", time.perf_counter() - t_llm)
            logger.info("Total query: %.2fs.", time.perf_counter() - t0)
            return RAGResult(answer=parse_model_answer(str(response)), sources=sources)

    return RAGWrapper(query_engine, query_language)


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
            print(rag_chain.invoke(query).answer)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break
