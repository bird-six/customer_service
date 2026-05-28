from __future__ import annotations

"""RAG 对外封装接口。

本模块将 `retrieval.py` 中的向量召回、BM25 召回、EnsembleRetriever 融合召回、
qwen3-rerank 重排封装为智能体节点可直接调用的函数。
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from rag.retrieval import (
        DEFAULT_BM25_TOP_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_RRF_C,
        DEFAULT_VECTOR_TOP_K,
        RetrievalResult,
        build_bm25_retriever,
        build_chroma_vectorstore,
        build_ensemble_retriever,
        build_vector_retriever,
        hybrid_retrieve_with_rerank,
    )
except ImportError:  # pragma: no cover - 兼容直接在 rag 目录内运行脚本
    from retrieval import (  # type: ignore[no-redef]
        DEFAULT_BM25_TOP_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_RRF_C,
        DEFAULT_VECTOR_TOP_K,
        RetrievalResult,
        build_bm25_retriever,
        build_chroma_vectorstore,
        build_ensemble_retriever,
        build_vector_retriever,
        hybrid_retrieve_with_rerank,
    )

DEFAULT_COLLECTION_NAME = "customer_service_knowledge"
DEFAULT_PERSIST_DIRECTORY = "chroma_db"
DEFAULT_CONTEXT_MAX_CHARS = 1200


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """RAG 检索配置。"""

    persist_directory: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    vector_top_k: int = DEFAULT_VECTOR_TOP_K
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    final_top_k: int = DEFAULT_FINAL_TOP_K
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_c: int = DEFAULT_RRF_C
    id_key: str | None = "chunk_id"


@dataclass(slots=True)
class RAGContext:
    """智能体节点可直接使用的 RAG 上下文。"""

    contexts: list[str]
    retrieval_result: RetrievalResult | None = None
    error: str | None = None


def load_rag_config() -> RAGConfig:
    """从环境变量读取 RAG 配置。"""
    load_dotenv()
    persist_directory = os.getenv("RAG_CHROMA_DIR", DEFAULT_PERSIST_DIRECTORY)
    collection_name = os.getenv("RAG_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

    return RAGConfig(
        persist_directory=persist_directory,
        collection_name=collection_name,
        vector_top_k=int(os.getenv("RAG_VECTOR_TOP_K", str(DEFAULT_VECTOR_TOP_K))),
        bm25_top_k=int(os.getenv("RAG_BM25_TOP_K", str(DEFAULT_BM25_TOP_K))),
        final_top_k=int(os.getenv("RAG_FINAL_TOP_K", str(DEFAULT_FINAL_TOP_K))),
        vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", "0.6")),
        bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", "0.4")),
        rrf_c=int(os.getenv("RAG_RRF_C", str(DEFAULT_RRF_C))),
        id_key=os.getenv("RAG_ID_KEY", "chunk_id") or None,
    )


def _normalize_persist_directory(persist_directory: str) -> str:
    """将相对路径转换为当前工作目录下的绝对路径字符串。"""
    path = Path(persist_directory).expanduser()
    return str(path if path.is_absolute() else Path.cwd() / path)


def load_documents_from_chroma(vectorstore: Any) -> list[Document]:
    """从 Chroma 中读取全部文档，用于构建 BM25 关键词召回器。"""
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise RuntimeError("当前向量库对象不支持直接读取 Chroma collection")

    data = collection.get(include=["documents", "metadatas"])
    texts = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    documents: list[Document] = []
    for index, text in enumerate(texts):
        if not text:
            continue
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        documents.append(Document(page_content=str(text), metadata=dict(metadata)))

    if not documents:
        raise RuntimeError("Chroma 向量库中没有可用于 RAG 检索的文档，请先完成知识库入库")
    return documents


@lru_cache(maxsize=1)
def build_rag_retriever():
    """构建并缓存智能客服 RAG 混合检索器。"""
    config = load_rag_config()
    vectorstore = build_chroma_vectorstore(
        persist_directory=_normalize_persist_directory(config.persist_directory),
        collection_name=config.collection_name,
    )
    bm25_documents = load_documents_from_chroma(vectorstore)
    vector_retriever = build_vector_retriever(vectorstore, top_k=config.vector_top_k)
    bm25_retriever = build_bm25_retriever(bm25_documents, top_k=config.bm25_top_k)

    return build_ensemble_retriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        weights=(config.vector_weight, config.bm25_weight),
        rrf_c=config.rrf_c,
        id_key=config.id_key,
    )


def format_retrieved_contexts(
    retrieval_result: RetrievalResult,
    max_chars_per_context: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> list[str]:
    """将 RetrievalResult 转换为 AgentState.retrieved_contexts 需要的字符串列表。"""
    contexts: list[str] = []
    for document in retrieval_result.documents:
        content = document.content.strip()
        if not content:
            continue

        source = document.metadata.get("file_name") or document.metadata.get("source") or "未知来源"
        chunk_id = document.metadata.get("chunk_id", "未知")
        score = (
            f"{document.rerank_score:.4f}" if document.rerank_score is not None else "无"
        )
        if len(content) > max_chars_per_context:
            content = content[:max_chars_per_context].rstrip() + "..."

        contexts.append(
            f"[知识片段{document.rank}] 来源：{source}；chunk_id：{chunk_id}；相关性：{score}\n{content}"
        )
    return contexts


def retrieve_rag_context(query: str) -> RAGContext:
    """执行完整 RAG 检索召回并返回智能体可用上下文。"""
    cleaned_query = query.strip()
    if not cleaned_query:
        return RAGContext(contexts=[])

    try:
        retriever = build_rag_retriever()
        config = load_rag_config()
        retrieval_result = hybrid_retrieve_with_rerank(
            query=cleaned_query,
            ensemble_retriever=retriever,
            final_top_k=config.final_top_k,
        )
        return RAGContext(
            contexts=format_retrieved_contexts(retrieval_result),
            retrieval_result=retrieval_result,
        )
    except Exception as exc:
        return RAGContext(contexts=[], error=str(exc))
