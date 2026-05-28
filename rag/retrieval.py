from __future__ import annotations

"""RAG 检索召回与重排模块。

功能：
- 从已持久化的向量数据库中构建向量检索器
- 使用 BM25 + 向量检索构建 EnsembleRetriever 混合检索
- EnsembleRetriever 底层使用加权 RRF（Reciprocal Rank Fusion）进行排名融合
- 使用阿里 DashScope qwen3-rerank 模型对混合召回结果进行二阶段重排

说明：
- 需要提前在 `.env` 中配置 `DASHSCOPE_API_KEY`
- 默认从本地 Chroma 向量库读取；如果后续换 Milvus / FAISS / Qdrant，只需要替换
  `build_chroma_vectorstore` 这一层
- BM25Retriever 是内存关键词检索器，需要传入与向量库同源的 documents
"""

import os
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Sequence

import dashscope
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - 兼容旧版 langchain 安装方式
    from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover - 兼容当前环境中的 langchain-classic
    from langchain_classic.retrievers import EnsembleRetriever  # type: ignore[no-redef]

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_INSTRUCT = (
    "Given a customer service query, retrieve relevant passages that answer the query."
)
DEFAULT_VECTOR_TOP_K = 20
DEFAULT_BM25_TOP_K = 20
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_C = 60


@dataclass(slots=True)
class RetrievedDocument:
    """最终检索结果。"""

    rank: int
    content: str
    metadata: dict[str, Any]
    rerank_score: float | None = None


@dataclass(slots=True)
class RetrievalResult:
    """混合召回和重排后的检索结果。"""

    query: str
    documents: list[RetrievedDocument]


def load_dashscope_api_key() -> str:
    """读取 DashScope API Key。"""
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "请在这里填写你的阿里云百炼 / 通义千问 API Key":
        raise EnvironmentError("请先在 .env 中配置 DASHSCOPE_API_KEY")
    return api_key


def build_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> DashScopeEmbeddings:
    """构建与入库阶段一致的 DashScope embedding 模型。"""
    return DashScopeEmbeddings(
        model=model_name,
        dashscope_api_key=load_dashscope_api_key(),
    )


def build_chroma_vectorstore(
    persist_directory: str | Path,
    collection_name: str = "customer_service_knowledge",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """从本地 Chroma 持久化目录加载向量数据库。"""
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(Path(persist_directory)),
        embedding_function=build_embeddings(embedding_model),
    )


def build_vector_retriever(
    vectorstore: Chroma,
    top_k: int = DEFAULT_VECTOR_TOP_K,
    search_type: str = "similarity",
) -> Any:
    """构建向量召回器。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k},
    )


def build_bm25_retriever(
    documents: Sequence[Document],
    top_k: int = DEFAULT_BM25_TOP_K,
) -> BM25Retriever:
    """构建 BM25 关键词召回器。"""
    if not documents:
        raise ValueError("构建 BM25Retriever 至少需要 1 个 Document")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    retriever = BM25Retriever.from_documents(list(documents))
    retriever.k = top_k
    return retriever


def build_ensemble_retriever(
    vector_retriever: Any,
    bm25_retriever: BM25Retriever,
    weights: Sequence[float] = (0.6, 0.4),
    rrf_c: int = DEFAULT_RRF_C,
    id_key: str | None = None,
) -> EnsembleRetriever:
    """构建混合检索器。

    LangChain 的 EnsembleRetriever 使用加权 RRF 算法融合多个召回器结果：
    score(doc) = sum(weight / (rank + c))

    Args:
        vector_retriever: 向量召回器。
        bm25_retriever: BM25 关键词召回器。
        weights: 两路召回权重，默认向量 0.6、BM25 0.4。
        rrf_c: RRF 平滑常数，越大越弱化靠前排名的优势。
        id_key: 去重字段；如果 metadata 中有稳定 chunk_id，建议传入该字段。
    """
    if len(weights) != 2:
        raise ValueError("weights 需要包含向量检索和 BM25 检索两个权重")
    if rrf_c < 0:
        raise ValueError("rrf_c 不能为负数")

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=list(weights),
        c=rrf_c,
        id_key=id_key,
    )


def rerank_with_qwen3(
    query: str,
    documents: Sequence[Document],
    top_n: int = DEFAULT_FINAL_TOP_K,
    model_name: str = DEFAULT_RERANK_MODEL,
    return_documents: bool = True,
    instruct: str = DEFAULT_RERANK_INSTRUCT,
) -> list[tuple[Document, float | None]]:
    """使用 DashScope SDK 调用 qwen3-rerank 对召回结果进行重排。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    if top_n <= 0:
        raise ValueError("top_n 必须大于 0")
    if not documents:
        return []

    candidate_docs = list(documents)
    dashscope.api_key = load_dashscope_api_key()
    response = dashscope.TextReRank.call(
        model=model_name,
        query=query,
        documents=[doc.page_content for doc in candidate_docs],
        top_n=min(top_n, len(candidate_docs)),
        return_documents=return_documents,
        instruct=instruct,
    )

    if response.status_code != HTTPStatus.OK:
        error_code = getattr(response, "code", None)
        error_message = getattr(response, "message", None)
        raise RuntimeError(f"DashScope rerank 调用失败: {error_code} {error_message}")

    output = getattr(response, "output", {}) or {}
    results = output.get("results", []) if isinstance(output, dict) else []
    reranked: list[tuple[Document, float | None]] = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(candidate_docs):
            continue
        score = item.get("relevance_score")
        reranked.append(
            (
                candidate_docs[index],
                float(score) if isinstance(score, int | float) else None,
            )
        )

    return reranked or [(doc, None) for doc in candidate_docs[:top_n]]


def hybrid_retrieve_with_rerank(
    query: str,
    ensemble_retriever: EnsembleRetriever,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    rerank_model: str = DEFAULT_RERANK_MODEL,
) -> RetrievalResult:
    """执行混合召回 + qwen3-rerank 重排。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    if final_top_k <= 0:
        raise ValueError("final_top_k 必须大于 0")

    recalled_documents = ensemble_retriever.invoke(query)
    reranked_documents = rerank_with_qwen3(
        query=query,
        documents=recalled_documents,
        top_n=final_top_k,
        model_name=rerank_model,
    )

    documents = [
        RetrievedDocument(
            rank=index,
            content=document.page_content,
            metadata=dict(document.metadata),
            rerank_score=score,
        )
        for index, (document, score) in enumerate(reranked_documents, start=1)
    ]
    return RetrievalResult(query=query, documents=documents)


def build_hybrid_retriever_from_chroma(
    persist_directory: str | Path,
    bm25_documents: Sequence[Document],
    collection_name: str = "customer_service_knowledge",
    vector_top_k: int = DEFAULT_VECTOR_TOP_K,
    bm25_top_k: int = DEFAULT_BM25_TOP_K,
    weights: Sequence[float] = (0.6, 0.4),
    rrf_c: int = DEFAULT_RRF_C,
    id_key: str | None = None,
) -> EnsembleRetriever:
    """从 Chroma 向量库和 BM25 文档集合构建混合检索器。"""
    vectorstore = build_chroma_vectorstore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    vector_retriever = build_vector_retriever(vectorstore, top_k=vector_top_k)
    bm25_retriever = build_bm25_retriever(bm25_documents, top_k=bm25_top_k)
    return build_ensemble_retriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        weights=weights,
        rrf_c=rrf_c,
        id_key=id_key,
    )
