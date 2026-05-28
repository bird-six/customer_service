from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.state import AgentState
from rag.pipeline import retrieve_rag_context


def get_latest_user_query(state: AgentState) -> str:
    """从对话历史中获取最新用户消息。"""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def retrieve_context(state: AgentState) -> dict:
    """
    2. RAG 检索节点。

    product_consultation 意图触发本地知识库 RAG 检索：
    - Chroma 向量召回
    - BM25 关键词召回
    - EnsembleRetriever 加权 RRF 融合
    - qwen3-rerank 二阶段重排
    """
    intent = state.get("intent_category")
    if intent != "product_consultation":
        return {"retrieved_contexts": []}

    query = get_latest_user_query(state)
    if not query:
        return {
            "retrieved_contexts": [],
            "is_human_required": True,
            "transfer_reason": "未获取到用户问题，无法执行 RAG 检索。",
        }

    rag_context = retrieve_rag_context(query)
    if rag_context.error:
        return {
            "retrieved_contexts": [],
            "is_human_required": True,
            "transfer_reason": f"RAG 检索失败：{rag_context.error}",
        }

    if not rag_context.contexts:
        return {
            "retrieved_contexts": [],
            "is_human_required": True,
            "transfer_reason": "知识库未检索到可用上下文，建议转人工客服。",
        }

    user_segment = state.get("user_segment", "普通用户")
    contexts = [f"用户分层：{user_segment}。", *rag_context.contexts]
    return {
        "retrieved_contexts": contexts,
        "is_human_required": False,
        "transfer_reason": None,
    }
