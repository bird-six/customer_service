from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    智能客服系统的全局状态通道（State Channels）
    """
    # 1. 对话历史（核心）
    messages: Annotated[List[BaseMessage], add_messages]

    # 2. 核心路由与控制凭证
    user_id: str  # 当前发起对话的用户ID
    user_segment: str  # 用户分层（如: "VIP", "普通用户"），用于Prompt差异化回复
    intent_category: Optional[str]  # 识别出的用户意图

    # 3. 业务上下文（动态提取）
    current_order_id: Optional[str]  # 抽取的当前关联订单号
    retrieved_contexts: List[str]  # RAG检索出来的本地知识库核心切片段落

    # 4. 人工协同与安全兜底
    is_human_required: bool  # 是否需要触发拦截，转交人工客服
    transfer_reason: Optional[str]  # 转人工的原因原因


