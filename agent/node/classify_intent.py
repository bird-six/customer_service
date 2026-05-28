from __future__ import annotations

from dotenv import load_dotenv
import json
import os
from functools import lru_cache
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState

load_dotenv()
IntentCategory = Literal["chit_chat", "product_consultation", "order_consultation"]


class IntentClassification(TypedDict):
    intent_category: IntentCategory


INTENT_TO_REQUIRED_CONTEXT = {
    "chit_chat": False,
    "product_consultation": True,
    "order_consultation": False,
}


INTENT_CLASSIFICATION_SYSTEM_PROMPT = """你是智能客服系统的意图分类器。
请根据用户最新一条消息，将意图严格分类为以下三类之一：
1. chit_chat：普通聊天，但只能是与产品、订单、客服服务相关的轻量寒暄或确认，不允许进行产品无关的闲聊
2. product_consultation：产品咨询相关，需要触发 RAG 检索，包括但不限于产品功能、规格、价格、使用方法、售后政策、活动规则等
3. order_consultation：订单咨询相关，包括但不限于订单查询、退款、支付、订单状态等

分类要求：
- 只输出 JSON
- 必须且只能输出一个字段：intent_category
- intent_category 的值只能是 chit_chat、product_consultation、order_consultation 之一
- 不要输出任何解释、额外字段或多余文本
"""


@lru_cache(maxsize=1)
def build_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")

    return init_chat_model(
        model=model,
        model_provider="deepseek",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def parse_intent_response(content: str) -> IntentCategory:
    """解析模型返回的 JSON 意图分类结果。"""
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return "chit_chat"

    intent = data.get("intent_category", "chit_chat")
    return intent if intent in INTENT_TO_REQUIRED_CONTEXT else "chit_chat"


def classify_intent(state: AgentState) -> dict:
    """
    1. 意图分类节点
    使用 LLM 对最新一轮用户消息进行意图分类。

    注意：这里不使用 with_structured_output，避免部分 DeepSeek thinking 模型
    对 tool_choice / function calling 的兼容问题。
    """
    last_message = state["messages"][-1] if state.get("messages") else None
    if not isinstance(last_message, HumanMessage):
        return {"intent_category": "chit_chat"}

    response = build_llm().invoke(
        [
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=last_message.content),
        ]
    )
    intent = parse_intent_response(str(response.content))

    return {"intent_category": intent}


