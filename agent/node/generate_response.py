from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.state import AgentState
from rag.generation import DEFAULT_FALLBACK_ANSWER, DEFAULT_SYSTEM_PROMPT, build_user_prompt, call_deepseek_chat_model


def get_latest_user_query(state: AgentState) -> str:
    """从对话历史中获取最新用户消息。"""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def build_context_text(contexts: list[str]) -> str:
    """将 RAG 检索片段拼接为 generation 模块可使用的上下文文本。"""
    return "\n\n".join(context.strip() for context in contexts if context.strip())


def generate_response(state: AgentState) -> dict:
    """
    生成客服回复。

    - chit_chat：调用 DeepSeek 生成简短客服式回复
    - product_consultation：基于 RAG 检索片段调用 DeepSeek 生成知识库增强回复
    - order_consultation / 转人工：输出明确的客服引导话术
    """
    intent = state.get("intent_category") or "unknown"
    contexts = state.get("retrieved_contexts", [])
    query = get_latest_user_query(state)

    if state.get("is_human_required"):
        transfer_reason = state.get("transfer_reason") or "当前问题需要人工客服进一步处理。"
        answer = f"抱歉，{transfer_reason} 我将为您转接人工客服继续处理。"
        return {
            "messages": [AIMessage(content=answer)],
        }

    if intent == "order_consultation":
        order_id = state.get("current_order_id")
        answer = "订单相关问题通常需要结合订单号、手机号或物流信息核实。"
        if order_id:
            answer += f"当前识别到的订单号是：{order_id}。"
        answer += "请您补充订单信息，我会继续帮您处理；如涉及支付、退款或隐私信息，也可以转人工客服。"
        return {
            "is_human_required": False,
            "transfer_reason": None,
            "messages": [AIMessage(content=answer)],
        }

    if intent == "chit_chat":
        user_prompt = query or "请用智能客服身份进行一句简短友好的欢迎和能力说明。"
        answer, _ = call_deepseek_chat_model(
            system_prompt=(
                "你是一个智能客服助手。请用简短、友好、专业的语气回复用户。"
                "不要闲聊发散，重点引导用户咨询产品、售后、订单、退款等客服问题。"
            ),
            user_prompt=user_prompt,
        )
        if not answer:
            answer = (
                "您好，我可以帮您处理产品咨询、售后政策、使用方法、订单和退款等问题。"
                "请直接告诉我您想了解的内容。"
            )
        return {
            "is_human_required": False,
            "transfer_reason": None,
            "messages": [AIMessage(content=answer)],
        }

    if contexts:
        context_text = build_context_text(contexts)
        user_prompt = build_user_prompt(
            query=query or "用户咨询产品相关问题。",
            context=context_text,
            extra_instructions="请不要原样罗列检索片段，要整合为自然、简洁、可执行的客服回复。",
        )
        answer, _ = call_deepseek_chat_model(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        if not answer:
            answer = DEFAULT_FALLBACK_ANSWER
    else:
        answer = DEFAULT_FALLBACK_ANSWER

    return {
        "is_human_required": False,
        "transfer_reason": None,
        "messages": [AIMessage(content=answer)],
    }
