from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.workflow import compile_workflow


def build_initial_state(user_input: str) -> dict:
    """构建单轮对话的初始 AgentState。"""
    return {
        "messages": [HumanMessage(content=user_input)],
        "user_id": "console_user",
        "user_segment": "普通用户",
        "intent_category": None,
        "current_order_id": None,
        "retrieved_contexts": [],
        "is_human_required": False,
        "transfer_reason": None,
    }


def get_latest_ai_message_content(result: dict) -> str:
    """从工作流执行结果中提取最后一条 AI 回复。"""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            return str(message.content)
    return "抱歉，当前没有生成有效回复。"


def print_debug_info(result: dict) -> None:
    """打印意图、转人工和 RAG 检索等调试信息。"""
    print("\n[调试信息]")
    print(f"意图分类: {result.get('intent_category')}")
    print(f"是否转人工: {result.get('is_human_required')}")

    transfer_reason = result.get("transfer_reason")
    if transfer_reason:
        print(f"转人工原因: {transfer_reason}")

    contexts = result.get("retrieved_contexts") or []
    print(f"检索片段数: {len(contexts)}")


def main() -> None:
    """控制台交互式调用智能客服工作流。"""
    app = compile_workflow()

    print("智能客服控制台已启动。")
    print("输入问题后回车即可调用智能体；输入 exit / quit / q 退出。")

    while True:
        try:
            user_input = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出智能客服控制台。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("已退出智能客服控制台。")
            break

        try:
            result = app.invoke(build_initial_state(user_input))
        except Exception as exc:
            print(f"\n系统异常：{exc}")
            continue

        print_debug_info(result)
        print("\n客服：")
        print(get_latest_ai_message_content(result))


if __name__ == "__main__":
    main()
