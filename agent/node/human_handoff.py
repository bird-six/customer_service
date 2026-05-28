from __future__ import annotations

from agent.state import AgentState


def human_handoff(state: AgentState) -> dict:
    """
    需要人工介入时，记录原因并结束流程。
    """
    intent = state.get("intent_category")
    return {
        "is_human_required": True,
        "transfer_reason": f"当前意图 `{intent}` 需要人工客服介入。",
    }
