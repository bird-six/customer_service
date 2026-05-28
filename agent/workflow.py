from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from agent.node import classify_intent, generate_response, human_handoff, retrieve_context
from agent.node.classify_intent import INTENT_TO_REQUIRED_CONTEXT
from agent.state import AgentState


def route_after_classification(state: AgentState) -> str:
    """
    条件路由：根据意图决定下一跳。
    """
    if state.get("is_human_required"):
        return "human_handoff"

    intent = state.get("intent_category")
    if intent is None:
        return "classify_intent"

    return "retrieve_context" if INTENT_TO_REQUIRED_CONTEXT.get(intent, False) else "generate_response"


def build_workflow() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("human_handoff", human_handoff)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        source="classify_intent",
        path=route_after_classification,
        path_map={
            "classify_intent": "classify_intent",
            "retrieve_context": "retrieve_context",
            "generate_response": "generate_response",
            "human_handoff": "human_handoff",
        },
    )
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("human_handoff", END)
    graph.add_edge("generate_response", END)

    return graph


def compile_workflow() -> Callable:
    """
    返回可直接运行的编译后工作流。
    """
    return build_workflow().compile()
