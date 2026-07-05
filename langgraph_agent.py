"""
langgraph_agent.py

The same privacy-assessment agent as agent.py, rebuilt on LangGraph
instead of a hand-rolled tool-calling loop.

*** ENVIRONMENT NOTE — read before trusting this file ***
This module requires the `langgraph` package (`pip install langgraph`),
which could not be installed in the sandboxed environment used to build
this repo (no PyPI access). Everything below EXCEPT `build_graph()` is
plain Python operating on a state dict, with no LangGraph import, and IS
fully unit-tested without the package installed — see
test_langgraph_agent.py. `build_graph()` is the one function that
constructs and compiles an actual `StateGraph`, and it could not be
executed before this was delivered. Please run:

    pip install langgraph
    python -m pytest test_langgraph_agent.py -v
    python langgraph_agent.py

yourself to confirm the graph compiles and runs end-to-end. If the
LangGraph API has shifted since this was written, `build_graph()` is the
only function that should need updating — the node functions and routing
logic underneath it are framework-independent.

Reuses agent.py's TOOL_SCHEMAS, dispatch_tool, and
build_draft_from_submission rather than redefining the same tool
contracts a third time (mcp_server.py already reuses them once) — one
source of truth for what each tool does, regardless of which of the three
runners (hand-rolled loop, LangGraph, MCP) is calling it.
"""

from __future__ import annotations

import json
from typing import Literal, Optional, TypedDict

from agent import (
    TOOL_SCHEMAS,
    AgentContext,
    ScriptedMockAgentClient,
    ToolCallingLLMClient,
    build_draft_from_submission,
    dispatch_tool,
)
from draft_generator import GDPRConsideration, PrivacyAssessmentDraft

SYSTEM_PROMPT = """You are an agent producing a privacy assessment draft for an internal \
Trust & Privacy team to review — you are not making the final call, a human reviewer is.

You have tools to gather context and one tool to submit your final answer. Work through \
this efficiently: gather the customer's documents, check for a reusable prior assessment \
before running the expensive attacker model, and submit as soon as you have what you need. \
Do not invent facts not supported by what your tools return."""


class GraphState(TypedDict):
    messages: list[dict]         # Anthropic-format message history
    context: AgentContext        # in-process only; a distributed/checkpointed deployment
                                  # would need this to be JSON-serializable instead
    final_draft: Optional[dict]  # set once submit_privacy_assessment has been processed
    trace: list[str]             # tool call names, in order — same audit trail as agent.py


# --------------------------------------------------------------------------
# Node functions — plain Python, no LangGraph import, fully unit-testable
# --------------------------------------------------------------------------

def agent_node(state: GraphState, llm_client: ToolCallingLLMClient) -> GraphState:
    """Calls the LLM with the current message history and available tools,
    appends its response (text and/or tool_use blocks) to the history."""
    response_blocks = llm_client.send(SYSTEM_PROMPT, state["messages"], TOOL_SCHEMAS)
    new_messages = state["messages"] + [{"role": "assistant", "content": response_blocks}]
    return {**state, "messages": new_messages}


def should_continue(state: GraphState) -> Literal["tools", "finalize", "end"]:
    """Routing logic: inspects the most recent assistant message to decide
    which node runs next. This is the function LangGraph's conditional edge
    calls — expressed as plain Python so it's testable on its own."""
    last_message = state["messages"][-1]
    tool_use_blocks = [b for b in last_message["content"] if b.get("type") == "tool_use"]

    if not tool_use_blocks:
        return "end"  # model produced no tool call — nothing more this graph can do
    if any(b["name"] == "submit_privacy_assessment" for b in tool_use_blocks):
        return "finalize"
    return "tools"


def tools_node(state: GraphState) -> GraphState:
    """Executes every (non-submit) tool call from the last assistant message,
    appends the results as a user-role tool_result message."""
    last_message = state["messages"][-1]
    tool_use_blocks = [b for b in last_message["content"] if b.get("type") == "tool_use"]

    tool_results = []
    new_trace = list(state["trace"])
    for call in tool_use_blocks:
        new_trace.append(call["name"])
        result = dispatch_tool(call["name"], call["input"], state["context"])
        tool_results.append({"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(result)})

    new_messages = state["messages"] + [{"role": "user", "content": tool_results}]
    return {**state, "messages": new_messages, "trace": new_trace}


def finalize_node(state: GraphState) -> GraphState:
    """Builds the final draft from the submit_privacy_assessment call.
    risk_level is computed deterministically inside build_draft_from_submission
    — never taken from the model, same guarantee as agent.py and
    draft_generator.py, now enforced a third time on a third runner."""
    last_message = state["messages"][-1]
    submit_call = next(b for b in last_message["content"] if b["name"] == "submit_privacy_assessment")
    draft = build_draft_from_submission(submit_call["input"])

    new_trace = state["trace"] + ["submit_privacy_assessment"]
    return {**state, "final_draft": _draft_to_dict(draft), "trace": new_trace}


def _draft_to_dict(draft: PrivacyAssessmentDraft) -> dict:
    """Serializes to plain dicts/lists rather than leaving nested dataclass
    instances in state — matters if this graph is ever run with LangGraph's
    checkpointing, which needs state to be serializable."""
    return {
        "customer_ref": draft.customer_ref,
        "risk_score": draft.risk_score,
        "risk_level": draft.risk_level,
        "threat_model": draft.threat_model,
        "config_recommendation": draft.config_recommendation,
        "gdpr_considerations": [{"article": c.article, "relevance": c.relevance} for c in draft.gdpr_considerations],
    }


def _draft_from_dict(data: dict) -> PrivacyAssessmentDraft:
    return PrivacyAssessmentDraft(
        customer_ref=data["customer_ref"],
        risk_score=data["risk_score"],
        risk_level=data["risk_level"],
        threat_model=data["threat_model"],
        config_recommendation=data["config_recommendation"],
        gdpr_considerations=[GDPRConsideration(**c) for c in data["gdpr_considerations"]],
    )


# --------------------------------------------------------------------------
# The one function that actually needs `langgraph` installed
# --------------------------------------------------------------------------

def build_graph(llm_client: ToolCallingLLMClient):
    """
    Constructs and compiles the real LangGraph StateGraph out of the node
    functions above. UNTESTED in this environment — see the module
    docstring. If this raises an ImportError, `pip install langgraph`.
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("agent", lambda state: agent_node(state, llm_client))
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "finalize": "finalize",
        "end": END,
    })
    graph.add_edge("tools", "agent")  # loop back to the model after executing tools
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(
    customer_ref: str, context: AgentContext, llm_client: Optional[ToolCallingLLMClient] = None
) -> tuple[PrivacyAssessmentDraft, list[str]]:
    """Mirrors agent.py's PrivacyAssessmentAgent.run() signature, but executes
    via the compiled LangGraph graph instead of a hand-rolled while-loop."""
    compiled_graph = build_graph(llm_client or ScriptedMockAgentClient())

    initial_state: GraphState = {
        "messages": [
            {"role": "user", "content": f"Produce a privacy assessment for {customer_ref} in space {context.space_key}."}
        ],
        "context": context,
        "final_draft": None,
        "trace": [],
    }

    final_state = compiled_graph.invoke(initial_state)

    if final_state["final_draft"] is None:
        raise RuntimeError("Graph ended without producing a final draft.")

    return _draft_from_dict(final_state["final_draft"]), final_state["trace"]


def _demo() -> None:
    from confluence_client import ConfluenceClient
    from vector_store import HashingEmbedder, PriorAssessmentStore
    from attacker_model import evaluate_reidentification_risk
    from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow

    confluence_client = ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(make_confluence_page_flow(
            "Architecture Overview", SAMPLE_STORAGE_HTML, "/spaces/CUST014/pages/page-1"
        )),
    )
    context = AgentContext(
        confluence_client=confluence_client,
        assessment_store=PriorAssessmentStore(embedder=HashingEmbedder(dim=128)),
        run_attacker_model=lambda use_case: evaluate_reidentification_risk(cell_size=30, n_individuals=20, seed=0),
        space_key="CUST014",
    )
    draft, trace = run_agent(customer_ref="cust_014", context=context)
    print(f"Tool call trace: {trace}\n")
    print(draft.as_markdown())


if __name__ == "__main__":
    _demo()
