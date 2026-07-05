"""
Tests for langgraph_agent.py.

Everything except test_build_graph_and_run_end_to_end runs with zero
dependency on the `langgraph` package — the node functions are plain
Python and are tested directly. The one test that needs the real package
uses pytest.importorskip, so it skips cleanly in an environment without
langgraph installed and actually verifies the graph compiles and runs
correctly in an environment that has it.

Run with: pytest test_langgraph_agent.py -v
"""

import pytest

from agent import AgentContext
from confluence_client import ConfluenceClient
from draft_generator import classify_risk_level
from fakes import SAMPLE_STORAGE_HTML, FakeSession, make_confluence_page_flow
from langgraph_agent import (
    _draft_from_dict,
    _draft_to_dict,
    agent_node,
    finalize_node,
    should_continue,
    tools_node,
)
from vector_store import HashingEmbedder, PriorAssessmentStore


def make_context(run_attacker_model=None):
    confluence_client = ConfluenceClient(
        base_url="https://example.atlassian.net",
        email="pipeline-bot@example.com",
        api_token="fake-token-not-real",
        session=FakeSession(make_confluence_page_flow(
            "Architecture Overview", SAMPLE_STORAGE_HTML, "/spaces/CUST014/pages/page-1"
        )),
    )
    return AgentContext(
        confluence_client=confluence_client,
        assessment_store=PriorAssessmentStore(embedder=HashingEmbedder(dim=128)),
        run_attacker_model=run_attacker_model or (lambda use_case: 0.5),
        space_key="CUST014",
    )


class FakeLLMClient:
    def __init__(self, response_blocks):
        self.response_blocks = response_blocks

    def send(self, system, messages, tools):
        return self.response_blocks


def initial_state(context):
    return {
        "messages": [{"role": "user", "content": "Produce a privacy assessment for cust_014 in space CUST014."}],
        "context": context,
        "final_draft": None,
        "trace": [],
    }


# --------------------------------------------------------------------------
# Node functions (no langgraph required)
# --------------------------------------------------------------------------

def test_agent_node_appends_assistant_message():
    context = make_context()
    llm = FakeLLMClient([{"type": "tool_use", "id": "1", "name": "search_confluence_docs", "input": {"space_key": "CUST014"}}])
    state = initial_state(context)

    new_state = agent_node(state, llm)

    assert len(new_state["messages"]) == 2
    assert new_state["messages"][-1]["role"] == "assistant"


def test_should_continue_routes_to_tools_for_regular_tool_call():
    state = {
        "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "1", "name": "search_confluence_docs", "input": {}}
        ]}]
    }
    assert should_continue(state) == "tools"


def test_should_continue_routes_to_finalize_for_submit_call():
    state = {
        "messages": [{"role": "assistant", "content": [
            {"type": "tool_use", "id": "1", "name": "submit_privacy_assessment", "input": {}}
        ]}]
    }
    assert should_continue(state) == "finalize"


def test_should_continue_routes_to_end_when_no_tool_calls():
    state = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]}
    assert should_continue(state) == "end"


def test_tools_node_executes_tool_and_records_trace():
    context = make_context()
    state = initial_state(context)
    state["messages"].append({"role": "assistant", "content": [
        {"type": "tool_use", "id": "1", "name": "search_confluence_docs", "input": {"space_key": "CUST014"}}
    ]})

    new_state = tools_node(state)

    assert new_state["trace"] == ["search_confluence_docs"]
    assert new_state["messages"][-1]["role"] == "user"
    assert new_state["messages"][-1]["content"][0]["type"] == "tool_result"


def test_finalize_node_computes_risk_level_deterministically():
    context = make_context()
    state = initial_state(context)
    state["messages"].append({"role": "assistant", "content": [
        {"type": "tool_use", "id": "1", "name": "submit_privacy_assessment", "input": {
            "customer_ref": "cust_014",
            "risk_score": 0.62,
            "threat_model": "text",
            "config_recommendation": "text",
            "gdpr_considerations": [{"article": "Art. 5", "relevance": "text"}],
            # deliberately try to smuggle a risk_level — must be ignored
            "risk_level": "low",
        }},
    ]})

    new_state = finalize_node(state)

    assert new_state["final_draft"]["risk_level"] == classify_risk_level(0.62) == "high"
    assert new_state["trace"] == ["submit_privacy_assessment"]


def test_draft_dict_roundtrip_preserves_all_fields():
    context = make_context()
    state = initial_state(context)
    state["messages"].append({"role": "assistant", "content": [
        {"type": "tool_use", "id": "1", "name": "submit_privacy_assessment", "input": {
            "customer_ref": "cust_014", "risk_score": 0.3, "threat_model": "t",
            "config_recommendation": "c", "gdpr_considerations": [{"article": "A", "relevance": "R"}],
        }},
    ]})
    new_state = finalize_node(state)

    draft = _draft_from_dict(new_state["final_draft"])
    round_tripped = _draft_to_dict(draft)
    assert round_tripped == new_state["final_draft"]


# --------------------------------------------------------------------------
# Full manual walkthrough — proves the node functions compose correctly,
# independent of whether LangGraph itself is installed
# --------------------------------------------------------------------------

def test_full_graph_walkthrough_matches_expected_trace():
    from agent import ScriptedMockAgentClient

    context = make_context(run_attacker_model=lambda use_case: 0.35)
    llm = ScriptedMockAgentClient()
    state = initial_state(context)

    for _ in range(10):
        state = agent_node(state, llm)
        route = should_continue(state)
        if route == "tools":
            state = tools_node(state)
        elif route == "finalize":
            state = finalize_node(state)
            break
        else:
            break

    assert state["trace"] == [
        "search_confluence_docs",
        "search_prior_assessments",
        "run_attacker_model",
        "submit_privacy_assessment",
    ]
    assert state["final_draft"]["risk_score"] == 0.35
    assert state["final_draft"]["risk_level"] == "medium"


# --------------------------------------------------------------------------
# The actual LangGraph wiring — skips cleanly if the package isn't installed
# --------------------------------------------------------------------------

def test_build_graph_and_run_end_to_end():
    pytest.importorskip("langgraph")
    from agent import ScriptedMockAgentClient
    from langgraph_agent import run_agent

    context = make_context(run_attacker_model=lambda use_case: 0.35)
    draft, trace = run_agent(customer_ref="cust_014", context=context, llm_client=ScriptedMockAgentClient())

    assert trace[-1] == "submit_privacy_assessment"
    assert draft.risk_level == "medium"
    assert draft.customer_ref == "cust_014"
