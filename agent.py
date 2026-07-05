"""
agent.py

A real agentic loop: the LLM is given a set of tools and decides for
itself which to call, in what order, and when it has enough information
to submit a final answer. This is the distinction that matters — everything
in pipeline.py is an LLM-orchestrated *workflow* (the code decides the call
order); this file is an actual *agent* (the model decides).

Implemented directly against Anthropic's tool-use API rather than through
LangGraph/LangChain, since a from-scratch implementation is a stronger
demonstration of understanding what those frameworks are doing underneath
— a graph engine's tool-calling node is this same loop with more
scaffolding around it.

Tool design notes:
  - `submit_privacy_assessment` doubles as the "final answer" tool. When
    the model calls it, the loop ends and its arguments become the draft.
  - Its schema deliberately does NOT accept a risk_level field — only a
    risk_score. This is a stronger version of the same guarantee
    draft_generator.py enforces at the code level: instead of stripping an
    unwanted field after the fact, the tool schema never gives the model
    the option to supply one in the first place. risk_level is always
    computed by `classify_risk_level` (see draft_generator.py).
  - `run_attacker_model` is flagged in its description as expensive, and
    the system prompt instructs the agent to check for a reusable prior
    assessment first — but nothing *forces* that order. Proving the agent
    actually behaves that way is what the tests in test_agent.py are for.

Everything in this file operates on synthetic/mocked data in its own
demo and tests. No real customer data is used or represented.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import requests

from confluence_client import ConfluenceClient
from draft_generator import GDPRConsideration, PrivacyAssessmentDraft, classify_risk_level
from vector_store import PriorAssessmentStore

MAX_AGENT_ITERATIONS = 6


class AgentError(RuntimeError):
    """Raised when the agent loop fails to reach a final submission."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool call fails during execution."""


# --------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# --------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_confluence_docs",
        "description": (
            "Fetch a customer's architecture docs, data specs, and use-case description "
            "from their dedicated internal Confluence space. Call this first — it's the "
            "only way to learn what the customer is actually requesting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"space_key": {"type": "string", "description": "The customer's Confluence space key."}},
            "required": ["space_key"],
        },
    },
    {
        "name": "search_prior_assessments",
        "description": (
            "Search past privacy assessments for one similar to the current use case. "
            "ALWAYS call this before run_attacker_model — if a sufficiently similar prior "
            "assessment exists (similarity above ~0.7), reuse its risk_score instead of "
            "paying the cost of a fresh attacker-model evaluation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_case_summary": {"type": "string", "description": "Plain-text summary of the customer's use case."}
            },
            "required": ["use_case_summary"],
        },
    },
    {
        "name": "run_attacker_model",
        "description": (
            "Run a fresh re-identification risk evaluation against synthetic population data. "
            "This is computationally expensive (trains a classifier) — only call this if "
            "search_prior_assessments did not return a sufficiently similar match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_case_summary": {"type": "string", "description": "Plain-text summary of the customer's use case."}
            },
            "required": ["use_case_summary"],
        },
    },
    {
        "name": "submit_privacy_assessment",
        "description": (
            "Submit the final privacy assessment draft. This ends the task — only call this "
            "once you have gathered customer context AND have a risk_score, either reused "
            "from a prior assessment or freshly computed. Note: risk_level is NOT an argument "
            "here — it is always computed deterministically from risk_score, not decided by you."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_ref": {"type": "string"},
                "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                "threat_model": {"type": "string", "description": "Plain-language description of the re-identification threat."},
                "config_recommendation": {"type": "string", "description": "Recommended anonymization configuration change, if any."},
                "gdpr_considerations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"article": {"type": "string"}, "relevance": {"type": "string"}},
                        "required": ["article", "relevance"],
                    },
                },
            },
            "required": ["customer_ref", "risk_score", "threat_model", "config_recommendation", "gdpr_considerations"],
        },
    },
]

SYSTEM_PROMPT = """You are an agent producing a privacy assessment draft for an internal \
Trust & Privacy team to review — you are not making the final call, a human reviewer is.

You have tools to gather context and one tool to submit your final answer. Work through \
this efficiently: gather the customer's documents, check for a reusable prior assessment \
before running the expensive attacker model, and submit as soon as you have what you need. \
Do not invent facts not supported by what your tools return."""


# --------------------------------------------------------------------------
# Tool dispatch — bridges tool calls to the real modules
# --------------------------------------------------------------------------

@dataclass
class AgentContext:
    confluence_client: ConfluenceClient
    assessment_store: PriorAssessmentStore
    run_attacker_model: Callable[[str], float]  # injected, same reasoning as pipeline.py
    space_key: str


def dispatch_tool(name: str, tool_input: dict, context: AgentContext) -> dict:
    if name == "search_confluence_docs":
        docs = context.confluence_client.get_customer_docs(space_key=tool_input["space_key"])
        return {"docs": [{"title": d.title, "content": d.content, "url": d.url} for d in docs]}

    if name == "search_prior_assessments":
        results = context.assessment_store.search(tool_input["use_case_summary"], top_k=3)
        return {
            "matches": [
                {
                    "customer_ref": a.customer_ref,
                    "use_case_summary": a.use_case_summary,
                    "risk_level": a.risk_level,
                    "risk_score": a.risk_score,
                    "similarity": round(score, 3),
                }
                for a, score in results
            ]
        }

    if name == "run_attacker_model":
        risk_score = context.run_attacker_model(tool_input["use_case_summary"])
        return {"risk_score": risk_score}

    raise ToolExecutionError(f"Unknown or non-dispatchable tool: {name}")


def build_draft_from_submission(tool_input: dict) -> PrivacyAssessmentDraft:
    """Construct the final draft from the agent's submit_privacy_assessment call.
    risk_level is computed here — deterministically — never taken from the model."""
    risk_score = tool_input["risk_score"]
    return PrivacyAssessmentDraft(
        customer_ref=tool_input["customer_ref"],
        risk_score=risk_score,
        risk_level=classify_risk_level(risk_score),
        threat_model=tool_input["threat_model"],
        config_recommendation=tool_input["config_recommendation"],
        gdpr_considerations=[
            GDPRConsideration(article=c["article"], relevance=c["relevance"])
            for c in tool_input["gdpr_considerations"]
        ],
    )


# --------------------------------------------------------------------------
# LLM backend (tool-calling)
# --------------------------------------------------------------------------

class ToolCallingLLMClient(Protocol):
    def send(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        """Returns a list of Anthropic-format content blocks (text and/or tool_use)."""
        ...


class AnthropicToolCallingClient:
    """Direct integration with Anthropic's tool-use API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def send(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": self.model, "max_tokens": 1500, "system": system, "messages": messages, "tools": tools},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["content"]


class ScriptedMockAgentClient:
    """
    Deterministic stand-in for a real tool-calling LLM, for offline demos
    and tests. Decides its next move by inspecting the conversation history
    so far — specifically, the tool results it has already seen — the same
    way a real model would condition on prior tool results. This is scripted
    (not reasoning), but it exercises the same branching logic a real agent
    would need to get right: check prior assessments before paying for a
    fresh attacker-model run, and only submit once both context and a risk
    score are in hand.
    """

    def __init__(self):
        self._call_count = 0

    def send(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        self._call_count += 1
        seen_tool_results = self._collect_tool_results(messages)

        if "search_confluence_docs" not in seen_tool_results:
            return [self._tool_use("search_confluence_docs", {"space_key": self._infer_space_key(messages)})]

        docs_result = seen_tool_results["search_confluence_docs"]
        use_case_summary = " ".join(d["content"] for d in docs_result["docs"])[:500]

        if "search_prior_assessments" not in seen_tool_results:
            return [self._tool_use("search_prior_assessments", {"use_case_summary": use_case_summary})]

        prior_matches = seen_tool_results["search_prior_assessments"]["matches"]
        has_reusable_match = any(
            m["similarity"] >= 0.70 and m["risk_score"] is not None for m in prior_matches
        )

        if not has_reusable_match and "run_attacker_model" not in seen_tool_results:
            return [self._tool_use("run_attacker_model", {"use_case_summary": use_case_summary})]

        risk_score = (
            prior_matches[0]["risk_score"]
            if has_reusable_match
            else seen_tool_results["run_attacker_model"]["risk_score"]
        )

        return [self._tool_use("submit_privacy_assessment", {
            "customer_ref": self._infer_customer_ref(messages),
            "risk_score": risk_score,
            "threat_model": "Synthesized from customer docs and quantified re-identification risk.",
            "config_recommendation": "Apply the anonymization configuration validated by the risk evaluation above.",
            "gdpr_considerations": [
                {"article": "Article 4(1)", "relevance": "Trajectory data may constitute personal data."},
            ],
        })]

    @staticmethod
    def _tool_use(name: str, tool_input: dict) -> dict:
        return {"type": "tool_use", "id": f"mock_{name}", "name": name, "input": tool_input}

    @staticmethod
    def _collect_tool_results(messages: list[dict]) -> dict[str, dict]:
        """Map tool name -> its most recent parsed result, by matching tool_use_id."""
        tool_use_names: dict[str, str] = {}
        results: dict[str, dict] = {}
        for msg in messages:
            if msg["role"] != "assistant" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if block.get("type") == "tool_use":
                    tool_use_names[block["id"]] = block["name"]
        for msg in messages:
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    name = tool_use_names.get(block["tool_use_id"])
                    if name:
                        results[name] = json.loads(block["content"])
        return results

    @staticmethod
    def _infer_space_key(messages: list[dict]) -> str:
        first_user_text = messages[0]["content"] if messages else ""
        return first_user_text.split("space ")[-1].split()[0].rstrip(".") if "space " in first_user_text else "UNKNOWN"

    @staticmethod
    def _infer_customer_ref(messages: list[dict]) -> str:
        first_user_text = messages[0]["content"] if messages else ""
        return first_user_text.split("for ")[-1].split(" in")[0] if "for " in first_user_text else "unknown"


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

class PrivacyAssessmentAgent:
    def __init__(self, llm_client: ToolCallingLLMClient, max_iterations: int = MAX_AGENT_ITERATIONS):
        self._llm = llm_client
        self._max_iterations = max_iterations

    def run(self, customer_ref: str, context: AgentContext) -> tuple[PrivacyAssessmentDraft, list[str]]:
        """Returns the final draft plus a trace of tool names called, in order — useful
        for tests and for auditing what the agent actually did."""
        messages: list[dict] = [
            {"role": "user", "content": f"Produce a privacy assessment for {customer_ref} in space {context.space_key}."}
        ]
        trace: list[str] = []

        for _ in range(self._max_iterations):
            response_blocks = self._llm.send(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
            tool_use_blocks = [b for b in response_blocks if b.get("type") == "tool_use"]

            if not tool_use_blocks:
                raise AgentError("Model returned no tool calls and no submission — cannot proceed.")

            messages.append({"role": "assistant", "content": response_blocks})

            tool_results = []
            for call in tool_use_blocks:
                trace.append(call["name"])
                if call["name"] == "submit_privacy_assessment":
                    return build_draft_from_submission(call["input"]), trace

                result = dispatch_tool(call["name"], call["input"], context)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        raise AgentError(f"Agent did not submit a final assessment within {self._max_iterations} iterations.")


def _demo() -> None:
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
    assessment_store = PriorAssessmentStore(embedder=HashingEmbedder(dim=128))  # empty -> forces attacker model

    context = AgentContext(
        confluence_client=confluence_client,
        assessment_store=assessment_store,
        run_attacker_model=lambda use_case: evaluate_reidentification_risk(cell_size=30, n_individuals=20, seed=0),
        space_key="CUST014",
    )

    agent = PrivacyAssessmentAgent(llm_client=ScriptedMockAgentClient())
    draft, trace = agent.run(customer_ref="cust_014", context=context)

    print(f"Tool call trace: {trace}\n")
    print(draft.as_markdown())


if __name__ == "__main__":
    _demo()
