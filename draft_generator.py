"""
draft_generator.py

Implements step 4 of the pipeline: "Draft generation (LLM)" — turning
gathered context (customer docs, a prior similar assessment if one was
found, and a quantified re-identification risk score) into a structured
privacy assessment draft: threat model, risk classification, a
configuration recommendation, and a GDPR mapping.

Key design decision, worth calling out explicitly:

    The risk LEVEL (low/medium/high) is computed deterministically in code
    from the attacker model's numeric score (see `classify_risk_level`),
    NOT left to the LLM to decide. The LLM writes the narrative — the
    threat model description, the recommendation, the regulatory framing
    — but a compliance document's risk tier should never depend on
    generative interpretation of a number that was already computed
    precisely upstream. This mirrors the fact-check step downstream: keep
    the LLM in the "draft assistant" role, not the "makes the call" role.

No draft produced by this module is meant to reach a customer without a
human reviewer and a separate fact-check pass (see the pipeline README).
This module only produces the draft; it does not sign off on anything.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import requests


# --------------------------------------------------------------------------
# Deterministic risk classification (NOT delegated to the LLM)
# --------------------------------------------------------------------------

RISK_THRESHOLDS = {"low": 0.15, "medium": 0.50}  # score < 0.15 -> low, < 0.50 -> medium, else high


def classify_risk_level(risk_score: float) -> str:
    """Pure, deterministic mapping from a re-identification risk score to a tier."""
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(f"risk_score must be in [0, 1], got {risk_score}")
    if risk_score < RISK_THRESHOLDS["low"]:
        return "low"
    if risk_score < RISK_THRESHOLDS["medium"]:
        return "medium"
    return "high"


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------

@dataclass
class GDPRConsideration:
    article: str        # e.g. "Article 4(1) — definition of personal data"
    relevance: str       # plain-language explanation of why it applies here


@dataclass
class PrivacyAssessmentDraft:
    customer_ref: str
    risk_score: float                 # from the attacker model, passed through untouched
    risk_level: str                   # computed by classify_risk_level, never by the LLM
    threat_model: str
    config_recommendation: str
    gdpr_considerations: list[GDPRConsideration]
    low_confidence_sections: list[str] = field(default_factory=list)  # populated by a later fact-check pass
    raw_llm_response: str = field(default="", repr=False)

    def as_markdown(self) -> str:
        gdpr_lines = "\n".join(f"- **{c.article}** — {c.relevance}" for c in self.gdpr_considerations)
        flags = (
            "\n\n**Flagged for review:**\n" + "\n".join(f"- {s}" for s in self.low_confidence_sections)
            if self.low_confidence_sections
            else ""
        )
        return (
            f"# Privacy Assessment Draft — {self.customer_ref}\n\n"
            f"**Risk level:** {self.risk_level.upper()} (score: {self.risk_score:.2f})\n\n"
            f"## Threat model\n{self.threat_model}\n\n"
            f"## Configuration recommendation\n{self.config_recommendation}\n\n"
            f"## GDPR considerations\n{gdpr_lines}"
            f"{flags}"
        )


class DraftGenerationError(RuntimeError):
    """Raised when the LLM's response can't be parsed into a valid draft."""


# --------------------------------------------------------------------------
# LLM backend
# --------------------------------------------------------------------------

class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLMClient:
    """
    Minimal direct integration with the Anthropic Messages API. In the
    production pipeline this call sits behind a LangChain wrapper (for
    prompt templating and output-parser reuse across pipeline steps) —
    this is the underlying HTTP call either way.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self._api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1500,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=30,
        )
        response.raise_for_status()
        blocks = response.json()["content"]
        return "".join(b["text"] for b in blocks if b.get("type") == "text")


class MockLLMClient:
    """
    Deterministic, zero-dependency stand-in for demos and tests. Returns a
    plausible structured draft without any network call or API key — same
    pattern as the fallback embedder in vector_store.py.
    """

    def complete(self, system: str, user: str) -> str:
        return json.dumps({
            "threat_model": (
                "The requested data-sharing use case involves trajectory data with home/work "
                "patterns that, per the attacker-model evaluation, retain measurable "
                "re-identification risk under the proposed configuration."
            ),
            "config_recommendation": (
                "Apply spatial generalization at a coarser grid resolution than currently "
                "configured, and suppress trip endpoints falling within residential-density "
                "cells during overnight hours."
            ),
            "gdpr_considerations": [
                {
                    "article": "Article 4(1) — definition of personal data",
                    "relevance": "Trajectory data with recoverable home/work locations may "
                                  "constitute personal data even after generalization.",
                },
                {
                    "article": "Article 25 — data protection by design",
                    "relevance": "The anonymization configuration should be documented as "
                                  "part of demonstrating privacy-by-design compliance.",
                },
            ],
        })


def default_llm_client() -> LLMClient:
    """Prefer a real Anthropic API client; fall back to the mock if no key is configured."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLMClient()
    return MockLLMClient()


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are drafting a privacy assessment for an internal Trust & Privacy \
team to review before it is sent to a customer. You are NOT making the final call — a human \
reviewer will accept, edit, or reject this draft.

Do not invent facts not present in the provided context. If the context is insufficient to \
support a claim, say so explicitly rather than filling the gap.

Respond with ONLY a JSON object with exactly these keys: "threat_model" (string), \
"config_recommendation" (string), "gdpr_considerations" (list of objects with "article" \
and "relevance" string fields). Do not include the risk level or risk score — those are \
computed separately and are not yours to determine. No prose outside the JSON object."""


def build_user_prompt(customer_docs: list[str], prior_assessment_summary: Optional[str], risk_score: float) -> str:
    sections = ["## Customer-provided context", *(f"\n{doc}\n" for doc in customer_docs)]

    if prior_assessment_summary:
        sections.append("\n## Most similar prior assessment (for reference, not a template to copy verbatim)")
        sections.append(prior_assessment_summary)

    sections.append(
        f"\n## Attacker-model result\nQuantified re-identification risk score: {risk_score:.3f} "
        f"(scale 0-1, computed independently of this draft)."
    )
    return "\n".join(sections)


# --------------------------------------------------------------------------
# Draft generator
# --------------------------------------------------------------------------

class DraftGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or default_llm_client()

    def generate(
        self,
        customer_ref: str,
        customer_docs: list[str],
        risk_score: float,
        prior_assessment_summary: Optional[str] = None,
    ) -> PrivacyAssessmentDraft:
        user_prompt = build_user_prompt(customer_docs, prior_assessment_summary, risk_score)
        raw_response = self._llm.complete(SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_response(raw_response)

        return PrivacyAssessmentDraft(
            customer_ref=customer_ref,
            risk_score=risk_score,
            risk_level=classify_risk_level(risk_score),  # deterministic, not from parsed
            threat_model=parsed["threat_model"],
            config_recommendation=parsed["config_recommendation"],
            gdpr_considerations=[
                GDPRConsideration(article=c["article"], relevance=c["relevance"])
                for c in parsed["gdpr_considerations"]
            ],
            raw_llm_response=raw_response,
        )

    @staticmethod
    def _parse_response(raw_response: str) -> dict:
        # Strip a ```json ... ``` fence if the model wrapped its output in one.
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE)

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise DraftGenerationError(f"LLM response was not valid JSON: {e}\n---\n{raw_response}") from e

        required_keys = {"threat_model", "config_recommendation", "gdpr_considerations"}
        missing = required_keys - parsed.keys()
        if missing:
            raise DraftGenerationError(f"LLM response missing required keys: {missing}\n---\n{raw_response}")

        return parsed


def _demo() -> None:
    generator = DraftGenerator()  # uses MockLLMClient unless ANTHROPIC_API_KEY is set
    print(f"LLM backend: {type(generator._llm).__name__}\n")

    draft = generator.generate(
        customer_ref="cust_014",  # anonymized example reference
        customer_docs=[
            "### Data Sharing Overview\nCustomer requests anonymized vehicle trajectory data "
            "for EU fleet vehicles, delivered as a daily batch, for route-optimization analytics."
        ],
        risk_score=0.42,  # example score, as would come from attacker_model.py
        prior_assessment_summary="A similar automotive fleet-routing use case (cust_009) was "
                                  "approved with spatial generalization at 50-unit grid cells.",
    )

    print(draft.as_markdown())


if __name__ == "__main__":
    _demo()
