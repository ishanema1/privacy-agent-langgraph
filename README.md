# Privacy Assessment Agent — LangGraph Rebuild

The same tool-calling privacy-assessment agent from [Agentic-Privacy-Assessment-Pipeline](https://github.com/ishanema1/Agentic-Privacy-Assessment-Pipeline)'s `agent.py`, rebuilt on [LangGraph](https://github.com/langchain-ai/langgraph) instead of a hand-rolled `while` loop.

## Why this exists as a separate exercise

The original repo deliberately implements its agent loop from scratch, directly against the tool-use API — a stronger way to demonstrate understanding what a framework like LangGraph is actually doing underneath. This repo is the natural follow-up: build the same agent *on* that framework, so both "I understand the mechanics" and "I can use the framework fluently" are backed by real code instead of one or the other.

## ✓ Verified

This was originally built in a sandboxed environment with no PyPI access, so `langgraph` itself could not be installed there before publishing. To keep the untested surface area as small as possible at the time:

- **`agent_node`, `should_continue`, `tools_node`, `finalize_node`** are plain Python functions operating on a state dict, with **no LangGraph import** — fully unit-tested (8 tests) without the package installed.
- **A full manual walkthrough test** composes those four functions in the same order LangGraph's engine would, and asserts the resulting trace and draft match `agent.py`'s hand-rolled loop exactly.
- **`build_graph()`** — the function that constructs and compiles the actual `StateGraph` — has since been verified: **all 9 tests pass**, including `test_build_graph_and_run_end_to_end`, confirmed on `langgraph` (pytest 8.1.2, Python 3.11.9). `python langgraph_agent.py` also runs cleanly end-to-end, producing a tool-call trace and draft identical to the pre-verified manual walkthrough.

Note: an earlier version of `attacker_model.py` (shared with the original repo) only caught `ImportError` when optionally importing `xgboost`. On a real machine with an xgboost/setuptools version mismatch, that import raised `AttributeError` instead, which wasn't caught — a real gap this verification pass surfaced and fixed. The except clause now catches any import-time failure, not just `ImportError`, since third-party packages can fail to import for reasons beyond "not installed."

## Run it

```bash
pip install -r requirements.txt   # includes langgraph
python langgraph_agent.py         # runs the demo
python -m pytest test_langgraph_agent.py -v   # 9 tests, all passing (verified on langgraph, pytest 8.1.2, Python 3.11.9)
```

## What's reused, not reimplemented

`TOOL_SCHEMAS`, `dispatch_tool`, `build_draft_from_submission`, and `ScriptedMockAgentClient` are all imported from the original repo's `agent.py` rather than redefined — one source of truth for what each tool does and how the mock model behaves, whether it's called by the hand-rolled loop, an MCP server, or this LangGraph graph. You'll need `agent.py`, `draft_generator.py`, `confluence_client.py`, `vector_store.py`, `attacker_model.py`, and `fakes.py` copied in alongside this file for it to run.

## Graph structure

```
START → agent ─┬─(tool call)──────────→ tools ──→ agent  (loop)
                ├─(submit_privacy_assessment)→ finalize → END
                └─(no tool call)────────────────────────→ END
```

Same routing decision as the hand-rolled version in `agent.py`, expressed as a LangGraph conditional edge instead of an `if/elif` inside a `while` loop.

## Stack

Python · LangGraph · (reused) Anthropic tool-use tool contracts · pytest

---

*Uses synthetic/mocked data throughout — same fakes as the original repo.*
