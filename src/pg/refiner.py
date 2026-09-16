"""The refiner: contrast successful and failed trajectories and propose structural graph edits."""
from __future__ import annotations

import json
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from openai import BadRequestError, NotFoundError
from pydantic import ValidationError

from pg.graph import EditSet, ProceduralGraph
from pg.llm import usage_of
from pg.trajectory import Trajectory

REFINER_SYSTEM = """You refine a procedural graph that guides an LLM agent through a task.

Graph semantics:
- Nodes are procedures. ACTION nodes are tool calls and their id MUST equal the tool name exactly (the agent
  is localized in the graph by exact match of its last tool call). REASONING nodes are thinking steps and
  STATE nodes are task statuses; those may use any id.
- Edges are (src, relation, dst) with relation in LEADS_TO, TRIGGERS, PROVIDES_INPUT_FOR, CONVERGES_TO, and
  carry a condition (when the transition applies), guidance (how to proceed) and pitfalls (what to avoid).
- Cycles are allowed (retries, monthly loops).

Your job: compare the successful and failed trajectories, find the procedural differences that explain the
outcomes, and propose ONE edit set that encodes them in the graph.
- add_nodes / add_edges add structure. Re-adding an existing (src, rel, dst) edge replaces its attributes.
- delete_nodes / delete_edges remove structure that is wrong or misleading.
- Prefer a few targeted, high-value edits (at most about 8 operations).
- Make conditions, guidance and pitfalls concrete: name tools, state thresholds and orderings you can justify
  from the trajectories.
- Write rules that transfer to ANY task in this environment. Environment-level facts (tool names, orderings,
  thresholds, step budgets, answer-format rules) belong in the graph; content of individual tasks does not:
  no task ids, entity names, question wording, specific queries or answers copied from the trajectories.
  Use trajectories as evidence for a rule, not as examples inside it; if an illustration helps, use a generic
  placeholder such as "<entity name> <attribute>". The rationale may cite specific trajectories.
- Do not re-propose edits listed in the rejection memory; they were validated and made things worse.
- If nothing is warranted, return an empty edit set."""


def _block(title: str, trajectories: list[Trajectory], compact: Callable[[Trajectory], str]) -> str:
    return f"## {title}\n" + ("\n\n".join(compact(t) for t in trajectories) or "(none)")


def build_prompt(
    env_description: str,
    tool_descriptions: list[tuple[str, str]],
    graph: ProceduralGraph,
    best: list[Trajectory],
    worst: list[Trajectory],
    rejected: list[dict],
    max_chars: int,
    compact: Callable[[Trajectory], str] | None = None,
) -> str:
    compact = compact or Trajectory.compact
    tools = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions)
    head = (
        f"## Environment\n{env_description}\n\n## Tools\n{tools}\n\n"
        f"## Current graph (JSON)\n{json.dumps(graph.model_dump(mode='json'), indent=1)}\n\n"
    )
    memory = "\n".join(
        f"- round {r['round']}: {r['edits']} | val {r['val_before']:.3f} -> {r['val_after']:.3f}"
        f" | rationale: {r['rationale']}"
        for r in rejected
    )
    evidence = (
        _block("Highest-scoring trajectories", best, compact)
        + "\n\n"
        + _block("Lowest-scoring trajectories", worst, compact)
        + f"\n\n## Rejection memory (edits already tried and rejected)\n{memory or '(empty)'}"
    )
    if len(evidence) > max_chars:  # keep the tail, like the paper's L_max truncation
        evidence = "...[truncated]\n" + evidence[-max_chars:]
    return head + evidence + "\n\nPropose the edit set now."


# A provider that cannot do structured output answers 400/404; a bad payload raises ValueError/ValidationError.
FALLBACK_ERRORS = (ValueError, ValidationError, BadRequestError, NotFoundError)


def propose_edits(llm: BaseChatModel, prompt: str) -> tuple[EditSet, dict]:
    messages = [SystemMessage(REFINER_SYSTEM), HumanMessage(prompt)]
    try:
        out = llm.with_structured_output(EditSet, method="json_schema", include_raw=True).invoke(messages)
        if out["parsed"] is None:
            raise ValueError(out["parsing_error"])
        return out["parsed"], usage_of(out["raw"])
    except FALLBACK_ERRORS as e:
        # Only a schema/parse problem falls back to a plain JSON reply; transport, auth and rate-limit
        # errors propagate so they are not misreported (and the prompt is not paid for twice).
        print(f"  structured output failed ({type(e).__name__}); retrying with plain JSON", flush=True)
        schema = json.dumps(EditSet.model_json_schema())
        resp = llm.invoke(messages + [HumanMessage(f"Reply with only a JSON object matching this schema:\n{schema}")])
        text = resp.text
        return EditSet.model_validate_json(text[text.find("{") : text.rfind("}") + 1]), usage_of(resp)
