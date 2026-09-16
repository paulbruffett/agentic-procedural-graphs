"""Online half of the method: localize the agent in the graph by its last tool call, extract the
h-hop neighborhood, and have the guidance model turn it into step-level guidance for the solver."""
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from pg.graph import ProceduralGraph
from pg.llm import usage_of
from pg.trajectory import Step

GUIDANCE_SYSTEM = """You give step-level procedural guidance to an agent that is solving a task with tools.
You see the task, the agent's most recent steps, and a procedural graph (or its neighborhood around the
agent's current position). Nodes are procedures: ACTION = a tool call, REASONING = a thinking step,
STATE = a task status. Transitions carry a condition (when it applies), guidance (how to proceed) and
pitfalls (what to avoid).

Write guidance for the agent's next step:
- Decide which transition from the active node applies, checking its condition against the recent steps.
- Say concretely what to do next and how, and which pitfalls to avoid.
- If the graph has nothing relevant, say so in one sentence instead of inventing procedure.
At most 5 sentences. Do not solve the task yourself and do not invent facts."""


def match(graph: ProceduralGraph, last_tool: str | None) -> str | None:
    """Exact match of the last tool call to a node id; None means 'not localized' (use the full graph)."""
    return last_tool if last_tool and graph.has_node(last_tool) else None


def build_context(graph: ProceduralGraph, task_text: str, recent_steps: list[Step], h: int) -> str:
    active = match(graph, recent_steps[-1].tool if recent_steps else None)
    sub = graph.neighborhood(active, h) if active else graph
    steps = "\n".join(f"{s.index}. {s.tool}({json.dumps(s.args)}) -> {s.observation}" for s in recent_steps)
    return f"Task:\n{task_text}\n\nRecent steps:\n{steps or '(none yet)'}\n\n{sub.serialize(active)}"


def generate_guidance(
    llm: BaseChatModel, graph: ProceduralGraph, task_text: str, recent_steps: list[Step], h: int
) -> tuple[str, dict]:
    resp = llm.invoke([SystemMessage(GUIDANCE_SYSTEM), HumanMessage(build_context(graph, task_text, recent_steps, h))])
    return resp.text.strip(), usage_of(resp)
