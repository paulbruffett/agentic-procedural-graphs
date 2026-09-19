"""The solver loop as a LangGraph StateGraph.

    START -> guide -> solver -> tools -> guide ...   (tool call)
                        solver -> text  -> guide ...   (text-only reply; nudge, or accept as final)

`guide` is a no-op when running without a procedural graph. An episode may restart the solver's conversation
(Episode.pop_context_reset), e.g. at each simulated month; step records and guidance carry on across restarts.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from pg.config import Config
from pg.envs.base import Environment, Task
from pg.graph import ProceduralGraph
from pg.guidance import generate_guidance
from pg.llm import add_usage, is_fatal, make_llm, usage_of
from pg.trajectory import Step, Trajectory

NUDGE = "The task is not complete. Continue by calling tools."


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: list[Step]
    guidance: str
    guidance_log: list[str]
    usage: dict
    nudges: int
    final_text: str | None
    done: bool


def run_episode(
    env: Environment,
    task: Task,
    graph: ProceduralGraph | None,
    cfg: Config,
    solver: BaseChatModel | None = None,
    guide_llm: BaseChatModel | None = None,
) -> Trajectory:
    episode = env.start(task)
    tools = {t.name: t for t in episode.tools}
    solver = (solver or make_llm(cfg.solver_model, cfg.temperature)).bind_tools(episode.tools)
    if graph is not None and guide_llm is None:
        guide_llm = make_llm(cfg.guidance_model_id, cfg.temperature)

    def opening(extra: str | None) -> list[AnyMessage]:
        return [HumanMessage(task.prompt if extra is None else f"{task.prompt}\n\n{extra}")]

    def fresh_context(extra: str) -> list[AnyMessage]:
        return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *opening(extra)]

    def drain_extra(steps: list[Step]) -> None:
        for tool, args, observation in episode.pop_extra_steps():
            steps.append(Step(index=len(steps), tool=tool, args=args,
                              observation=observation[: cfg.observation_max_chars]))

    def guide(state: AgentState) -> dict:
        if graph is None:
            return {}
        text, usage = generate_guidance(guide_llm, graph, task.prompt, state["steps"][-cfg.w :], cfg.h)
        return {
            "guidance": text,
            "guidance_log": state["guidance_log"] + [text],
            "usage": add_usage(state["usage"], usage, "guidance"),
        }

    def solve(state: AgentState) -> dict:
        # Guidance goes into this turn's system prompt only; it is never stored in the message history.
        system = env.system_prompt()
        if state["guidance"]:
            system += "\n\nProcedural Graph Guidance:\n" + state["guidance"]
        resp = solver.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [resp], "usage": add_usage(state["usage"], usage_of(resp), "solver")}

    def act(state: AgentState) -> dict:
        steps, replies = list(state["steps"]), []
        for call in state["messages"][-1].tool_calls:
            if episode.is_done() or len(steps) >= env.max_steps:
                replies.append(ToolMessage("Not executed: the episode has ended.", tool_call_id=call["id"]))
                continue
            if call["name"] not in tools:
                obs = f"Error: unknown tool {call['name']!r}. Available: {', '.join(tools)}."
            else:
                try:
                    obs = str(tools[call["name"]].invoke(call["args"]))
                except Exception as e:  # bad arguments etc. are shown to the solver, not raised
                    obs = f"Error: {e}"
            steps.append(Step(index=len(steps), tool=call["name"], args=call["args"], observation=obs[: cfg.observation_max_chars]))
            replies.append(ToolMessage(obs, tool_call_id=call["id"]))
            drain_extra(steps)  # the call may have advanced the episode on its own (e.g. closed a month)
        done = episode.is_done() or len(steps) >= env.max_steps
        restart = None if done else episode.pop_context_reset()
        if restart is not None:
            return {"messages": fresh_context(restart), "steps": steps, "done": False, "nudges": 0}
        return {"messages": replies, "steps": steps, "done": done}

    def text(state: AgentState) -> dict:
        reply = state["messages"][-1].text
        accepted = episode.on_text(reply)
        steps = list(state["steps"])
        drain_extra(steps)  # on_text may have advanced the episode (e.g. closed a month with pass)
        if accepted or episode.is_done() or state["nudges"] >= cfg.max_nudges:
            return {"final_text": reply, "done": True, "steps": steps}
        restart = episode.pop_context_reset()
        if restart is not None:
            return {"messages": fresh_context(restart), "steps": steps, "nudges": 0}
        return {"messages": [HumanMessage(NUDGE)], "steps": steps, "nudges": state["nudges"] + 1}

    def after_solver(state: AgentState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "text"

    def continue_or_end(state: AgentState) -> str:
        return END if state["done"] else "guide"

    builder = StateGraph(AgentState)
    builder.add_node("guide", guide)
    builder.add_node("solver", solve)
    builder.add_node("tools", act)
    builder.add_node("text", text)
    builder.add_edge(START, "guide")
    builder.add_edge("guide", "solver")
    builder.add_conditional_edges("solver", after_solver, ["tools", "text"])
    builder.add_conditional_edges("tools", continue_or_end, ["guide", END])
    builder.add_conditional_edges("text", continue_or_end, ["guide", END])
    app = builder.compile()

    state: AgentState = {
        "messages": opening(episode.pop_context_reset()), "steps": [], "guidance": "", "guidance_log": [],
        "usage": {}, "nudges": 0, "final_text": None, "done": False,
    }
    error, fatal = None, False
    try:
        # Stream full state values so a mid-episode failure still leaves the last good state for scoring.
        limit = 3 * (env.max_steps + cfg.max_nudges) + 5
        deadline = time.monotonic() + cfg.episode_timeout_s if cfg.episode_timeout_s else None
        for state in app.stream(state, {"recursion_limit": limit}, stream_mode="values"):
            if deadline and time.monotonic() > deadline:  # one stuck episode must not stall a whole batch
                raise TimeoutError(f"episode exceeded {cfg.episode_timeout_s}s")
    except Exception as e:
        error, fatal = f"{type(e).__name__}: {e}", is_fatal(e)

    result = episode.score()
    result["stopped_early"] = not episode.is_done()  # step cap, timeout or error, rather than a real ending
    return Trajectory(
        task_id=task.id,
        steps=state["steps"],
        final_text=state["final_text"],
        score=float(result.pop("score")),
        success=bool(result.pop("success")),
        metrics=result,
        usage=state["usage"],
        guidance_log=state["guidance_log"],
        error=error,
        fatal=fatal,
    )


def run_batch(env: Environment, tasks: list[Task], graph: ProceduralGraph | None, cfg: Config) -> list[Trajectory]:
    """Run episodes concurrently. Each episode gets its own env.start(), so threads share no state."""
    solver = make_llm(cfg.solver_model, cfg.temperature)
    guide_llm = make_llm(cfg.guidance_model_id, cfg.temperature) if graph is not None else None

    def one(task: Task) -> Trajectory:
        t = run_episode(env, task, graph, cfg, solver, guide_llm)
        print(f"    {task.id}: score={t.score:.2f} steps={len(t.steps)}" + (f" error={t.error}" if t.error else ""), flush=True)
        return t

    with ThreadPoolExecutor(cfg.concurrency) as pool:
        return list(pool.map(one, tasks))
