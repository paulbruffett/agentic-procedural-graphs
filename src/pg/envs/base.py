"""The only scenario-facing interface. Everything in pg/ outside envs/ is task-agnostic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from pg.trajectory import Trajectory


class Task(BaseModel):
    id: str
    prompt: str  # what the agent sees
    meta: dict = Field(default_factory=dict)  # hidden ground truth / simulator parameters


def write_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(t.model_dump_json() + "\n" for t in tasks))


def read_tasks(path: Path, hint: str) -> list[Task]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run pg gen-data {hint}` first")
    return [Task.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


class Episode(ABC):
    """Per-episode state. One instance per (task, rollout); tools close over it, so
    concurrent episodes never share state."""

    tools: list[BaseTool]

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def score(self) -> dict:
        """Must contain 'score' (float, higher is better) and 'success' (bool)."""

    def on_text(self, text: str) -> bool:
        """Called when the solver replies with text and no tool call. Return True if the
        text should be accepted as the final answer (episode ends); False to nudge the
        solver to keep calling tools."""
        return False

    def pop_extra_steps(self) -> list[tuple[str, dict, str]]:
        """Actions the episode took on its own since the last call (e.g. auto-closing a simulated month), as
        (tool, args, observation) triples. The agent records them as steps so trajectories stay complete."""
        return []

    def pop_context_reset(self) -> str | None:
        """Called at the start and after every tool batch or text-only turn. Return an opening message to
        restart the solver's conversation from (task prompt + this message), e.g. at each new simulated
        month, or None to keep the history. Step records and guidance are unaffected."""
        return None


class Environment(ABC):
    name: str
    description: str  # one paragraph for the refiner
    max_steps: int  # tool calls per episode

    @abstractmethod
    def tasks(self, split: str) -> list[Task]: ...

    @abstractmethod
    def start(self, task: Task) -> Episode: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    def compact_trajectory(self, trajectory: Trajectory) -> str:
        """Textual form of one trajectory for the refiner prompt. Override when episodes are long enough that
        the raw step list would be tail-truncated away (e.g. summarize one line per simulated month)."""
        return trajectory.compact()

    def tool_descriptions(self) -> list[tuple[str, str]]:
        """(tool name, description) pairs; used for the scratch skeleton graph and refiner context."""
        ep = self.start(self.tasks("train")[0])
        return [(t.name, t.description) for t in ep.tools]
