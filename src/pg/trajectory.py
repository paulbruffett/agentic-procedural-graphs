from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Step(BaseModel):
    index: int
    tool: str
    args: dict
    observation: str  # truncated to Config.observation_max_chars


class Trajectory(BaseModel):
    task_id: str
    steps: list[Step] = Field(default_factory=list)
    final_text: str | None = None
    score: float = 0.0
    success: bool = False
    metrics: dict = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)  # solver_/guidance_ input_tokens, output_tokens, cost
    guidance_log: list[str] = Field(default_factory=list)
    error: str | None = None

    def compact(self, max_obs: int = 200) -> str:
        """Short textual form for the refiner prompt."""
        lines = [f"task={self.task_id} score={self.score:.2f} success={self.success} steps={len(self.steps)}"]
        for s in self.steps:
            obs = s.observation.replace("\n", " ")
            if len(obs) > max_obs:
                obs = obs[:max_obs] + "..."
            lines.append(f"  {s.index}. {s.tool}({json.dumps(s.args)}) -> {obs}")
        if self.final_text:
            lines.append(f"  final: {self.final_text[:max_obs]}")
        if self.error:
            lines.append(f"  error: {self.error[:max_obs]}")
        return "\n".join(lines)


def write_jsonl(path: str | Path, trajectories: list[Trajectory]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Trajectory]:
    return [Trajectory.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def episode_rows(trajectories: list[Trajectory], **labels) -> list[dict]:
    """One flat row per episode (scalars only: no steps, no guidance text) for tables and paired analysis.
    `labels` name the condition, e.g. graph="none" or round=3, phase="val"."""
    rows = []
    for t in trajectories:
        scalars = {k: v for k, v in t.metrics.items() if isinstance(v, (bool, int, float, str))}
        rows.append({**labels, "task_id": t.task_id, "score": t.score, "success": t.success,
                     "steps": len(t.steps), "error": t.error, **scalars, **t.usage})
    return rows


def summarize(trajectories: list[Trajectory]) -> dict:
    """Scores average over episodes that ran to completion; episodes that errored (crash, timeout) are
    counted separately so infrastructure failures are not reported as task failures."""
    scored = [t for t in trajectories if not t.error]
    n = len(scored) or 1
    usage: dict = {}
    for t in trajectories:
        for k, v in t.usage.items():
            usage[k] = usage.get(k, 0) + v
    return {
        "n": len(trajectories),
        "n_scored": len(scored),
        "mean_score": sum(t.score for t in scored) / n,
        "success_rate": sum(t.success for t in scored) / n,
        "mean_steps": sum(len(t.steps) for t in scored) / n,
        "errors": len(trajectories) - len(scored),
        "stopped_early": sum(1 for t in scored if t.metrics.get("stopped_early")),
        **usage,
    }
