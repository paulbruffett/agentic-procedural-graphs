"""Run a split with no graph and/or several graphs; report score, success, steps and token overhead."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pg.agent import run_batch
from pg.config import Config
from pg.envs.base import Environment
from pg.graph import ProceduralGraph
from pg.tracking import log_table
from pg.trajectory import episode_rows, raise_if_fatal, summarize, write_jsonl


def load_graph(spec: str) -> ProceduralGraph | None:
    return None if spec == "none" else ProceduralGraph.load(spec)


def label_of(spec: str) -> str:
    """File-name-safe form of a graph spec; `<label>.jsonl` holds that condition's trajectories."""
    return re.sub(r"[^\w.-]+", "_", spec)


def table_row(r: dict) -> dict:
    """Headline metrics for one graph, from a summarize() row."""
    solver = r.get("solver_input_tokens", 0) + r.get("solver_output_tokens", 0)
    guidance = r.get("guidance_input_tokens", 0) + r.get("guidance_output_tokens", 0)
    return {
        "graph": r["graph"],
        "graph_sha256": r.get("graph_sha256"),
        "n": r["n"],
        "mean_score": r["mean_score"],
        "success_rate": r["success_rate"],
        "mean_steps": r["mean_steps"],
        "solver_tokens": solver,
        "guidance_tokens": guidance,
        "guidance_overhead": guidance / solver if solver else None,
        "cost": r.get("solver_cost", 0) + r.get("guidance_cost", 0),
        "stopped": r.get("stopped_early", 0),
        "errors": r["errors"],
    }


def evaluate(
    env: Environment, split: str, n: int | None, specs: list[str], cfg: Config, out_dir: Path, run=None
) -> list[dict]:
    tasks = env.tasks(split)[:n]
    rows, episodes = [], []
    for spec in specs:
        print(f"== graph={spec}: {len(tasks)} {env.name}/{split} tasks", flush=True)
        trajectories = run_batch(env, tasks, load_graph(spec), cfg)
        label = label_of(spec)
        write_jsonl(out_dir / f"{label}.jsonl", trajectories)
        # The hash pins which version of a graph file (e.g. best.json, rewritten by every evolve run) was used.
        sha = None if spec == "none" else hashlib.sha256(Path(spec).read_bytes()).hexdigest()[:12]
        rows.append({"graph": spec, "graph_sha256": sha, **summarize(trajectories)})
        episodes += episode_rows(trajectories, graph=spec)
        (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))  # after every graph: a stopped run stays analyzable
        raise_if_fatal(trajectories)
        if run:
            for key, value in table_row(rows[-1]).items():
                if key != "graph":
                    run.summary[f"{label}/{key}"] = value
    if run:
        log_table(run, "eval", [table_row(r) for r in rows])
        log_table(run, "episodes", episodes)  # scalars per episode, for paired analysis (see analysis/paired_comparison.py)
    return rows


def format_table(rows: list[dict]) -> str:
    header = (f"{'graph':<45} {'n':>4} {'score':>6} {'success':>7} {'steps':>6} {'solver tok':>11} "
              f"{'guide tok':>10} {'overhead':>8} {'cost $':>8} {'stop':>5} {'err':>4}")
    lines = [header, "-" * len(header)]
    for t in map(table_row, rows):
        overhead = "-" if t["guidance_overhead"] is None else f"{t['guidance_overhead']:.0%}"
        lines.append(
            f"{t['graph'][-45:]:<45} {t['n']:>4} {t['mean_score']:>6.3f} {t['success_rate']:>7.1%} {t['mean_steps']:>6.1f}"
            f" {t['solver_tokens']:>11,} {t['guidance_tokens']:>10,} {overhead:>8} {t['cost']:>8.3f}"
            f" {t['stopped']:>5} {t['errors']:>4}"
        )
    return "\n".join(lines)
