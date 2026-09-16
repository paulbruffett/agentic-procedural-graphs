"""Offline half of the method. Per round k:
rollout G_{k-1} on a train batch -> refiner proposes edits from best/worst trajectories -> G_cand ->
validate on the fixed val set -> accept if S_val(G_cand) >= S_val(G_{k-1}), else add to rejection memory."""
from __future__ import annotations

import json
from pathlib import Path

from pg.agent import run_batch
from pg.config import Config
from pg.envs.base import Environment, Task
from pg.graph import ProceduralGraph
from pg.llm import make_llm
from pg.refiner import build_prompt, propose_edits
from pg.tracking import log_files, log_table
from pg.trajectory import summarize, write_jsonl


def total_cost(usage: dict) -> float:
    return sum(v for k, v in usage.items() if k.endswith("cost"))


def tokens(usage: dict, prefix: str) -> int:
    return usage.get(f"{prefix}_input_tokens", 0) + usage.get(f"{prefix}_output_tokens", 0)


def _validate(env: Environment, tasks: list[Task], graph: ProceduralGraph, cfg: Config, path: Path) -> dict:
    trajectories = run_batch(env, tasks, graph, cfg)
    write_jsonl(path, trajectories)
    return summarize(trajectories)


def _canonical(graph: ProceduralGraph) -> tuple:
    """Order-independent identity of a graph, to spot edit sets that changed nothing."""
    return (sorted(json.dumps(n.model_dump(mode="json"), sort_keys=True) for n in graph.nodes),
            sorted(json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in graph.edges))


def _append(path: Path, entry: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def evolve(
    env: Environment,
    init: ProceduralGraph,
    cfg: Config,
    rounds: int,
    batch: int,
    val_n: int | None,  # None = whole val split
    out_dir: Path,
    run=None,  # optional wandb run (see pg.tracking)
) -> ProceduralGraph:
    train, val = env.tasks("train"), env.tasks("val")[:val_n]
    refiner = make_llm(cfg.refiner_model, cfg.temperature)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts").mkdir(exist_ok=True)
    log_path = out_dir / "evolution_log.jsonl"
    log_path.write_text("")

    print(f"[round 0] validating initial graph ({init.summary()}) on {len(val)} val tasks", flush=True)
    s = _validate(env, val, init, cfg, out_dir / "trajectories" / "round_0_val.jsonl")
    current, current_val = init, s["mean_score"]
    current.save(out_dir / "round_0.json")
    current.save(out_dir / "best.json")
    spent = total_cost(s)
    _append(log_path, {"round": 0, "val_score": current_val, "graph": current.summary(), "cost": spent})
    if run:
        run.log({"val/score": current_val, "graph/nodes": len(current.nodes), "graph/edges": len(current.edges),
                 "cost/round": spent, "cost/total": spent}, step=0)
    print(f"[round 0] val {current_val:.3f}", flush=True)
    rejected: list[dict] = []
    history: list[dict] = []

    for k in range(1, rounds + 1):
        tasks = [train[((k - 1) * batch + i) % len(train)] for i in range(min(batch, len(train)))]
        print(f"[round {k}] rollout on {len(tasks)} train tasks", flush=True)
        trajectories = run_batch(env, tasks, current, cfg)
        write_jsonl(out_dir / "trajectories" / f"round_{k}_train.jsonl", trajectories)
        batch_summary = summarize(trajectories)

        ranked = sorted(trajectories, key=lambda t: t.score, reverse=True)
        n = max(1, min(cfg.refiner_k, len(ranked) // 2))
        prompt = build_prompt(
            env.description, env.tool_descriptions(), current, ranked[:n], ranked[-n:], rejected,
            cfg.refiner_max_chars, env.compact_trajectory,
        )
        (out_dir / "prompts" / f"round_{k}.txt").write_text(prompt)
        edits, refiner_usage = propose_edits(refiner, prompt)
        candidate, warnings = current.apply(edits)
        print(f"[round {k}] refiner proposed: {edits.summary()}", flush=True)

        cand_val, val_summary = None, {}
        no_op = _canonical(candidate) == _canonical(current)  # e.g. every edge dropped for unknown endpoints
        if no_op and not edits.is_empty():
            print(f"[round {k}] edits changed nothing ({len(warnings)} warnings); skipping validation", flush=True)
        if not edits.is_empty() and not no_op:
            val_summary = _validate(env, val, candidate, cfg, out_dir / "trajectories" / f"round_{k}_val.jsonl")
            cand_val = val_summary["mean_score"]
        accepted = cand_val is not None and cand_val >= current_val

        entry = {
            "round": k,
            "batch_score": batch_summary["mean_score"],
            "batch_success": batch_summary["success_rate"],
            "val_before": current_val,
            "val_after": cand_val,
            "accepted": accepted,
            "edits": edits.summary() + (" (no-op: nothing applied)" if no_op and not edits.is_empty() else ""),
            "rationale": edits.rationale,
            "warnings": warnings,
            "cost": total_cost(batch_summary) + total_cost(val_summary) + refiner_usage["cost"],
        }
        if accepted:
            current, current_val = candidate, cand_val
        elif not edits.is_empty():
            rejected.append({k2: entry[k2] for k2 in ("round", "edits", "rationale", "val_before", "val_after")})
        entry["graph"] = current.summary()
        current.save(out_dir / f"round_{k}.json")
        current.save(out_dir / "best.json")
        _append(log_path, entry)
        spent += entry["cost"]
        history.append(entry)

        if run:
            metrics = {
                "train/score": batch_summary["mean_score"],
                "train/success_rate": batch_summary["success_rate"],
                "train/mean_steps": batch_summary["mean_steps"],
                "val/score": current_val,  # score of the graph kept after this round's decision
                "accepted": int(accepted),
                "rejection_memory": len(rejected),
                "graph/nodes": len(current.nodes),
                "graph/edges": len(current.edges),
                "tokens/solver": tokens(batch_summary, "solver") + tokens(val_summary, "solver"),
                "tokens/guidance": tokens(batch_summary, "guidance") + tokens(val_summary, "guidance"),
                "tokens/refiner": refiner_usage["input_tokens"] + refiner_usage["output_tokens"],
                "cost/round": entry["cost"],
                "cost/total": spent,
            }
            if cand_val is not None:
                metrics["val/candidate_score"] = cand_val
            run.log(metrics, step=k)

        after = "n/a (empty edit set)" if cand_val is None else f"{cand_val:.3f}"
        verdict = "ACCEPTED" if accepted else "rejected"
        print(f"[round {k}] val {entry['val_before']:.3f} -> {after}: {verdict}; round cost ${entry['cost']:.4f}", flush=True)

    if run:
        run.summary["val/best_score"] = current_val
        columns = ("round", "accepted", "val_before", "val_after", "batch_score", "cost", "edits", "rationale")
        log_table(run, "rounds", [{c: e[c] for c in columns} for e in history])
        log_files(run, f"{env.name}-evolved-graph", "graph", [out_dir / "best.json", log_path])
    return current
