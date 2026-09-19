"""Paired comparison of graph conditions from one or more `pg eval` run directories.

    uv run python -m analysis.paired_comparison runs/<stamp>-eval-<env> [more run dirs = repeats] \\
        [--baseline none] [--metrics months_survived,steps] [--experiment <W&B group>]

Conditions are compared on the tasks they share, because every condition ran the same tasks (same simulator seeds /
questions) and task difficulty varies far more than the effect we are looking for. Several run directories with the
same graph spec are repeats: each (graph, task) cell is averaged over them, and tasks stay the unit of resampling.

Primary metric: `success` (e.g. survival). `score` and any `--metrics` from the episode rows are secondary.
Errored episodes (crash, timeout) are dropped, as in `summarize()`.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from datetime import datetime
from pathlib import Path
from statistics import mean

from pg.config import Config
from pg.evaluate import label_of
from pg.tracking import log_table, start_run
from pg.trajectory import episode_rows, read_jsonl

Cells = dict[str, dict[str, list[dict]]]  # graph spec -> task id -> one episode row per repeat


def load(run_dirs: list[Path]) -> Cells:
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for d in run_dirs:
        for condition in json.loads((d / "metrics.json").read_text()):
            spec = condition["graph"]
            for row in episode_rows(read_jsonl(d / f"{label_of(spec)}.jsonl")):
                if not row["error"]:
                    cells[spec][row["task_id"]].append(row)
    return cells


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (tasks only one condition succeeded on)."""
    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n)


def paired_bootstrap(diffs: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference and its 95% percentile interval, resampling tasks."""
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return mean(diffs), means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def compare(cells: Cells, baseline: str, metrics: list[str]) -> list[dict]:
    """One row per (graph, metric): the graph's paired difference from the baseline."""
    if baseline not in cells:
        raise ValueError(f"baseline {baseline!r} not among the evaluated graphs: {sorted(cells)}")

    def cell(spec: str, task: str, metric: str) -> float:
        return mean(float(row[metric]) for row in cells[spec][task])

    rows = []
    for spec in cells:
        if spec == baseline:
            continue
        tasks = sorted(set(cells[spec]) & set(cells[baseline]))
        single = all(len(cells[s][t]) == 1 for s in (spec, baseline) for t in tasks)
        for metric in metrics:
            diff, lo, hi = paired_bootstrap([cell(spec, t, metric) - cell(baseline, t, metric) for t in tasks])
            row = {"graph": spec, "baseline": baseline, "metric": metric, "tasks": len(tasks),
                   "repeats": max(len(cells[spec][t]) for t in tasks),
                   "graph_mean": mean(cell(spec, t, metric) for t in tasks),
                   "baseline_mean": mean(cell(baseline, t, metric) for t in tasks),
                   "diff": diff, "ci_low": lo, "ci_high": hi, "mcnemar_p": None}
            if metric == "success" and single:  # with repeats the pairs are not independent, so only the CI is given
                gained = sum(cell(spec, t, metric) > cell(baseline, t, metric) for t in tasks)
                lost = sum(cell(spec, t, metric) < cell(baseline, t, metric) for t in tasks)
                row.update(gained=gained, lost=lost, mcnemar_p=mcnemar_exact(gained, lost))
            rows.append(row)
    return rows


def _num(x: float, sign: str = "") -> str:
    return f"{x:{sign},.3f}" if abs(x) < 1000 else f"{x:{sign},.0f}"  # dollars do not need decimals


def format_report(rows: list[dict]) -> str:
    header = (f"{'graph':<42} {'metric':<16} {'tasks':>5} {'rep':>3} {'graph':>12} {'baseline':>12} "
              f"{'diff':>13} {'95% CI':>29} {'McNemar p':>18}")
    lines = [header, "-" * len(header)]
    for r in rows:
        p = "-" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.4f} (+{r['gained']}/-{r['lost']})"
        ci = f"[{_num(r['ci_low'], '+')}, {_num(r['ci_high'], '+')}]"
        lines.append(
            f"{r['graph'][-42:]:<42} {r['metric']:<16} {r['tasks']:>5} {r['repeats']:>3} {_num(r['graph_mean']):>12} "
            f"{_num(r['baseline_mean']):>12} {_num(r['diff'], '+'):>13} {ci:>29} {p:>18}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dirs", type=Path, nargs="+", help="runs/<stamp>-eval-<env> directories; several = repeats")
    p.add_argument("--baseline", default="none", help="graph spec every other graph is compared against")
    p.add_argument("--metrics", default="", help="extra episode metrics, comma-separated (success and score are always reported)")
    p.add_argument("--experiment", default=None, help="also log the table to W&B, in this group")
    args = p.parse_args(argv)
    metrics = ["success", "score"] + [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = compare(load(args.run_dirs), args.baseline, metrics)
    print(format_report(rows))
    if args.experiment:  # analysis is cheap and re-run often, so it is only logged when it belongs to an experiment
        with start_run(Config.from_env(), "analyze", f"analyze-{datetime.now():%Y%m%d-%H%M%S}", vars(args)) as run:
            if run:
                log_table(run, "paired", rows)


if __name__ == "__main__":
    main()
