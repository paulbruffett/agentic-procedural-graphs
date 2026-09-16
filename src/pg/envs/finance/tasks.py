"""Seeded task generator: each task is a company with hidden financing lag and crisis schedule."""
from __future__ import annotations

import random
from pathlib import Path

from pg.envs.base import Task, write_tasks
from pg.envs.finance.sim import HORIZON

CRISIS_KINDS = ("revenue_shock", "cost_spike", "churn")


def sample_params(rng: random.Random, horizon: int = HORIZON) -> dict:
    revenue = rng.uniform(40_000, 80_000)
    burn = revenue * rng.uniform(1.6, 2.2)
    months = sorted(rng.sample(range(3, horizon - 2), rng.choice([2, 3])))
    return {
        "horizon": horizon,
        "cash": round((burn - revenue) * rng.uniform(5.0, 8.0)),
        "revenue": round(revenue),
        "burn": round(burn),
        "growth": round(rng.uniform(0.02, 0.04), 4),
        "lag": rng.choice([2, 3, 4]),
        "crises": [{"month": m, "kind": rng.choice(CRISIS_KINDS)} for m in months],
    }


def make_task(task_id: str, params: dict) -> Task:
    h = params["horizon"]
    prompt = (
        f"You are the CFO of a venture-backed startup. It is month 0 of a {h}-month horizon. "
        f"Keep the company solvent (cash never below zero) through month {h}. Time only moves forward when you "
        f"call advance_month; the episode ends at month {h} or at insolvency."
    )
    return Task(id=task_id, prompt=prompt, meta=params)


def generate(out_dir: Path, seed: int = 0, n_train: int = 30, n_val: int = 20, n_test: int = 30) -> dict[str, int]:
    counts = {"train": n_train, "val": n_val, "test": n_test}
    i = 0
    for split, n in counts.items():
        tasks = []
        for _ in range(n):
            tasks.append(make_task(f"finance-{seed}-{i}", sample_params(random.Random(seed * 1_000_003 + i))))
            i += 1
        write_tasks(out_dir / f"{split}.jsonl", tasks)
    return counts
