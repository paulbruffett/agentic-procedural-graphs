"""Sample the HotpotQA distractor split once and cache it as JSONL, so `datasets` is only hit by gen-data."""
from __future__ import annotations

import random
from pathlib import Path

from pg.envs.base import Task, write_tasks


def generate(out_dir: Path, seed: int = 0, n: int = 300) -> dict[str, int]:
    """Split n sampled validation examples 1/3 train, 1/6 val, rest test (100/50/150 for n=300)."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = random.Random(seed).sample(range(len(ds)), n)
    n_train, n_val = n // 3, n // 6
    splits = {"train": idx[:n_train], "val": idx[n_train : n_train + n_val], "test": idx[n_train + n_val :]}

    for split, ids in splits.items():
        tasks = []
        for i in ids:
            ex = ds[i]
            paragraphs = [
                {"title": title, "text": "".join(sents)}
                for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"])
            ]
            tasks.append(
                Task(
                    id=ex["id"],
                    prompt=f"Question: {ex['question']}",
                    meta={"answer": ex["answer"], "type": ex["type"], "level": ex["level"], "paragraphs": paragraphs},
                )
            )
        write_tasks(out_dir / f"{split}.jsonl", tasks)
    return {k: len(v) for k, v in splits.items()}
