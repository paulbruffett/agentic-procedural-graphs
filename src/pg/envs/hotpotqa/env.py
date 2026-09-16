from __future__ import annotations

from pg.config import Config
from pg.envs.base import Environment, Episode, Task, read_tasks
from pg.envs.hotpotqa.scoring import exact_match, f1_score
from pg.envs.hotpotqa.tools import make_tools

SYSTEM_PROMPT = """You answer multi-hop questions using tools over a fixed set of context paragraphs.
Use `search` and `lookup` to gather evidence, then call `finish` with a short answer span
(an entity name, a date, a number, or yes/no), not a sentence."""


class HotpotQAEpisode(Episode):
    def __init__(self, task: Task):
        self.gold = task.meta["answer"]
        self.answer: str | None = None
        self.tools = make_tools(task.meta["paragraphs"], self.submit)

    def submit(self, answer: str) -> None:
        self.answer = answer.strip()

    def is_done(self) -> bool:
        return self.answer is not None

    def on_text(self, text: str) -> bool:
        self.submit(text)
        return True

    def score(self) -> dict:
        pred = self.answer or ""
        f1 = f1_score(pred, self.gold)
        return {"score": f1, "success": f1 >= HotpotQAEnv.success_threshold, "em": exact_match(pred, self.gold),
                "f1": f1, "answer": pred, "gold": self.gold}


class HotpotQAEnv(Environment):
    name = "hotpotqa"
    description = (
        "HotpotQA (distractor setting): answer a multi-hop question whose evidence is spread over two of ten "
        "context paragraphs. Tools search the paragraphs by word overlap, look up a paragraph by title, and "
        "submit a short answer span. Score is token F1 against the gold answer."
    )
    max_steps = 8
    success_threshold = 0.5

    def __init__(self, cfg: Config):
        self.data_dir = cfg.data_dir / "hotpotqa"

    def tasks(self, split: str) -> list[Task]:
        return read_tasks(self.data_dir / f"{split}.jsonl", "hotpotqa")

    def start(self, task: Task) -> HotpotQAEpisode:
        return HotpotQAEpisode(task)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
