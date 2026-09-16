from __future__ import annotations

from pg.config import Config
from pg.envs.base import Environment, Episode, Task, read_tasks
from pg.envs.finance.sim import HORIZON, FinanceSim
from pg.envs.finance.tools import make_tools

MAX_STEPS = 3 * HORIZON

SYSTEM_PROMPT = f"""You are an autonomous CFO agent operating a company finance simulator through tools.
Act only by calling tools. Every tool call uses one step of a {MAX_STEPS}-step budget, and all {HORIZON} months
must be advanced with advance_month within that budget, so keep moving time forward."""


class FinanceEpisode(Episode):
    def __init__(self, task: Task):
        self.sim = FinanceSim(task.meta)
        self.tools = make_tools(self.sim)

    def is_done(self) -> bool:
        return self.sim.done

    def score(self) -> dict:
        s = self.sim
        return {
            "score": s.months_survived / s.horizon,  # 1.0 exactly when the company survives the horizon
            "success": s.survived,
            "survived": s.survived,
            "months_survived": s.months_survived,
            "capital_raised": round(s.capital_raised),
            "final_cash": round(s.cash),
        }


class FinanceEnv(Environment):
    name = "finance"
    description = (
        f"Startup finance simulator: the agent is CFO for {HORIZON} simulated months and must keep cash >= 0. "
        "Revenue grows but the company burns cash. Financing requests close only after a hidden lag, only one can "
        "be pending, and investors decline or cap requests depending on runway. Unannounced crises (revenue shocks, "
        "cost spikes, churn) hit mid-episode, and the forecast tool ignores them. Score is the fraction of the "
        "horizon survived; success means surviving all months."
    )
    max_steps = MAX_STEPS
    success_threshold = 1.0

    def __init__(self, cfg: Config):
        self.data_dir = cfg.data_dir / "finance"

    def tasks(self, split: str) -> list[Task]:
        return read_tasks(self.data_dir / f"{split}.jsonl", "finance")

    def start(self, task: Task) -> FinanceEpisode:
        return FinanceEpisode(task)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
