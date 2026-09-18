from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Models (OpenRouter ids). Guidance defaults to the solver model, as in the paper.
    solver_model: str = "openai/gpt-5.6-luna"
    guidance_model: str | None = None
    refiner_model: str = "anthropic/claude-opus-5"
    temperature: float | None = None  # None = do not send (reasoning models reject it)

    # Paper hyperparameters
    h: int = 2  # neighborhood hops
    w: int = 3  # trajectory window shown to the guidance model

    # Runtime
    concurrency: int = 4
    episode_timeout_s: int = 14_400  # wall-clock budget per episode (4h); 0 disables. A guided 132-month
    # EnterpriseArena episode is 500-800 steps, i.e. 1000-1600 LLM calls, so this only catches real hangs.
    max_nudges: int = 3  # times to prompt the solver to continue after a text-only turn
    observation_max_chars: int = 600  # tool observation truncation in Step records
    refiner_k: int = 3  # best / worst trajectories shown to the refiner
    refiner_max_chars: int = 60_000  # tail-truncate refiner context (paper's L_max)
    val_min_scored: float = 0.8  # a validation pass counts only if this fraction of episodes ran without error

    wandb_project: str | None = None  # set to log metrics to Weights & Biases

    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    graphs_dir: Path = field(default_factory=lambda: ROOT / "graphs")
    runs_dir: Path = field(default_factory=lambda: ROOT / "runs")

    @property
    def guidance_model_id(self) -> str:
        return self.guidance_model or self.solver_model

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(ROOT / ".env")
        cfg = cls()
        cfg.solver_model = os.getenv("PG_SOLVER_MODEL", cfg.solver_model)
        cfg.guidance_model = os.getenv("PG_GUIDANCE_MODEL") or None
        cfg.refiner_model = os.getenv("PG_REFINER_MODEL", cfg.refiner_model)
        cfg.concurrency = int(os.getenv("PG_CONCURRENCY", cfg.concurrency))
        cfg.wandb_project = os.getenv("PG_WANDB_PROJECT") or None
        return cfg
