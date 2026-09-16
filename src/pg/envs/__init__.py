from __future__ import annotations

from pg.config import Config
from pg.envs.base import Environment


def make_env(name: str, cfg: Config) -> Environment:
    if name == "hotpotqa":
        from pg.envs.hotpotqa.env import HotpotQAEnv

        return HotpotQAEnv(cfg)
    if name == "finance":
        from pg.envs.finance.env import FinanceEnv

        return FinanceEnv(cfg)
    if name == "enterprisearena":
        from pg.envs.enterprisearena.env import EnterpriseArenaEnv

        return EnterpriseArenaEnv(cfg)
    raise ValueError(f"unknown environment {name!r}; choose hotpotqa, finance or enterprisearena")
