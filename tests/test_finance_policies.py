"""Calibration without an LLM: the scenario must reward the procedural knowledge the graph is meant to capture."""
import random

from pg.envs.finance.env import MAX_STEPS, FinanceEpisode
from pg.envs.finance.sim import INVESTOR_MAX_RUNWAY
from pg.envs.finance.tasks import make_task, sample_params

N = 50


def episodes():
    for i in range(N):
        yield FinanceEpisode(make_task(f"t{i}", sample_params(random.Random(i))))


def naive(ep: FinanceEpisode) -> int:
    tools, steps = {t.name: t for t in ep.tools}, 0
    while not ep.is_done():
        tools["advance_month"].invoke({})
        steps += 1
    return steps


def expert(ep: FinanceEpisode) -> int:
    """Raise as soon as investors allow (runway under their ceiling), never stack requests, cut burn after a crisis.
    Raising later is much worse: with runway < 3 / 5 / 7 months about 30% / 60% / 85% of companies survive."""
    tools, sim, steps = {t.name: t for t in ep.tools}, ep.sim, 0
    while not ep.is_done():
        if sim.pending is None and sim.runway() < INVESTOR_MAX_RUNWAY:
            tools["submit_financing_request"].invoke({"amount": 1e9})
            steps += 1
        summary = tools["advance_month"].invoke({})
        steps += 1
        if "ALERT" in summary:
            tools["adjust_burn"].invoke({"pct": -20})
            steps += 1
    return steps


def survival(policy) -> float:
    survived = 0
    for ep in episodes():
        assert policy(ep) <= MAX_STEPS
        survived += ep.score()["survived"]
    return survived / N


def test_naive_policy_mostly_fails():
    assert survival(naive) < 0.3


def test_expert_policy_mostly_survives():
    assert survival(expert) > 0.9
