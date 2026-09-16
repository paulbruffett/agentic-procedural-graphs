"""EnterpriseArena adapter over the real simulator (no network). Skipped until the code is fetched."""
import pytest

from pg.envs.enterprisearena.data import make_task, verify

pytestmark = pytest.mark.skipif(bool(verify()), reason="CFO-Env code not fetched; run `uv run pg gen-data enterprisearena`")


def episode(seed, months):
    from pg.envs.enterprisearena.env import EnterpriseArenaEpisode

    return EnterpriseArenaEpisode(make_task(seed, months))


def tools(ep):
    return {t.name: t for t in ep.tools}


def test_month_cycle_restarts_context_and_drops_late_calls():
    ep = episode(seed=1, months=6)
    t = tools(ep)
    assert set(t) == {"check_cash_in_bank", "access_financial_docs", "cash_flow_forecast_calculation",
                      "check_market_data", "fund_raising_request", "book_closing", "pass", "save_note", "recall_notes"}
    opening = ep.pop_context_reset()
    assert "Month: 0\n" in opening and "final month" not in opening and "6" not in ep.tools[0].description
    assert ep.pop_context_reset() is None

    assert "cash_balance" in t["check_cash_in_bank"].invoke({})
    assert "Tool calls left this month: 18" in t["check_market_data"].invoke({"data_type": "VIX"})
    assert "Note saved: note_1" in t["save_note"].invoke({"content": "raised equity in month 0", "tags": ["fundraising"]})
    assert "Month 0 closed" in t["fund_raising_request"].invoke({"type": "equity", "amount": 20e6})
    assert t["check_cash_in_bank"].invoke({}).startswith("Not executed")  # same solver turn, month already closed

    restart = ep.pop_context_reset()
    assert restart.startswith("Last month:") and "Month: 1\n" in restart
    assert "raised equity" in t["recall_notes"].invoke({"tags": ["fundraising"]})
    assert ep.on_text("thinking out loud") is False  # benchmark default: a reply without a decision passes
    assert [name for name, _, _ in ep.pop_extra_steps()] == ["pass"]  # reported so the agent records the month
    assert "Month: 2" in ep.pop_context_reset()


def run(policy, seed, months=36):
    ep = episode(seed, months)
    t = tools(ep)
    while not ep.is_done():
        ep.pop_context_reset()
        pending = any(not pf.delivered for pf in ep.state.pending_fundraisings)
        policy(t, pending)
    return ep.score()


def test_scripted_policies_bracket_survival():
    never = [run(lambda t, pending: t["pass"].invoke({}), seed) for seed in range(3)]
    greedy = [
        run(lambda t, pending: t["pass"].invoke({}) if pending
            else t["fund_raising_request"].invoke({"type": "equity", "amount": 20e6}), seed)
        for seed in range(3)
    ]
    assert not any(s["survived"] for s in never) and all(s["score"] < 1 for s in never)
    assert all(s["survived"] and s["score"] == 1.0 and s["months_survived"] == 37 for s in greedy)


def test_compact_trajectory_summarizes_months():
    from pg.config import Config
    from pg.envs.enterprisearena.env import EnterpriseArenaEnv
    from pg.trajectory import Step, Trajectory

    ep = episode(seed=2, months=12)
    t = tools(ep)
    steps = []

    def record(name, **args):
        ep.pop_context_reset()
        steps.append(Step(index=len(steps), tool=name, args=args, observation=t[name].invoke(args)[:600]))

    record("fund_raising_request", type="equity", amount=20e6)
    while not ep.is_done():
        record("pass")
    traj = Trajectory(task_id="ea", steps=steps, score=1.0, success=True, metrics=ep.score())

    text = EnterpriseArenaEnv(Config()).compact_trajectory(traj)
    assert "m0 fund_raising_request(equity $20M)" in text
    assert "m?" not in text and "cash ?" not in text  # rendered from the month log, not parsed from text
    assert "pass x" in text  # runs of uneventful months are collapsed into one line
    assert len(text.splitlines()) < len(steps) and len(text) < len(traj.compact())


def test_note_budget_closes_the_month():
    from pg.envs.enterprisearena.env import NOTE_BUDGET_PER_MONTH

    ep = episode(seed=3, months=6)
    t = tools(ep)
    for i in range(NOTE_BUDGET_PER_MONTH):  # notes are free in the simulator, so the cap forces progress
        out = t["save_note"].invoke({"content": f"note {i}"})
    assert "closed with pass" in out and ep.state.current_month == 1
    assert [name for name, _, _ in ep.pop_extra_steps()] == ["pass"]
