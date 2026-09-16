from pg.envs.finance.sim import MAX_RAISE_MONTHS, FinanceSim

BASE = {"horizon": 6, "cash": 300_000, "revenue": 50_000, "burn": 100_000, "growth": 0.0, "lag": 2, "crises": []}


def sim(**over):
    return FinanceSim({**BASE, **over})


def test_monthly_dynamics_and_bankruptcy():
    s = sim()
    assert s.runway() == 6.0
    s.advance()
    assert s.cash == 250_000 and s.month == 1
    for _ in range(5):
        s.advance()
    assert s.bankrupt is False and s.cash == 0 and s.survived and s.done
    s2 = sim(cash=120_000)
    s2.advance(), s2.advance()
    msg = s2.advance()
    assert s2.bankrupt and "insolvent" in msg and s2.months_survived == 2 and not s2.survived


def test_financing_lag_stacking_and_cap():
    s = sim(cash=200_000)  # runway 4 months
    assert "Declined" in sim(cash=1_000_000).request_financing(100_000)  # runway 20 > investor ceiling
    msg = s.request_financing(10_000_000)
    assert "capped" in msg and s.pending["amount"] == MAX_RAISE_MONTHS * 50_000
    assert "still pending" in s.request_financing(1)
    s.advance()
    assert s.cash == 150_000 and s.pending
    s.advance()
    assert s.cash == 100_000 + 300_000 and s.pending is None and s.capital_raised == 300_000


def test_crises_and_burn_adjustment():
    s = sim(crises=[{"month": 1, "kind": "revenue_shock"}, {"month": 4, "kind": "cost_spike"}])
    assert "ALERT" in s.advance()
    assert s.effective_revenue() == 30_000 and s.cash == 230_000
    s.advance(), s.advance()
    s.advance()  # month 4: shock over, spike hits
    assert s.effective_revenue() == 50_000 and s.cash == 230_000 - 140_000 - 150_000 - 50_000
    assert "-30.0%" in s.adjust_burn(-50)  # clamped per call
    assert s.burn == 70_000 and abs(s.growth - (-0.015)) < 1e-9
    s.adjust_burn(-30)
    assert s.burn == 60_000  # floor at 60% of initial burn


def test_forecast_ignores_future_crises():
    s = sim(crises=[{"month": 1, "kind": "cost_spike"}])
    assert "cash-out" in s.forecast(3) and "$150,000" in s.forecast(3)
