"""Deterministic month-by-month startup finance simulator. No LLM, no randomness after task sampling.

The procedural knowledge it rewards: financing closes only after a hidden lag (2-4 months), investors
decline when runway is comfortable and cap round size, crises arrive unannounced, and the built-in
forecast ignores them. Survival needs early, repeated, non-stacked raises plus burn cuts after shocks.
"""
from __future__ import annotations

import math

HORIZON = 24
INVESTOR_MAX_RUNWAY = 9.0  # investors decline requests while runway exceeds this many months
MAX_RAISE_MONTHS = 6.0  # a round is capped at this many months of current net burn
SHOCK_FACTOR, SHOCK_MONTHS = 0.6, 3  # revenue shock: -40% for 3 months
CHURN_FACTOR = 0.8  # churn step-down: permanent -20% revenue
SPIKE_BURN_MULTIPLE = 1.5  # one-time cost spike, in months of burn
MAX_BURN_CHANGE_PCT = 30.0  # per adjust_burn call
BURN_FLOOR = 0.6  # burn cannot go below this fraction of the initial burn
GROWTH_PER_BURN = 0.05  # a 10% burn change moves monthly growth by 0.5 percentage points

CRISIS_TEXT = {
    "revenue_shock": "ALERT: a major customer paused contracts; revenue is down 40% for the next 3 months.",
    "cost_spike": "ALERT: an unexpected legal settlement was charged against cash this month.",
    "churn": "ALERT: a wave of customer churn permanently reduced revenue by 20%.",
}


def usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


class FinanceSim:
    def __init__(self, params: dict):
        self.horizon: int = params.get("horizon", HORIZON)
        self.cash: float = params["cash"]
        self.revenue: float = params["revenue"]  # underlying monthly revenue, before any active shock
        self.burn: float = params["burn"]
        self.initial_burn: float = params["burn"]
        self.growth: float = params["growth"]
        self.lag: int = params["lag"]
        self.crises: dict[int, str] = {c["month"]: c["kind"] for c in params["crises"]}
        self.month = 0
        self.shock_until = -1
        self.pending: dict | None = None  # {"amount", "submitted", "arrives"}
        self.capital_raised = 0.0
        self.bankrupt = False
        self.notes: list[str] = []

    # ---- derived quantities ---------------------------------------------
    @property
    def done(self) -> bool:
        return self.bankrupt or self.month >= self.horizon

    @property
    def survived(self) -> bool:
        return self.month >= self.horizon and not self.bankrupt

    @property
    def months_survived(self) -> int:
        return self.month - 1 if self.bankrupt else self.month

    def effective_revenue(self) -> float:
        return self.revenue * (SHOCK_FACTOR if self.month <= self.shock_until else 1.0)

    def net_burn(self) -> float:
        return self.burn - self.effective_revenue()

    def runway(self) -> float:
        nb = self.net_burn()
        return self.cash / nb if nb > 0 else math.inf

    # ---- actions ----------------------------------------------------------
    def audit(self) -> str:
        shock = " (depressed by an active revenue shock)" if self.month <= self.shock_until else ""
        runway = "profitable" if self.runway() == math.inf else f"{self.runway():.1f} months"
        if self.pending:
            pending = f"{usd(self.pending['amount'])} requested in month {self.pending['submitted']}, not yet closed"
        else:
            pending = "none"
        return (
            f"Month {self.month}/{self.horizon} | cash {usd(self.cash)} | monthly revenue {usd(self.effective_revenue())}"
            f"{shock}, growth {self.growth:.1%}/mo | monthly burn {usd(self.burn)} | net burn {usd(self.net_burn())}"
            f" | runway {runway} | pending financing: {pending} | capital raised so far {usd(self.capital_raised)}"
        )

    def forecast(self, months: int) -> str:
        months = max(1, min(int(months), self.horizon))
        cash, revenue, cash_out = self.cash, self.revenue, None
        for m in range(1, months + 1):
            revenue *= 1 + self.growth
            cash += revenue * (SHOCK_FACTOR if self.month + m <= self.shock_until else 1.0) - self.burn
            if cash < 0 and cash_out is None:
                cash_out = self.month + m
        verdict = f"projected cash-out in month {cash_out}" if cash_out else "no projected cash-out in this window"
        return (
            f"Naive projection from month {self.month} (current revenue trend and burn; ignores unforeseen events "
            f"and pending financing): cash at month {self.month + months} = {usd(cash)}; {verdict}."
        )

    def request_financing(self, amount: float) -> str:
        if self.pending:
            p = self.pending
            return (f"Rejected: a request for {usd(p['amount'])} from month {p['submitted']} is still pending. "
                    "Only one request can be pending at a time.")
        if amount <= 0:
            return "Rejected: amount must be positive."
        if self.runway() > INVESTOR_MAX_RUNWAY:
            return "Declined by investors: current runway is too comfortable to raise now. Try again when runway is shorter."
        cap = MAX_RAISE_MONTHS * self.net_burn()
        granted = min(amount, cap)
        self.pending = {"amount": granted, "submitted": self.month, "arrives": self.month + self.lag}
        capped = f" (investors capped the round at {usd(cap)})" if amount > cap else ""
        return f"Term sheet signed for {usd(granted)}{capped}. Funds arrive when the round closes."

    def adjust_burn(self, pct: float) -> str:
        pct = max(-MAX_BURN_CHANGE_PCT, min(MAX_BURN_CHANGE_PCT, float(pct)))
        old = self.burn
        self.burn = max(old * (1 + pct / 100), BURN_FLOOR * self.initial_burn)
        change = self.burn / old - 1
        self.growth += GROWTH_PER_BURN * change
        floor = " (hit the minimum viable burn)" if self.burn == BURN_FLOOR * self.initial_burn and pct < 0 else ""
        return f"Monthly burn {usd(old)} -> {usd(self.burn)} ({change:+.1%}){floor}; revenue growth now {self.growth:.1%}/mo."

    def advance(self) -> str:
        if self.done:
            return "The episode is over."
        self.month += 1
        events = []
        kind = self.crises.get(self.month)
        if kind == "revenue_shock":
            self.shock_until = self.month + SHOCK_MONTHS - 1
        elif kind == "churn":
            self.revenue *= CHURN_FACTOR
        elif kind == "cost_spike":
            self.cash -= SPIKE_BURN_MULTIPLE * self.burn
        if kind:
            events.append(CRISIS_TEXT[kind])

        start_cash = self.cash
        self.revenue *= 1 + self.growth
        self.cash += self.effective_revenue() - self.burn
        if self.pending and self.month >= self.pending["arrives"]:
            self.cash += self.pending["amount"]
            self.capital_raised += self.pending["amount"]
            events.append(f"Financing round of {usd(self.pending['amount'])} closed and the cash arrived.")
            self.pending = None

        if self.cash < 0:
            self.bankrupt = True
            events.append("Cash fell below zero: the company is insolvent. Episode over.")
        elif self.month >= self.horizon:
            events.append(f"Reached month {self.horizon} solvent. Episode over.")
        summary = (f"Month {self.month}/{self.horizon}: revenue {usd(self.effective_revenue())}, burn {usd(self.burn)}, "
                   f"cash {usd(self.cash)} ({usd(self.cash - start_cash)} this month).")
        return " ".join([summary, *events])
