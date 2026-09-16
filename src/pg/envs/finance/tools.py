from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from pg.envs.finance.sim import FinanceSim


def make_tools(sim: FinanceSim) -> list[BaseTool]:
    @tool
    def audit_cash() -> str:
        """Report the current month, cash, revenue, burn, net burn, runway and any pending financing request."""
        return sim.audit()

    @tool
    def forecast_runway(months: int) -> str:
        """Project cash forward `months` months from the current revenue trend and burn. The projection is naive:
        it ignores unforeseen events and pending financing."""
        return sim.forecast(months)

    @tool
    def submit_financing_request(amount: float) -> str:
        """Ask investors for `amount` dollars. Funds arrive only when the round closes, which takes several months.
        Only one request may be pending at a time, and investors may decline or cap the amount."""
        return sim.request_financing(amount)

    @tool
    def adjust_burn(pct: float) -> str:
        """Change monthly burn by `pct` percent (between -30 and +30 per call). Cutting burn slows revenue growth;
        raising it speeds growth up."""
        return sim.adjust_burn(pct)

    @tool
    def write_note(text: str) -> str:
        """Save a short note for later in this episode."""
        sim.notes.append(f"month {sim.month}: {text}")
        return f"Saved note {len(sim.notes)}."

    @tool
    def read_notes() -> str:
        """Read all notes saved in this episode."""
        return "\n".join(sim.notes) or "No notes yet."

    @tool
    def advance_month() -> str:
        """Advance the simulation by one month and report what happened, including any unexpected events."""
        return sim.advance()

    return [audit_cash, forecast_runway, submit_financing_request, adjust_burn, write_note, read_notes, advance_month]
