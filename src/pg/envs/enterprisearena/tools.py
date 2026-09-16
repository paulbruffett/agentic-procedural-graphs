"""LangChain tools over one EnterpriseArena episode. Descriptions come from the simulator's own registries."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from pg.envs.enterprisearena.env import EnterpriseArenaEpisode


def describe(spec: dict) -> str:
    params = "; ".join(f"{name}: {p.get('description', '')}" for name, p in spec.get("parameters", {}).items())
    return spec["description"] + (f" Parameters: {params}" if params else "")


def make_tools(ep: EnterpriseArenaEpisode, specs: dict[str, dict]) -> list[BaseTool]:
    # Information tools (count toward the monthly budget)
    def check_cash_in_bank() -> str:
        return ep.use_tool("check_cash_in_bank")

    def access_financial_docs(action: Literal["list", "retrieve"], filename: str | None = None) -> str:
        return ep.use_tool("access_financial_docs", action=action, **({"filename": filename} if filename else {}))

    def cash_flow_forecast_calculation(months: int = 3, growth_rate_override: float | None = None) -> str:
        extra = {} if growth_rate_override is None else {"growth_rate_override": growth_rate_override}
        return ep.use_tool("cash_flow_forecast_calculation", months=months, **extra)

    def check_market_data(data_type: str, start_month: int = 0, end_month: int | None = None) -> str:
        extra = {} if end_month is None else {"end_month": end_month}
        return ep.use_tool("check_market_data", data_type=data_type, start_month=start_month, **extra)

    # Actions (exactly one per month; each closes the month)
    def fund_raising_request(type: Literal["equity", "debt"], amount: float) -> str:
        return ep.act("fund_raising_request", type=type, amount=amount)

    def book_closing() -> str:
        return ep.act("book_closing")

    def pass_() -> str:
        return ep.act("pass")

    # Notepad (free; the only memory across months)
    def save_note(content: str, tags: list[str] | None = None) -> str:
        return ep.save_note(content, tags or [])

    def recall_notes(query: str = "", tags: list[str] | None = None) -> str:
        return ep.recall_notes(query, tags or [])

    simulator_tools = [check_cash_in_bank, access_financial_docs, cash_flow_forecast_calculation, check_market_data,
                       fund_raising_request, book_closing, pass_]
    tools = [
        StructuredTool.from_function(f, name=f.__name__.rstrip("_"), description=describe(specs[f.__name__.rstrip("_")]))
        for f in simulator_tools
    ]
    tools.append(StructuredTool.from_function(
        save_note, description="Save a note to your notepad (does not use the tool budget). Tags organize notes, "
                               "e.g. cash, fundraising, macro, strategy."))
    tools.append(StructuredTool.from_function(
        recall_notes, description="Retrieve notes by keyword and/or tags (does not use the tool budget). Without "
                                  "filters, returns the most recent notes."))
    return tools
