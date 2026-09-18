"""EnterpriseArena (CFO-Env, Han et al. 2026) behind the Environment / Episode interface.

Mirrors the benchmark's own agent loop: each month the solver may call information tools (at most 20) and notes,
then exactly one action closes the month. The solver's conversation then restarts from the new month's status, so
only notes carry information across months. Nothing is written to disk (no trial folder is created).
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

from pg.config import Config
from pg.envs.base import Environment, Episode, Task, read_tasks
from pg.envs.enterprisearena.data import CODE_DIR, FULL_MONTHS, verify
from pg.envs.enterprisearena.tools import make_tools
from pg.trajectory import Trajectory

STEPS_PER_MONTH = 8  # episode step cap (tool, action and note calls) = STEPS_PER_MONTH * months
NOTE_BUDGET_PER_MONTH = 20  # notes are free in the simulator; cap them so a note loop cannot stall a month
DROPPED = "Not executed: this month already ended with an action. Continue in the new month."

SYSTEM_PROMPT = """You are an AI CFO (Chief Financial Officer) agent for a fintech company. Steward liquidity: maintain enough cash to support operations, growth, and investments. Do not allow the company to run out of cash.

Rules:
- You have a tool budget of 20 information-tool calls per month (check_cash_in_bank, access_financial_docs, cash_flow_forecast_calculation, check_market_data). You may use anywhere from 0 to 20 in a month.
- Memory operations (save_note, recall_notes) do not count toward the tool budget.
- Each month must end with exactly one action (fund_raising_request, book_closing, or pass). You can take an action at any time; you do not need to use any tools first. The action closes the month.
- Your conversation history resets every month. The only way to carry information across months is your notepad: use recall_notes to read notes when you need them.
- There is no prescribed order. You decide what information you need (if any) and when to act."""

_import_lock = threading.Lock()
_simulator = None


def simulator():
    """The simulator's `arena` module, imported once from third_party/cfo-env."""
    global _simulator
    with _import_lock:
        if _simulator is None:
            if verify():
                raise FileNotFoundError("CFO-Env code is missing or modified; run `uv run pg gen-data enterprisearena`")
            sys.path.insert(0, str(CODE_DIR))
            import arena  # the simulator's top-level modules: arena, environment_state, tools, actions

            _simulator = arena
    return _simulator


def _usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _musd(x: float) -> str:
    """19038075 -> '$19.0M' (compact enough for a 132-line trajectory summary)."""
    value = x / 1e6
    return f"-${-value:.1f}M" if value < 0 else f"${value:.1f}M"


class EnterpriseArenaEpisode(Episode):
    def __init__(self, task: Task):
        sim = simulator()
        config = json.loads((CODE_DIR / "config.json").read_text())
        config["data_paths"] = {k: str(CODE_DIR / v) for k, v in config["data_paths"].items()}
        config["environment_config"]["max_episode_months"] = task.meta["months"]
        config["stochastic_config"]["seed"] = task.meta["seed"]
        with tempfile.TemporaryDirectory() as tmp:  # the simulator only accepts a config path, read at construction
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            self.arena = sim.EnterpriseArena(str(path))
        self.arena.reset()
        self.state = self.arena.env_state
        self.notes: list[dict] = []
        self.month_log: list[dict] = []  # one structured record per closed month (used by compact_trajectory)
        self.month_closed = False  # an action already ended the month during the current solver turn
        self._extra: list[tuple[str, dict, str]] = []  # months the episode closed by itself, for the agent
        self._tools, self._note_calls = 0, 0  # this month's usage
        self._restart: str | None = self.status()
        specs = {s["name"]: s for s in sim.list_tools() + sim.list_actions()}
        self.tools = make_tools(self, specs)

    # ---- what the solver sees at the start of each month ---------------------
    def status(self) -> str:
        o = self.arena._get_observation()
        lines = [
            f"Month: {o['current_month']}",  # horizon not disclosed, as in the benchmark's own agent
            f"Date: {o['date']}",
            f"Cash: {_usd(o['cash_balance'])}",
            f"Borrowers: {o['active_users']:,}",
            f"Loan portfolio: {_usd(o['loan_portfolio_gross'])}",
            f"Total debt: {_usd(o['total_debt'])}",
            f"Tool calls left this month: {o['tool_calls_remaining']}",
        ]
        if o["pending_fundraisings"]:
            lines.append(f"Pending fundraisings: {len(o['pending_fundraisings'])}")
        return "## Environment status\n" + "\n".join(f"- {line}" for line in lines)

    # ---- tool implementations -------------------------------------------------
    def use_tool(self, name: str, **params) -> str:
        if self.month_closed:
            return DROPPED
        result, _ = self.arena.use_tool(name, **params)
        self._tools += 1
        text = f"{result.message}\n{json.dumps(result.data, separators=(',', ':'), default=str)}" if result.success \
            else f"Failed: {result.message}"
        text += f"\nTool calls left this month: {self.state.tool_budget_per_step - self.state.tool_calls_this_step}"
        if not self.state.can_use_tool() and not self.is_done():  # the benchmark's agent passes when the budget runs out
            text += "\n\n" + self._auto_close("Tool budget exhausted, so the month was closed with pass.")
        return text

    def act(self, name: str, **params) -> str:
        if self.month_closed:
            return DROPPED
        result, _ = self.arena.take_action(name, **params)
        head = result.message if result.success else f"Action failed: {result.message}"
        return f"{head}\n{self._close_month(name, params, result)}"

    def _auto_close(self, reason: str) -> str:
        """Close the month with `pass` on the episode's own initiative, and report it to the agent so the
        month is recorded as a step rather than disappearing inside another tool's observation."""
        result, _ = self.arena.take_action("pass")
        text = f"{reason}\n{self._close_month('pass', {}, result)}"
        self._extra.append(("pass", {}, text))
        return text

    def _close_month(self, action: str, args: dict, result) -> str:
        month = self.state.current_month
        events: list[str] = []
        if not result.success:
            events.append("rejected(pending)" if "already have a pending" in result.message
                          else f"action failed: {result.message[:60]}")
        step = self.arena.end_step()
        notifications = step.info.get("fundraising_notifications") or []
        for n in notifications:
            events.append(f"approved {n['financing_type']} {_musd(n['amount_received'])}" if n["success"]
                          else f"declined {n['financing_type']}")
        if self.is_done():
            events.append(f"episode over: {self.state.termination_reason}")
        self.month_log.append({"month": month, "action": action, "args": args, "cash": round(self.state.cash_balance),
                               "tools": self._tools, "notes": self._note_calls, "events": events})
        self._tools, self._note_calls = 0, 0
        self.month_closed = True

        update = step.info.get("update_result") or {}
        lines = [f"Month {month} closed: revenue {_usd(update.get('revenue', 0))}, net cash flow "
                 f"{_usd(update.get('net_cash_flow', 0))}, cash {_usd(self.state.cash_balance)}."]
        lines += [n["message"] for n in notifications]
        if self.is_done():
            lines.append(f"Episode over: {self.state.termination_reason}.")
        else:
            self._restart = "Last month:\n" + "\n".join(lines) + "\n\n" + self.status()
        return "\n".join(lines)

    def _note_call(self) -> str | None:
        """Notes do not consume the simulator's tool budget, so cap them per month; hitting the cap closes
        the month, which guarantees an episode always makes progress."""
        self._note_calls += 1
        if self._note_calls >= NOTE_BUDGET_PER_MONTH and not self.is_done():
            return self._auto_close("Note budget exhausted for this month, so the month was closed with pass.")
        return None

    def save_note(self, content: str, tags: list[str]) -> str:
        if self.month_closed:
            return DROPPED
        note_id = f"note_{len(self.notes) + 1}"
        self.notes.append({"id": note_id, "content": content, "tags": tags, "created_month": self.state.current_month})
        closed = self._note_call()
        return f"Note saved: {note_id}" + (f"\n\n{closed}" if closed else "")

    def recall_notes(self, query: str, tags: list[str], limit: int = 10) -> str:
        if self.month_closed:
            return DROPPED
        matches = [
            n for n in reversed(self.notes)
            if (not tags or set(tags) & set(n["tags"])) and (not query or query.lower() in n["content"].lower())
        ][:limit]
        closed = self._note_call()
        return json.dumps({"count": len(matches), "notes": matches}) + (f"\n\n{closed}" if closed else "")

    # ---- Episode interface ------------------------------------------------------
    def is_done(self) -> bool:
        return self.state.episode_terminated

    def on_text(self, text: str) -> bool:
        if not self.is_done() and not self.month_closed:  # the benchmark's agent defaults to pass
            self._auto_close("No decision in the reply, so the month was closed with pass.")
        return False

    def pop_extra_steps(self) -> list[tuple[str, dict, str]]:
        extra, self._extra = self._extra, []
        return extra

    def pop_context_reset(self) -> str | None:
        restart, self._restart = self._restart, None
        self.month_closed = False
        return restart

    def score(self) -> dict:
        s = self.state
        horizon = s.max_episode_months + 1  # months 0..max_episode_months
        bankrupt = s.termination_reason == "Cash balance went negative"
        survived = s.episode_terminated and not bankrupt
        months = horizon if survived else s.current_month
        return {
            "score": months / horizon,
            "success": survived,
            # "stopped" means our harness ended it (step cap, timeout), not the simulator
            "outcome": "survived" if survived else ("bankrupt" if bankrupt else "stopped"),
            "survived": survived,
            "months_survived": months,
            "horizon": horizon,
            "valuation_score": round(s.calculate_score()),
            "final_cash": round(s.cash_balance),
            "equity_raised": round(s.total_equity_raised),
            "equity_rounds": len(s.equity_rounds),
            "debt_instruments": len(s.debt_instruments),
            "fundraising_requests": sum(a["action_name"] == "fund_raising_request" for a in self.arena.action_history),
            "info_tool_calls": s.total_tool_calls,
            "month_log": self.month_log,
        }


class EnterpriseArenaEnv(Environment):
    name = "enterprisearena"
    description = (
        "EnterpriseArena (CFO-Env): the agent is CFO of a consumer-lending fintech for up to 132 simulated months "
        "driven by real 2015-2025 macro data. Each month it may call up to 20 information tools (cash, documents, "
        "a heuristic cash-flow forecast, market data) plus notes, then must take exactly one action "
        "(fund_raising_request for equity or debt, book_closing, or pass), which closes the month; the "
        "conversation resets every month, so only notes persist. Fundraising approval is probabilistic (market "
        "conditions, fewer approvals after each equity round, leverage limits on debt), results arrive 1-6 months "
        "later at 70-100% of the amount, and only one request can be pending. Hidden user-growth surges drain "
        "cash through loan originations. Score is the fraction of months survived; success means cash never "
        "went negative."
    )
    max_steps = STEPS_PER_MONTH * (FULL_MONTHS + 1)

    def __init__(self, cfg: Config):
        self.data_dir = cfg.data_dir / "enterprisearena"

    def tasks(self, split: str) -> list[Task]:
        return read_tasks(self.data_dir / f"{split}.jsonl", "enterprisearena")

    def start(self, task: Task) -> EnterpriseArenaEpisode:
        return EnterpriseArenaEpisode(task)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def compact_trajectory(self, trajectory: Trajectory) -> str:
        """One line per simulated month, read from the episode's month log (no parsing of observation text)
        and collapsing runs of uneventful `pass` months, so a 132-month episode fits the refiner's context."""
        log = trajectory.metrics.get("month_log")
        if not log:
            return super().compact_trajectory(trajectory)
        m = trajectory.metrics
        lines = [f"task={trajectory.task_id} score={trajectory.score:.2f} survived={trajectory.success}"
                 f" months={m.get('months_survived')} valuation={_usd(m.get('valuation_score', 0))}"
                 f" equity_raised={_usd(m.get('equity_raised', 0))} equity_rounds={m.get('equity_rounds')}"
                 f" info_tool_calls={m.get('info_tool_calls')}"]
        quiet: dict | None = None

        def flush() -> None:
            nonlocal quiet
            if quiet:
                if quiet["n"] == 1:
                    lines.append(f"m{quiet['start']} pass, cash {quiet['to']}, tools={quiet['tools']}")
                else:
                    lines.append(f"m{quiet['start']}-m{quiet['end']} pass x{quiet['n']}, "
                                 f"cash {quiet['from']}->{quiet['to']}, tools={quiet['tools']}")
                quiet = None

        for row in log:
            cash, events = _musd(row["cash"]), row.get("events") or []
            tools = row.get("tools", 0) + row.get("notes", 0)
            if row["action"] == "pass" and not events:
                quiet = ({"start": row["month"], "end": row["month"], "from": cash, "to": cash, "n": 1, "tools": tools}
                         if quiet is None else
                         {**quiet, "end": row["month"], "to": cash, "n": quiet["n"] + 1, "tools": quiet["tools"] + tools})
                continue
            flush()
            args = row.get("args") or {}
            detail = (f"({args.get('type')} ${float(args.get('amount', 0)) / 1e6:.0f}M)"
                      if row["action"] == "fund_raising_request" else "")
            note = f" | {'; '.join(events)}" if events else ""
            lines.append(f"m{row['month']} {row['action']}{detail}, cash {cash}, tools={tools}{note}")
        flush()
        if trajectory.error:
            lines.append(f"error: {trajectory.error[:200]}")
        return "\n".join(lines)
