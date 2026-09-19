"""Agent loop with scripted fake chat models (no network)."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from pg.envs.base import Environment, Episode, Task

from pg.agent import NUDGE, run_episode
from pg.config import ROOT, Config
from pg.envs.hotpotqa.env import HotpotQAEnv
from pg.graph import ProceduralGraph
from tests.test_hotpotqa import TASK


class ScriptedModel(GenericFakeChatModel):
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


def cfg(tmp_path):
    return Config(data_dir=tmp_path, graphs_dir=tmp_path, runs_dir=tmp_path)


def test_tool_loop_records_steps_and_injects_guidance(tmp_path):
    solver = ScriptedModel(messages=iter([call("search", query="Eiffel Tower"), call("finish", answer="Paris")]), seen=[])
    guide = ScriptedModel(messages=iter(["Search for the tower first.", "Now finish."]), seen=[])
    graph = ProceduralGraph.load(ROOT / "graphs/hotpotqa_expert.json")

    t = run_episode(HotpotQAEnv(cfg(tmp_path)), TASK, graph, cfg(tmp_path), solver, guide)

    assert t.error is None
    assert [s.tool for s in t.steps] == ["search", "finish"]
    assert t.score == 1.0 and t.success
    assert t.guidance_log == ["Search for the tower first.", "Now finish."]
    # guidance is in the system prompt of the matching solver turn, not in the stored history
    assert "Procedural Graph Guidance:\nNow finish." in solver.seen[1][0].content
    assert all("Procedural Graph Guidance" not in m.content for m in solver.seen[1][1:])
    # second guidance call is localized on the `search` node
    assert "Active node: [search]" in guide.seen[1][1].content
    assert "Active node: none" in guide.seen[0][1].content


def test_no_graph_and_text_answer(tmp_path):
    solver = ScriptedModel(messages=iter(["Paris"]), seen=[])
    t = run_episode(HotpotQAEnv(cfg(tmp_path)), TASK, None, cfg(tmp_path), solver)
    assert t.steps == [] and t.final_text == "Paris" and t.score == 1.0 and t.guidance_log == []
    assert "Procedural Graph Guidance" not in solver.seen[0][0].content


class MonthEpisode(Episode):
    """Two 'months'; each advance restarts the solver's context."""

    def __init__(self):
        self.month, self.restart = 0, "Month 0"
        self.tools = [StructuredTool.from_function(self.advance, name="advance", description="Close the month.")]

    def advance(self) -> str:
        self.month += 1
        self.restart = f"Month {self.month}"
        return "closed"

    def is_done(self):
        return self.month >= 2

    def score(self):
        return {"score": self.month / 2, "success": self.month >= 2}

    def pop_context_reset(self):
        restart, self.restart = self.restart, None
        return restart


class MonthEnv(Environment):
    name, description, max_steps = "months", "", 5

    def tasks(self, split):
        return []

    def start(self, task):
        return MonthEpisode()

    def system_prompt(self):
        return "sys"


def test_context_reset_replaces_history(tmp_path):
    solver = ScriptedModel(messages=iter([call("advance"), call("advance")]), seen=[])
    t = run_episode(MonthEnv(), Task(id="m", prompt="Run it."), None, cfg(tmp_path), solver)
    assert t.score == 1.0 and [s.tool for s in t.steps] == ["advance", "advance"]
    assert [m.content for m in solver.seen[0][1:]] == ["Run it.\n\nMonth 0"]
    assert [m.content for m in solver.seen[1][1:]] == ["Run it.\n\nMonth 1"]  # old turn and tool result are gone


class AutoCloseEpisode(MonthEpisode):
    """Closes a month by itself when the solver replies without a tool call."""

    def on_text(self, text):
        self.advance()
        self.extra = [("auto_pass", {}, "month closed by the environment")]
        return False

    def pop_extra_steps(self):
        extra, self.extra = getattr(self, "extra", []), []
        return extra


class AutoCloseEnv(MonthEnv):
    def start(self, task):
        return AutoCloseEpisode()


def test_extra_steps_from_episode_are_recorded(tmp_path):
    solver = ScriptedModel(messages=iter(["thinking out loud", call("advance")]), seen=[])
    t = run_episode(AutoCloseEnv(), Task(id="m", prompt="Run it."), None, cfg(tmp_path), solver)
    assert [s.tool for s in t.steps] == ["auto_pass", "advance"]  # the self-closed month is not lost
    assert t.score == 1.0 and t.metrics["stopped_early"] is False


def test_nudge_when_text_not_accepted(tmp_path, monkeypatch):
    from pg.envs.hotpotqa import env as hp

    monkeypatch.setattr(hp.HotpotQAEpisode, "on_text", lambda self, text: False)
    solver = ScriptedModel(messages=iter(["thinking...", call("finish", answer="Paris")]), seen=[])
    t = run_episode(HotpotQAEnv(cfg(tmp_path)), TASK, None, cfg(tmp_path), solver)
    assert [s.tool for s in t.steps] == ["finish"] and t.score == 1.0
    assert solver.seen[1][-1].content == NUDGE


class OutOfCredits(Exception):
    status_code = 402  # shaped like openai.APIStatusError


class FailingModel(ScriptedModel):
    failure: Exception = None

    def _generate(self, messages, *args, **kwargs):
        raise self.failure


def test_fatal_api_errors_are_flagged_and_other_errors_are_not(tmp_path):
    env = HotpotQAEnv(cfg(tmp_path))
    broke = run_episode(env, TASK, None, cfg(tmp_path), FailingModel(messages=iter([]), failure=OutOfCredits("402 no credits")))
    assert broke.fatal and broke.error.startswith("OutOfCredits") and broke.steps == []
    flaky = run_episode(env, TASK, None, cfg(tmp_path), FailingModel(messages=iter([]), failure=TimeoutError("slow")))
    assert flaky.error.startswith("TimeoutError") and not flaky.fatal
