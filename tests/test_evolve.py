"""Evolution loop with rollouts and refiner stubbed out (no network)."""
import json

from pg import evolve as ev
from pg.config import Config
from pg.envs.base import Task
from pg.graph import Edge, EdgeRef, EditSet, ProceduralGraph, Relation
from pg.refiner import build_prompt
from pg.trajectory import Trajectory

GOOD = Edge(src="Start", rel=Relation.LEADS_TO, dst="search", guidance="search first")


class StubEnv:
    name = "stub"
    description = "stub env"

    def tasks(self, split):
        return [Task(id=f"{split}{i}", prompt="p") for i in range(4)]

    def tool_descriptions(self):
        return [("search", "find")]

    def compact_trajectory(self, t):
        return t.compact()


def fake_run_batch(env, tasks, graph, cfg):
    score = 1.0 if GOOD.key() in {e.key() for e in graph.edges} else 0.5
    return [Trajectory(task_id=t.id, score=score, usage={"solver_cost": 0.01}) for t in tasks]


class FakeRun:
    def __init__(self):
        self.logs, self.summary, self.tables, self.files = [], {}, {}, []

    def log(self, data, step=None):
        self.logs.append((step, data))


def test_accept_then_reject_then_memory_in_prompt(tmp_path, monkeypatch):
    proposals = iter([
        EditSet(rationale="add backbone", add_edges=[GOOD]),
        EditSet(rationale="remove it", delete_edges=[EdgeRef(src="Start", rel=Relation.LEADS_TO, dst="search")]),
        EditSet(),
        EditSet(rationale="ghost", add_edges=[Edge(src="ghost", rel=Relation.LEADS_TO, dst="search")]),
    ])
    monkeypatch.setattr(ev, "run_batch", fake_run_batch)
    monkeypatch.setattr(ev, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(ev, "propose_edits", lambda llm, prompt: (next(proposals), {"cost": 0.1, "input_tokens": 7, "output_tokens": 3}))
    run = FakeRun()
    monkeypatch.setattr(ev, "log_table", lambda r, key, rows: r.tables.__setitem__(key, rows))
    monkeypatch.setattr(ev, "log_files", lambda r, name, kind, paths: r.files.extend(paths))

    init = ProceduralGraph.skeleton("stub", [("search", "find")])
    best = ev.evolve(StubEnv(), init, Config(), rounds=4, batch=2, val_n=4, out_dir=tmp_path, run=run)

    assert [step for step, _ in run.logs] == [0, 1, 2, 3, 4]
    assert [d["val/score"] for _, d in run.logs] == [0.5, 1.0, 1.0, 1.0, 1.0]
    assert [d["accepted"] for _, d in run.logs[1:]] == [1, 0, 0, 0]
    assert "val/candidate_score" not in run.logs[3][1]  # empty edit set was not validated
    assert "val/candidate_score" not in run.logs[4][1]  # edits that applied to nothing skip validation too
    assert run.logs[1][1]["tokens/refiner"] == 10 and run.logs[3][1]["rejection_memory"] == 1
    assert run.logs[-1][1]["cost/total"] > run.logs[0][1]["cost/total"]
    assert run.summary["val/best_score"] == 1.0 and len(run.tables["rounds"]) == 4
    assert tmp_path / "best.json" in run.files

    log = [json.loads(line) for line in (tmp_path / "evolution_log.jsonl").read_text().splitlines()]
    assert [e.get("accepted") for e in log[1:]] == [True, False, False, False]
    assert log[1]["val_before"] == 0.5 and log[1]["val_after"] == 1.0
    assert log[2]["val_after"] == 0.5 and log[3]["val_after"] is None  # empty edit set skips validation
    assert log[4]["val_after"] is None and "(no-op: nothing applied)" in log[4]["edits"]
    assert GOOD.key() in {e.key() for e in best.edges}
    assert ProceduralGraph.load(tmp_path / "best.json") == best

    assert "(empty)" in (tmp_path / "prompts" / "round_2.txt").read_text()
    round3 = (tmp_path / "prompts" / "round_3.txt").read_text()
    assert "round 2: delete edges: Start-LEADS_TO->search | val 1.000 -> 0.500" in round3


def test_wandb_disabled_by_default():
    from pg.tracking import start_run

    with start_run(Config(), "evolve", "name", {}) as run:
        assert run is None


def test_prompt_tail_truncation_keeps_memory():
    ts = [Trajectory(task_id=f"t{i}", final_text="x" * 500) for i in range(10)]
    memory = [{"round": 1, "edits": "add nodes: a", "rationale": "r", "val_before": 0.5, "val_after": 0.4}]
    p = build_prompt("d", [("search", "find")], ProceduralGraph(name="g"), ts, ts, memory, max_chars=1000)
    assert "...[truncated]" in p and "round 1: add nodes: a" in p and "## Tools\n- search: find" in p

    summarized = build_prompt("d", [("search", "find")], ProceduralGraph(name="g"), ts, ts, memory,
                              max_chars=10_000, compact=lambda t: f"SUMMARY {t.task_id}")
    assert "SUMMARY t0" in summarized and "x" * 100 not in summarized
