"""Evolution loop with rollouts and refiner stubbed out (no network)."""
import json

import pytest

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


def test_refiner_failure_keeps_graph_and_errored_episodes_are_not_evidence(tmp_path, monkeypatch):
    def run_batch_with_crash(env, tasks, graph, cfg):
        ts = fake_run_batch(env, tasks, graph, cfg)
        if len(tasks) == 3:  # train batches only; validation stays clean
            ts[0] = Trajectory(task_id="crashed", score=0.0, error="TimeoutError: boom")
        return ts

    replies = iter([ValueError("no JSON in reply"), EditSet(rationale="add backbone", add_edges=[GOOD])])

    def propose(llm, prompt):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"cost": 0.1, "input_tokens": 7, "output_tokens": 3}

    monkeypatch.setattr(ev, "run_batch", run_batch_with_crash)
    monkeypatch.setattr(ev, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(ev, "propose_edits", propose)

    init = ProceduralGraph.skeleton("stub", [("search", "find")])
    best = ev.evolve(StubEnv(), init, Config(), rounds=2, batch=3, val_n=4, out_dir=tmp_path)

    log = [json.loads(line) for line in (tmp_path / "evolution_log.jsonl").read_text().splitlines()]
    assert log[1]["accepted"] is False and log[1]["val_after"] is None  # failed round: no validation, run goes on
    assert log[1]["rationale"].startswith("refiner failed: ValueError")
    assert log[2]["accepted"] is True and GOOD.key() in {e.key() for e in best.edges}
    assert "crashed" not in (tmp_path / "prompts" / "round_1.txt").read_text()


def test_thin_validation_is_no_data_and_existing_run_is_not_overwritten(tmp_path, monkeypatch):
    def run_batch_flaky_candidate(env, tasks, graph, cfg):
        ts = fake_run_batch(env, tasks, graph, cfg)
        if graph.edges and len(tasks) == 4:  # candidate validation: half the episodes crash
            ts[:2] = [Trajectory(task_id=t.task_id, error="TimeoutError: boom") for t in ts[:2]]
        return ts

    monkeypatch.setattr(ev, "run_batch", run_batch_flaky_candidate)
    monkeypatch.setattr(ev, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(ev, "propose_edits", lambda llm, prompt: (
        EditSet(rationale="add backbone", add_edges=[GOOD]), {"cost": 0.1, "input_tokens": 7, "output_tokens": 3}))

    init = ProceduralGraph.skeleton("stub", [("search", "find")])
    best = ev.evolve(StubEnv(), init, Config(), rounds=1, batch=2, val_n=4, out_dir=tmp_path)

    entry = json.loads((tmp_path / "evolution_log.jsonl").read_text().splitlines()[1])
    # 2/4 scored at 1.0 would beat the 0.5 baseline, but 2 < 0.8 * 4, so the round is not decided on it.
    assert entry["accepted"] is False and entry["val_after"] is None and "no validation data" in entry["edits"]
    assert best == init

    with pytest.raises(FileExistsError):
        ev.evolve(StubEnv(), init, Config(), rounds=1, batch=2, val_n=4, out_dir=tmp_path)
    ev.evolve(StubEnv(), init, Config(), rounds=1, batch=2, val_n=4, out_dir=tmp_path, overwrite=True)


def test_out_of_credits_stops_the_run_after_writing_the_batch(tmp_path, monkeypatch):
    def run_batch(env, tasks, graph, cfg):
        if len(tasks) == 4:  # baseline validation is fine
            return fake_run_batch(env, tasks, graph, cfg)
        return [Trajectory(task_id=t.id, error="APIStatusError: Error code: 402", fatal=True) for t in tasks]

    proposals = []
    monkeypatch.setattr(ev, "run_batch", run_batch)
    monkeypatch.setattr(ev, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(ev, "propose_edits", lambda llm, prompt: proposals.append(prompt))

    init = ProceduralGraph.skeleton("stub", [("search", "find")])
    with pytest.raises(RuntimeError, match="fatal API error"):
        ev.evolve(StubEnv(), init, Config(), rounds=5, batch=2, val_n=4, out_dir=tmp_path)

    assert not proposals  # stopped in round 1, before the refiner and before four more useless rounds
    assert (tmp_path / "trajectories" / "round_1_train.jsonl").exists() and (tmp_path / "best.json").exists()


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
