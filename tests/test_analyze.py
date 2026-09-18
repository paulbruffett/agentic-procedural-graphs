"""Paired analysis over eval run directories, and the per-episode rows it is built on (no network)."""
import json

import pytest

from pg import evaluate as evl
from pg.analyze import compare, load, mcnemar_exact, paired_bootstrap
from pg.config import Config
from pg.envs.base import Task
from pg.trajectory import Step, Trajectory, episode_rows, write_jsonl


def write_run(run_dir, outcomes: dict[str, list[bool | None]]) -> None:
    """One eval run directory; outcomes[spec][i] is task i's success, or None for an errored episode."""
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps([{"graph": spec} for spec in outcomes]))
    for spec, results in outcomes.items():
        write_jsonl(run_dir / f"{evl.label_of(spec)}.jsonl", [
            Trajectory(task_id=f"t{i}", success=bool(ok), score=1.0 if ok else 0.25, metrics={"months": 12 if ok else 3},
                       error=None if ok is not None else "TimeoutError: boom")
            for i, ok in enumerate(results)
        ])


def test_episode_rows_are_flat_scalars_without_guidance_text():
    t = Trajectory(task_id="a", score=0.5, success=True, steps=[Step(index=0, tool="x", args={}, observation="o")],
                   metrics={"outcome": "survived", "months": 12, "month_log": [{"month": 0}]},
                   usage={"solver_cost": 0.1}, guidance_log=["secret guidance"])
    (row,) = episode_rows([t], graph="none")
    assert row == {"graph": "none", "task_id": "a", "score": 0.5, "success": True, "steps": 1, "error": None,
                   "outcome": "survived", "months": 12, "solver_cost": 0.1}


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(3, 3) == 1.0
    assert mcnemar_exact(11, 0) == pytest.approx(2 / 2**11)  # 0.00098: the EnterpriseArena expert-vs-none result
    diff, lo, hi = paired_bootstrap([1.0] * 5 + [0.0] * 5)
    assert diff == 0.5 and 0.1 <= lo < 0.5 < hi <= 0.9
    assert paired_bootstrap([0.2, 0.2, 0.2])[1:] == pytest.approx((0.2, 0.2))


def test_paired_comparison_drops_errors_and_pairs_by_task(tmp_path):
    write_run(tmp_path / "r1", {"none": [False, False, True, False], "graphs/g.json": [True, True, True, None]})
    (success, score, months) = compare(load([tmp_path / "r1"]), "none", ["success", "score", "months"])
    assert success["tasks"] == 3  # t3 errored under the graph, so it is not a pair
    assert (success["gained"], success["lost"], success["mcnemar_p"]) == (2, 0, 0.5)
    assert success["graph_mean"] == 1.0 and success["baseline_mean"] == pytest.approx(1 / 3)
    assert score["diff"] == pytest.approx(0.5) and score["mcnemar_p"] is None
    assert months["diff"] == pytest.approx(6.0)

    with pytest.raises(ValueError, match="baseline"):
        compare(load([tmp_path / "r1"]), "graphs/missing.json", ["success"])


def test_repeats_average_per_task_and_skip_mcnemar(tmp_path):
    write_run(tmp_path / "r1", {"none": [False, False], "graphs/g.json": [True, False]})
    write_run(tmp_path / "r2", {"none": [False, True], "graphs/g.json": [True, True]})
    (success,) = compare(load([tmp_path / "r1", tmp_path / "r2"]), "none", ["success"])
    assert success["repeats"] == 2 and success["mcnemar_p"] is None
    assert success["graph_mean"] == 0.75 and success["baseline_mean"] == 0.25 and success["diff"] == 0.5


def test_evaluate_logs_episode_rows_and_graph_hash(tmp_path, monkeypatch):
    class Env:
        name = "stub"

        def tasks(self, split):
            return [Task(id=f"t{i}", prompt="p") for i in range(2)]

    class Run:
        summary: dict = {}

    tables = {}
    monkeypatch.setattr(evl, "run_batch", lambda env, tasks, graph, cfg: [
        Trajectory(task_id=t.id, score=1.0, success=True, guidance_log=["text"]) for t in tasks])
    monkeypatch.setattr(evl, "log_table", lambda run, key, rows: tables.__setitem__(key, rows))
    graph = tmp_path / "g.json"
    graph.write_text('{"name": "g"}')

    rows = evl.evaluate(Env(), "test", None, ["none", str(graph)], Config(), tmp_path / "out", Run())

    assert rows[0]["graph_sha256"] is None and len(rows[1]["graph_sha256"]) == 12
    assert [(r["graph"], r["task_id"]) for r in tables["episodes"]] == [
        ("none", "t0"), ("none", "t1"), (str(graph), "t0"), (str(graph), "t1")]
    assert "text" not in json.dumps(tables["episodes"])  # guidance stays in the local trajectory files
    assert compare(load([tmp_path / "out"]), "none", ["success"])[0]["diff"] == 0.0
