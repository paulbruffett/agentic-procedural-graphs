from pg.trajectory import Step, Trajectory, read_jsonl, summarize, write_jsonl


def test_jsonl_roundtrip_and_summary(tmp_path):
    ts = [
        Trajectory(task_id="t1", steps=[Step(index=0, tool="search", args={"q": "x"}, observation="obs")], score=1.0,
                   success=True, usage={"solver_input_tokens": 10}),
        Trajectory(task_id="t2", score=0.0, usage={"solver_input_tokens": 5}, error="boom"),
    ]
    write_jsonl(tmp_path / "t.jsonl", ts)
    assert read_jsonl(tmp_path / "t.jsonl") == ts

    s = summarize(ts)
    # the errored trajectory is counted but kept out of the averages
    assert s["n"] == 2 and s["n_scored"] == 1 and s["errors"] == 1
    assert s["mean_score"] == 1.0 and s["success_rate"] == 1.0
    assert s["solver_input_tokens"] == 15

    text = ts[0].compact()
    assert 'search({"q": "x"}) -> obs' in text
