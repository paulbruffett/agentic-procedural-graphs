"""analysis/graph_evolution.py: frames, text diff, Mermaid and the report files (no network)."""
import json

from analysis.graph_evolution import load_frames, report, text_diff, to_mermaid
from pg.graph import Edge, EdgeRef, EditSet, Node, NodeType, ProceduralGraph, Relation

E1 = Edge(src="Start", rel=Relation.LEADS_TO, dst="search", guidance="search first")


def write_run(run_dir, with_edits: bool):
    """Round 1 adds an edge (accepted); round 2 proposes a revision, a new node and a removal (rejected)."""
    g0 = ProceduralGraph.skeleton("demo", [("search", "find"), ("finish", "answer")])
    g1 = g0.apply(EditSet(add_edges=[E1, Edge(src="search", rel=Relation.LEADS_TO, dst="finish")]))[0]
    rejected = EditSet(rationale="try a check", add_nodes=[Node(id="End", type=NodeType.STATE, description="done!")],
                       add_edges=[E1.model_copy(update={"guidance": "search the entity"})],
                       delete_edges=[EdgeRef(src="search", rel=Relation.LEADS_TO, dst="finish")])
    for k, g in enumerate([g0, g1, g1]):
        g.save(run_dir / f"round_{k}.json")
    if with_edits:
        (run_dir / "edits").mkdir()
        (run_dir / "edits" / "round_2.json").write_text(rejected.model_dump_json())
    log = [{"round": 0, "val_score": 0.5, "graph": g0.summary()},
           {"round": 1, "val_before": 0.5, "val_after": 0.7, "accepted": True, "edits": "add edges", "rationale": "backbone"},
           {"round": 2, "val_before": 0.7, "val_after": 0.6, "accepted": False, "edits": "revise", "rationale": "try a check"}]
    (run_dir / "evolution_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))


def test_frames_show_the_rejected_proposal_when_edits_were_saved(tmp_path):
    write_run(tmp_path, with_edits=True)
    f0, f1, f2 = load_frames(tmp_path)
    assert f0["edges"] == [] and f0["accepted"] is None
    assert [e["status"] for e in f1["edges"]] == ["added", "added"]
    status = {(e["src"], e["dst"]): e["status"] for e in f2["edges"]}
    assert status == {("Start", "search"): "revised", ("search", "finish"): "removed"} and f2["recorded"]
    revised = next(e for e in f2["edges"] if e["status"] == "revised")
    assert revised["was"]["guidance"] == "search first" and revised["guidance"] == "search the entity"
    assert {n["id"]: n["status"] for n in f2["nodes"]}["End"] == "revised"  # same id, new description
    assert "guidance was: search first" in f2["changes"] and "- edge search --LEADS_TO--> finish" in f2["changes"]


def test_older_runs_without_saved_edits_fall_back_to_the_kept_graph(tmp_path):
    write_run(tmp_path, with_edits=False)
    f2 = load_frames(tmp_path)[2]
    assert not f2["recorded"] and {e["status"] for e in f2["edges"]} == {"kept"}


def test_mermaid_and_report_files(tmp_path):
    write_run(tmp_path, with_edits=True)
    mermaid = to_mermaid(load_frames(tmp_path)[2])
    assert mermaid.startswith("flowchart LR") and "-.->|LEADS_TO|" in mermaid  # removed edge is dashed
    assert "linkStyle 0 stroke:#e9a23b" in mermaid and "linkStyle 1 stroke:#d1495b" in mermaid
    assert '"End"' in mermaid and "| End" not in mermaid  # reserved word only ever appears as a label

    md, html = report(tmp_path, tmp_path / "out")
    assert "rejected: val 0.700 -> 0.600" in md.read_text() and "```mermaid" in md.read_text()
    page = html.read_text()
    assert "/*DATA*/null" not in page and '"rationale": "try a check"' in page


def test_text_diff_of_identical_graphs():
    g = ProceduralGraph.skeleton("demo", [("search", "find")])
    assert text_diff(g.diff(g)) == "(no changes)"
