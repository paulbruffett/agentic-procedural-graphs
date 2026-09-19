"""See how a procedural graph changed over an `pg evolve` run.

    uv run python -m analysis.graph_evolution diff graphs/a.json graphs/b.json
    uv run python -m analysis.graph_evolution report graphs/evolved/<run>      # writes evolution.md + timeline.html

`report` reads what evolve already saves: round_k.json (the graph kept after round k), evolution_log.jsonl and,
for runs made since edits were recorded, edits/round_k.json (what the refiner proposed, accepted or not). Older
runs still work; their rejected rounds just cannot show the candidate that was tried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pg.graph import Edge, EditSet, GraphDiff, Node, NodeType, ProceduralGraph

TEMPLATE = Path(__file__).with_name("timeline_template.html")
FIELDS = ("condition", "guidance", "pitfalls")


def label(e: Edge) -> str:
    return f"{e.src} --{e.rel.value}--> {e.dst}"


# ---- text diff -----------------------------------------------------------------------------------------------
def text_diff(diff: GraphDiff) -> str:
    """Readable change list. Most of what a graph learns is in the edge text, so revised edges show each changed
    field before and after."""
    lines: list[str] = []
    lines += [f"+ node [{n.id}] ({n.type.value}) {n.description}".rstrip() for n in diff.added_nodes]
    lines += [f"- node [{n.id}]" for n in diff.removed_nodes]
    for old, new in diff.revised_nodes:
        lines += [f"~ node [{new.id}]", f"    was: ({old.type.value}) {old.description}", f"    now: ({new.type.value}) {new.description}"]
    for e in diff.added_edges:
        lines += [f"+ edge {label(e)}"] + [f"    {f}: {getattr(e, f)}" for f in FIELDS if getattr(e, f)]
    lines += [f"- edge {label(e)}" for e in diff.removed_edges]
    for old, new in diff.revised_edges:
        lines.append(f"~ edge {label(new)}")
        for f in FIELDS:
            if getattr(old, f) != getattr(new, f):
                lines += [f"    {f} was: {getattr(old, f)}", f"    {f} now: {getattr(new, f)}"]
    return "\n".join(lines) or "(no changes)"


# ---- one frame per round ---------------------------------------------------------------------------------------
def load_frames(run_dir: Path) -> list[dict]:
    """One frame per round: the graph that round *proposed* (or, when the proposal was not recorded, the graph it
    kept), with every node and edge marked added / revised / removed / kept relative to the graph before it."""
    log = [json.loads(line) for line in (run_dir / "evolution_log.jsonl").read_text().splitlines() if line.strip()]
    current = ProceduralGraph.load(run_dir / "round_0.json")
    frames = [_frame(log[0], current, GraphDiff(), recorded=True)]
    for entry in log[1:]:
        k = entry["round"]
        kept = ProceduralGraph.load(run_dir / f"round_{k}.json")
        edits_path = run_dir / "edits" / f"round_{k}.json"
        if edits_path.exists():
            shown, recorded = current.apply(EditSet.model_validate_json(edits_path.read_text()))[0], True
        else:  # run from before edits were saved: only accepted changes can be reconstructed
            shown, recorded = kept, bool(entry["accepted"]) or entry["val_after"] is None
        frames.append(_frame(entry, shown, current.diff(shown), recorded))
        current = kept
    return frames


def _frame(entry: dict, shown: ProceduralGraph, diff: GraphDiff, recorded: bool) -> dict:
    node_status = {n.id: "added" for n in diff.added_nodes} | {new.id: "revised" for _, new in diff.revised_nodes}
    edge_status = {e.key(): "added" for e in diff.added_edges} | {new.key(): "revised" for _, new in diff.revised_edges}
    before = {new.key(): old for old, new in diff.revised_edges}

    def node(n: Node, status: str) -> dict:
        return {"id": n.id, "type": n.type.value, "description": n.description, "status": status}

    def edge(e: Edge, status: str) -> dict:
        old = before.get(e.key())
        return {"src": e.src, "rel": e.rel.value, "dst": e.dst, "status": status,
                **{f: getattr(e, f) for f in FIELDS}, "was": {f: getattr(old, f) for f in FIELDS} if old else None}

    return {
        "round": entry["round"],
        "accepted": entry.get("accepted"),  # None for round 0
        "val_before": entry.get("val_before"),
        "val_after": entry.get("val_score", entry.get("val_after")),
        "summary": entry.get("edits", "initial graph"),
        "rationale": entry.get("rationale", ""),
        "recorded": recorded,
        "changes": text_diff(diff),
        "nodes": [node(n, node_status.get(n.id, "kept")) for n in shown.nodes] + [node(n, "removed") for n in diff.removed_nodes],
        "edges": [edge(e, edge_status.get(e.key(), "kept")) for e in shown.edges] + [edge(e, "removed") for e in diff.removed_edges],
    }


def verdict(frame: dict) -> str:
    if frame["accepted"] is None:
        return f"initial graph, val {frame['val_after']:.3f}"
    if frame["val_after"] is None:
        return "nothing validated (empty or no-op proposal, or the round failed)"
    outcome = "ACCEPTED" if frame["accepted"] else "rejected"
    return f"{outcome}: val {frame['val_before']:.3f} -> {frame['val_after']:.3f}"


# ---- mermaid ---------------------------------------------------------------------------------------------------
COLORS = {"added": "#2a9d8f", "revised": "#e9a23b", "removed": "#d1495b"}
SHAPES = {NodeType.STATE.value: '(["{}"])', NodeType.ACTION.value: '["{}"]', NodeType.REASONING.value: '{{{{"{}"}}}}'}


def to_mermaid(frame: dict) -> str:
    """Flowchart of one frame; this round's changes are coloured. Nodes that no edge touches are left out."""
    used = {e[end] for e in frame["edges"] for end in ("src", "dst")}
    nodes = [n for n in frame["nodes"] if n["id"] in used or n["status"] != "kept"]
    ids = {n["id"]: f"n{i}" for i, n in enumerate(nodes)}  # mermaid reserves words such as `end`
    lines = ["flowchart LR"]
    lines += [f"  {ids[n['id']]}{SHAPES[n['type']].format(n['id'])}" for n in nodes]
    styles = []
    for i, e in enumerate(edge for edge in frame["edges"] if edge["src"] in ids and edge["dst"] in ids):
        arrow = "-.->" if e["status"] == "removed" else "-->"
        lines.append(f"  {ids[e['src']]} {arrow}|{e['rel']}| {ids[e['dst']]}")
        if e["status"] in COLORS:
            styles.append(f"  linkStyle {i} stroke:{COLORS[e['status']]},stroke-width:3px")
    for status, color in COLORS.items():
        members = [ids[n["id"]] for n in nodes if n["status"] == status]
        if members:
            styles += [f"  classDef {status} stroke:{color},stroke-width:3px", f"  class {','.join(members)} {status}"]
    return "\n".join(lines + styles)


def to_markdown(run_dir: Path, frames: list[dict]) -> str:
    out = [f"# Graph evolution: {run_dir.name}", "",
           "Green = added this round, amber = revised, red dashed = removed. Generated by `analysis/graph_evolution.py`.", "",
           "| Round | Proposal | Outcome | Graph shown |", "|---|---|---|---|"]
    for f in frames:
        size = f"{sum(n['status'] != 'removed' for n in f['nodes'])} nodes, {sum(e['status'] != 'removed' for e in f['edges'])} edges"
        out.append(f"| {f['round']} | {f['summary'][:140]} | {verdict(f)} | {size} |")
    for f in frames:
        out += ["", f"## Round {f['round']}", "", f"**{verdict(f)}**", ""]
        if not f["recorded"]:
            out += ["*The rejected candidate was not recorded for this run; the graph below is the one that was kept.*", ""]
        if f["rationale"]:
            out += ["> " + f["rationale"].replace("\n", "\n> "), ""]
        if f["edges"]:
            out += ["```mermaid", to_mermaid(f), "```", ""]
        if f["accepted"] is not None:
            out += ["```diff", f["changes"], "```"]
    return "\n".join(out) + "\n"


# ---- outputs ---------------------------------------------------------------------------------------------------
def report(run_dir: Path, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(run_dir)
    data = json.dumps({"run": run_dir.name, "frames": frames}).replace("</", "<\\/")
    written = [out_dir / "evolution.md", out_dir / "timeline.html"]
    written[0].write_text(to_markdown(run_dir, frames))
    written[1].write_text(TEMPLATE.read_text().replace("/*DATA*/null", data))
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="what changed between two graph files")
    d.add_argument("before", type=Path)
    d.add_argument("after", type=Path)
    r = sub.add_parser("report", help="write evolution.md and timeline.html for an evolve run directory")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--out", type=Path, default=None, help="default: the run directory itself")
    args = p.parse_args(argv)
    if args.cmd == "diff":
        print(text_diff(ProceduralGraph.load(args.before).diff(ProceduralGraph.load(args.after))))
    else:
        for path in report(args.run_dir, args.out):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
