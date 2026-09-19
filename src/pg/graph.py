"""Procedural graph: G = (V, R, E, Phi).

Nodes abstract tool functions, reasoning steps, or task states. Edges are
(procedure, relation, procedure) triplets carrying three textual attributes:
condition (when the transition applies), guidance (how to proceed) and
pitfalls (what to avoid).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    ACTION = "ACTION"  # a tool call
    REASONING = "REASONING"  # an internal reasoning step
    STATE = "STATE"  # a task status such as Start / End


class Relation(str, Enum):
    LEADS_TO = "LEADS_TO"
    TRIGGERS = "TRIGGERS"
    PROVIDES_INPUT_FOR = "PROVIDES_INPUT_FOR"
    CONVERGES_TO = "CONVERGES_TO"


class Node(BaseModel):
    id: str = Field(description="Unique node id. For ACTION nodes this must equal the tool name.")
    type: NodeType = NodeType.ACTION
    description: str = ""


class EdgeRef(BaseModel):
    src: str
    rel: Relation
    dst: str


class Edge(EdgeRef):
    condition: str = Field(default="", description="When this transition applies.")
    guidance: str = Field(default="", description="How to proceed along this transition.")
    pitfalls: str = Field(default="", description="What to avoid on this transition.")

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel.value, self.dst)


class EditSet(BaseModel):
    """A structural mutation proposed by the refiner. Revising an edge's attributes is
    expressed by re-adding an edge with the same (src, rel, dst) key."""

    rationale: str = Field(default="", description="Why these edits should help, grounded in the trajectories.")
    add_nodes: list[Node] = Field(default_factory=list)
    add_edges: list[Edge] = Field(default_factory=list)
    delete_nodes: list[str] = Field(default_factory=list)
    delete_edges: list[EdgeRef] = Field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.add_nodes:
            parts.append("add nodes: " + ", ".join(n.id for n in self.add_nodes))
        if self.add_edges:
            parts.append("add/revise edges: " + ", ".join(f"{e.src}-{e.rel.value}->{e.dst}" for e in self.add_edges))
        if self.delete_nodes:
            parts.append("delete nodes: " + ", ".join(self.delete_nodes))
        if self.delete_edges:
            parts.append("delete edges: " + ", ".join(f"{e.src}-{e.rel.value}->{e.dst}" for e in self.delete_edges))
        return "; ".join(parts) or "no-op"

    def is_empty(self) -> bool:
        return not (self.add_nodes or self.add_edges or self.delete_nodes or self.delete_edges)


@dataclass
class GraphDiff:
    """What changed from one graph to another. Revised = same id / (src, rel, dst) key, different content;
    each revised entry is an (old, new) pair. Falsy when nothing changed."""

    added_nodes: list[Node] = field(default_factory=list)
    removed_nodes: list[Node] = field(default_factory=list)
    revised_nodes: list[tuple[Node, Node]] = field(default_factory=list)
    added_edges: list[Edge] = field(default_factory=list)
    removed_edges: list[Edge] = field(default_factory=list)
    revised_edges: list[tuple[Edge, Edge]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return any(vars(self).values())


class ProceduralGraph(BaseModel):
    name: str = ""
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    # ---- basic access -------------------------------------------------
    def node(self, node_id: str) -> Node | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def has_node(self, node_id: str) -> bool:
        return self.node(node_id) is not None

    def summary(self) -> str:
        return f"{len(self.nodes)} nodes, {len(self.edges)} edges"

    # ---- persistence --------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ProceduralGraph":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="json"), indent=2))

    # ---- construction -------------------------------------------------
    @classmethod
    def skeleton(cls, name: str, tool_descriptions: list[tuple[str, str]]) -> "ProceduralGraph":
        """Minimal scratch graph: Start/End states plus one ACTION node per tool, no edges.
        The self-evolution loop has to discover the procedural backbone itself."""
        nodes = [
            Node(id="Start", type=NodeType.STATE, description="Beginning of the task."),
            Node(id="End", type=NodeType.STATE, description="Task complete."),
        ] + [Node(id=t, type=NodeType.ACTION, description=d) for t, d in tool_descriptions]
        return cls(name=name, nodes=nodes, edges=[])

    # ---- localization -------------------------------------------------
    def neighborhood(self, node_id: str, h: int) -> "ProceduralGraph":
        """Induced subgraph within h hops of node_id, following edges in both directions."""
        if not self.has_node(node_id):
            return self
        seen = {node_id}
        frontier = deque([(node_id, 0)])
        while frontier:
            cur, d = frontier.popleft()
            if d >= h:
                continue
            for e in self.edges:
                for nxt in (e.dst if e.src == cur else None, e.src if e.dst == cur else None):
                    if nxt and nxt not in seen:
                        seen.add(nxt)
                        frontier.append((nxt, d + 1))
        return ProceduralGraph(
            name=self.name,
            nodes=[n for n in self.nodes if n.id in seen],
            edges=[e for e in self.edges if e.src in seen and e.dst in seen],
        )

    # ---- serialization for the guidance / refiner prompts --------------
    def serialize(self, active: str | None = None) -> str:
        lines = [f'Procedural graph "{self.name}" ({self.summary()})']
        if active and self.has_node(active):
            n = self.node(active)
            lines.append(f"Active node: [{n.id}] ({n.type.value}) {n.description}".rstrip())
        else:
            lines.append("Active node: none (agent not yet localized; full graph shown)")
        lines.append("Nodes:")
        for n in self.nodes:
            lines.append(f"  [{n.id}] ({n.type.value}) {n.description}".rstrip())
        lines.append("Transitions:")
        if not self.edges:
            lines.append("  (none)")
        for e in self.edges:
            lines.append(f"  [{e.src}] --{e.rel.value}--> [{e.dst}]")
            if e.condition:
                lines.append(f"      condition: {e.condition}")
            if e.guidance:
                lines.append(f"      guidance: {e.guidance}")
            if e.pitfalls:
                lines.append(f"      pitfalls: {e.pitfalls}")
        return "\n".join(lines)

    # ---- comparison ---------------------------------------------------
    def diff(self, other: "ProceduralGraph") -> GraphDiff:
        """Changes that turn this graph into `other`, ignoring the order of nodes and edges."""
        d = GraphDiff()
        for mine, theirs, added, removed, revised in (
            ({n.id: n for n in self.nodes}, {n.id: n for n in other.nodes}, d.added_nodes, d.removed_nodes, d.revised_nodes),
            ({e.key(): e for e in self.edges}, {e.key(): e for e in other.edges}, d.added_edges, d.removed_edges, d.revised_edges),
        ):
            added += [v for k, v in theirs.items() if k not in mine]
            removed += [v for k, v in mine.items() if k not in theirs]
            revised += [(v, theirs[k]) for k, v in mine.items() if k in theirs and theirs[k] != v]
        return d

    # ---- mutation -----------------------------------------------------
    def apply(self, edits: EditSet) -> tuple["ProceduralGraph", list[str]]:
        """Return a new graph with the edits applied; the original is untouched.
        Deletes run before adds so delete+add of the same key acts as a revision.
        Edges pointing at unknown nodes are dropped and reported as warnings."""
        warnings: list[str] = []
        nodes = {n.id: n.model_copy() for n in self.nodes}
        edges = {e.key(): e.model_copy() for e in self.edges}

        for ref in edits.delete_edges:
            if edges.pop((ref.src, ref.rel.value, ref.dst), None) is None:
                warnings.append(f"delete_edge: no such edge {ref.src}-{ref.rel.value}->{ref.dst}")
        for nid in edits.delete_nodes:
            if nodes.pop(nid, None) is None:
                warnings.append(f"delete_node: no such node {nid}")
            for k in [k for k in edges if k[0] == nid or k[2] == nid]:
                edges.pop(k)
        for n in edits.add_nodes:
            nodes[n.id] = n.model_copy()
        for e in edits.add_edges:
            if e.src not in nodes or e.dst not in nodes:
                warnings.append(f"add_edge: unknown endpoint in {e.src}-{e.rel.value}->{e.dst}; skipped")
                continue
            edges[e.key()] = e.model_copy()

        return ProceduralGraph(name=self.name, nodes=list(nodes.values()), edges=list(edges.values())), warnings
