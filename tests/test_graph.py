from pg.graph import Edge, EdgeRef, EditSet, Node, NodeType, ProceduralGraph, Relation


def chain() -> ProceduralGraph:
    """Start -> a -> b -> c -> End, plus an isolated node z."""
    ids = ["Start", "a", "b", "c", "End", "z"]
    nodes = [Node(id=i, type=NodeType.STATE if i in ("Start", "End") else NodeType.ACTION) for i in ids]
    pairs = [("Start", "a"), ("a", "b"), ("b", "c"), ("c", "End")]
    edges = [Edge(src=s, rel=Relation.LEADS_TO, dst=d, guidance=f"{s} then {d}") for s, d in pairs]
    return ProceduralGraph(name="chain", nodes=nodes, edges=edges)


def test_neighborhood_two_hops_both_directions():
    sub = chain().neighborhood("b", 2)
    assert {n.id for n in sub.nodes} == {"Start", "a", "b", "c", "End"}
    sub1 = chain().neighborhood("b", 1)
    assert {n.id for n in sub1.nodes} == {"a", "b", "c"}
    assert {e.key() for e in sub1.edges} == {("a", "LEADS_TO", "b"), ("b", "LEADS_TO", "c")}


def test_neighborhood_unknown_node_returns_full_graph():
    g = chain()
    assert g.neighborhood("nope", 2) is g


def test_serialize_marks_active_node_and_attributes():
    text = chain().serialize(active="a")
    assert "Active node: [a] (ACTION)" in text
    assert "[a] --LEADS_TO--> [b]" in text
    assert "guidance: a then b" in text
    assert "not yet localized" in chain().serialize()


def test_apply_add_delete_revise_is_pure():
    g = chain()
    edits = EditSet(
        add_nodes=[Node(id="check", type=NodeType.REASONING)],
        add_edges=[
            Edge(src="b", rel=Relation.TRIGGERS, dst="check"),
            Edge(src="a", rel=Relation.LEADS_TO, dst="b", guidance="revised"),
            Edge(src="ghost", rel=Relation.LEADS_TO, dst="b"),
        ],
        delete_nodes=["c"],
        delete_edges=[EdgeRef(src="Start", rel=Relation.LEADS_TO, dst="a"), EdgeRef(src="x", rel=Relation.LEADS_TO, dst="y")],
    )
    new, warnings = g.apply(edits)

    keys = {e.key() for e in new.edges}
    assert ("Start", "LEADS_TO", "a") not in keys
    assert ("b", "LEADS_TO", "c") not in keys and ("c", "LEADS_TO", "End") not in keys  # dangling edges dropped
    assert ("b", "TRIGGERS", "check") in keys
    assert next(e for e in new.edges if e.key() == ("a", "LEADS_TO", "b")).guidance == "revised"
    assert new.has_node("check") and not new.has_node("c")
    assert len(warnings) == 2  # ghost endpoint + missing delete

    # original untouched
    assert g.has_node("c") and len(g.edges) == 4
    assert next(e for e in g.edges if e.key() == ("a", "LEADS_TO", "b")).guidance == "a then b"


def test_save_load_roundtrip(tmp_path):
    g = chain()
    g.save(tmp_path / "g.json")
    assert ProceduralGraph.load(tmp_path / "g.json") == g


def test_skeleton_has_no_edges():
    g = ProceduralGraph.skeleton("s", [("search", "find"), ("finish", "answer")])
    assert [n.id for n in g.nodes] == ["Start", "End", "search", "finish"]
    assert g.edges == []


def test_diff_reports_added_removed_revised_and_ignores_order():
    g = chain()
    assert not g.diff(g)
    shuffled = ProceduralGraph(name="chain", nodes=g.nodes[::-1], edges=g.edges[::-1])
    assert not g.diff(shuffled)  # same content in another order is not a change

    edits = EditSet(
        add_nodes=[Node(id="check", type=NodeType.REASONING)],
        add_edges=[Edge(src="a", rel=Relation.LEADS_TO, dst="check"),
                   Edge(src="a", rel=Relation.LEADS_TO, dst="b", guidance="revised")],
        delete_edges=[EdgeRef(src="c", rel=Relation.LEADS_TO, dst="End")],
        delete_nodes=["z"],
    )
    d = g.diff(g.apply(edits)[0])
    assert d and [n.id for n in d.added_nodes] == ["check"] and [n.id for n in d.removed_nodes] == ["z"]
    assert [e.key() for e in d.added_edges] == [("a", "LEADS_TO", "check")]
    assert [e.key() for e in d.removed_edges] == [("c", "LEADS_TO", "End")]
    ((old, new),) = d.revised_edges
    assert (old.guidance, new.guidance) == ("a then b", "revised")
