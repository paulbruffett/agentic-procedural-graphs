"""Hand-written graphs must line up with the tools, since localization is an exact tool-name match."""
import random

import pytest

from pg.config import ROOT
from pg.envs.enterprisearena.data import make_task as ea_task
from pg.envs.enterprisearena.data import verify as ea_missing
from pg.envs.finance.env import FinanceEpisode
from pg.envs.finance.tasks import make_task, sample_params
from pg.envs.hotpotqa.env import HotpotQAEpisode
from pg.graph import NodeType, ProceduralGraph
from tests.test_hotpotqa import TASK


def ea_episode():
    from pg.envs.enterprisearena.env import EnterpriseArenaEpisode

    return EnterpriseArenaEpisode(ea_task(0, 3))


EPISODES = {
    "finance": lambda: FinanceEpisode(make_task("t", sample_params(random.Random(0)))),
    "hotpotqa": lambda: HotpotQAEpisode(TASK),
    "enterprisearena": ea_episode,
}
NEEDS_CFO_ENV = pytest.mark.skipif(bool(ea_missing()), reason="CFO-Env code not fetched")


@pytest.mark.parametrize("name", ["finance", "hotpotqa", pytest.param("enterprisearena", marks=NEEDS_CFO_ENV)])
def test_action_nodes_are_tools(name):
    g = ProceduralGraph.load(ROOT / "graphs" / f"{name}_expert.json")
    tools = {t.name for t in EPISODES[name]().tools}
    assert {n.id for n in g.nodes if n.type == NodeType.ACTION} == tools
    ids = {n.id for n in g.nodes}
    assert all(e.src in ids and e.dst in ids for e in g.edges)
