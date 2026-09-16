from pg.envs.base import Task
from pg.envs.hotpotqa.env import HotpotQAEpisode
from pg.envs.hotpotqa.scoring import exact_match, f1_score, normalize_answer

TASK = Task(
    id="q1",
    prompt="Question: Which city is the capital of the country where the Eiffel Tower stands?",
    meta={
        "answer": "Paris",
        "paragraphs": [
            {"title": "Eiffel Tower", "text": "The Eiffel Tower is a lattice tower in France."},
            {"title": "France", "text": "France is a country whose capital is Paris."},
            {"title": "Berlin", "text": "Berlin is the capital of Germany."},
        ],
    },
)


def tools(ep):
    return {t.name: t for t in ep.tools}


def test_normalize_em_f1():
    assert normalize_answer("The  Paris!") == "paris"
    assert exact_match("the paris", "Paris") == 1.0
    assert f1_score("Paris France", "Paris") == 2 * 0.5 * 1 / 1.5
    assert f1_score("yes", "no") == 0.0
    assert f1_score("London", "Paris") == 0.0


def test_search_lookup_finish():
    ep = HotpotQAEpisode(TASK)
    t = tools(ep)
    assert t["search"].invoke({"query": "Eiffel Tower"}).startswith("[Eiffel Tower]")
    assert "No paragraphs matched" in t["search"].invoke({"query": "zzz"})
    assert "capital is Paris" in t["lookup"].invoke({"title": "france"})
    assert "Available titles" in t["lookup"].invoke({"title": "Spain"})
    assert not ep.is_done()
    t["finish"].invoke({"answer": " Paris "})
    assert ep.is_done()
    s = ep.score()
    assert s["score"] == 1.0 and s["success"] and s["em"] == 1.0


def test_on_text_accepts_final_answer():
    ep = HotpotQAEpisode(TASK)
    assert ep.on_text("Berlin") is True
    assert ep.is_done() and ep.score()["score"] == 0.0
