"""search / lookup / finish over the example's own 10 context paragraphs."""
from __future__ import annotations

import difflib
from typing import Callable

from langchain_core.tools import BaseTool, tool

from pg.envs.hotpotqa.scoring import normalize_answer

STOPWORDS = set("of in on at to for by with from and or is was were are be been what which who whom when where how".split())


def _tokens(text: str) -> set[str]:
    return set(normalize_answer(text).split()) - STOPWORDS


def make_tools(paragraphs: list[dict], submit: Callable[[str], None]) -> list[BaseTool]:
    @tool
    def search(query: str) -> str:
        """Search the context paragraphs. Returns the 3 paragraphs with the highest word overlap with the query."""
        q = _tokens(query)
        scored = sorted(paragraphs, key=lambda p: len(q & _tokens(p["title"] + " " + p["text"])), reverse=True)
        top = [p for p in scored[:3] if q & _tokens(p["title"] + " " + p["text"])]
        if not top:
            return "No paragraphs matched. Try different keywords."
        return "\n\n".join(f"[{p['title']}] {p['text']}" for p in top)

    @tool
    def lookup(title: str) -> str:
        """Return the full paragraph with the given title."""
        by_title = {p["title"].lower(): p for p in paragraphs}
        p = by_title.get(title.strip().lower())
        if p is None:
            close = difflib.get_close_matches(title.strip().lower(), list(by_title), n=1, cutoff=0.6)
            p = by_title[close[0]] if close else None
        if p is None:
            return "No paragraph with that title. Available titles: " + "; ".join(x["title"] for x in paragraphs)
        return f"[{p['title']}] {p['text']}"

    @tool
    def finish(answer: str) -> str:
        """Submit the final answer (a short span such as an entity name, a date, or yes/no). Ends the episode."""
        submit(answer)
        return "Answer submitted."

    return [search, lookup, finish]
