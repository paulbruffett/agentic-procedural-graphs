"""The refiner falls back to plain JSON only for schema problems, never for transport errors."""
import json

import pytest
from langchain_core.messages import AIMessage

from pg.refiner import propose_edits

EDITS = {"rationale": "r", "add_nodes": [], "add_edges": [], "delete_nodes": [], "delete_edges": []}


class FakeStructured:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, messages):
        raise self.exc


class FakeLLM:
    def __init__(self, exc, reply=""):
        self.exc, self.reply, self.calls = exc, reply, 0

    def with_structured_output(self, schema, **kwargs):
        return FakeStructured(self.exc)

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)


def test_schema_failure_falls_back_to_plain_json():
    llm = FakeLLM(ValueError("could not parse"), "here you go\n" + json.dumps(EDITS))
    edits, usage = propose_edits(llm, "prompt")
    assert edits.rationale == "r" and llm.calls == 1 and usage["cost"] == 0.0


def test_transport_errors_are_not_retried_as_json():
    llm = FakeLLM(RuntimeError("429 rate limited"))
    with pytest.raises(RuntimeError):
        propose_edits(llm, "prompt")
    assert llm.calls == 0  # the prompt is not paid for twice
