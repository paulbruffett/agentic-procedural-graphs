from langchain_core.messages import AIMessage

from pg.llm import add_usage, usage_of


def test_usage_of_handles_missing_or_null_usage():
    zero = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    for meta in ({}, {"token_usage": None}, {"token_usage": {"cost": None}}):
        assert usage_of(AIMessage(content="x", response_metadata=meta)) == zero


def test_usage_of_reads_tokens_and_cost():
    msg = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"token_usage": {"cost": 0.25}},
    )
    usage = usage_of(msg)
    assert usage == {"input_tokens": 10, "output_tokens": 5, "cost": 0.25}
    assert add_usage({"solver_cost": 1.0}, usage, "solver")["solver_cost"] == 1.25
