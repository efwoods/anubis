"""The message path's dollar cost is derived from the configured per-token rates.

``MODEL_PROMPT_COST`` / ``MODEL_COMPLETION_COST`` were declared but never read, so
every reply reached the Prometheus cost counter and ``api_metrics.cost_usd`` as 0.
``_attach_token_usage_metadata`` now folds them into ``response_metadata["total_cost"]``.
"""

import pytest
from langchain_core.messages import AIMessage

from src.anubis.graph import _attach_token_usage_metadata
from src.anubis.utils.context import GlobalContext


def _message(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="reply",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _context() -> GlobalContext:
    # Non-default values: ``__post_init__`` falls back to the environment for any
    # field still at its default, and the env may carry the real rates.
    return GlobalContext(model_prompt_cost=0.000001, model_completion_cost=0.000002)


def test_total_cost_is_prompt_and_completion_tokens_at_their_rates():
    final = _message(100, 50)
    _attach_token_usage_metadata(final, [_message(300, 10), final], context=_context())
    token_usage = final.response_metadata["token_usage"]
    assert token_usage == {"prompt_tokens": 400, "completion_tokens": 60, "total_tokens": 460}
    assert final.response_metadata["total_cost"] == pytest.approx(
        400 * 0.000001 + 60 * 0.000002
    )


def test_without_a_context_no_cost_is_written():
    final = _message(100, 50)
    _attach_token_usage_metadata(final, [final])
    assert "token_usage" in final.response_metadata
    assert "total_cost" not in final.response_metadata


def test_zero_token_turn_writes_nothing():
    final = AIMessage(content="reply")
    _attach_token_usage_metadata(final, [final], context=_context())
    assert "token_usage" not in (final.response_metadata or {})
    assert "total_cost" not in (final.response_metadata or {})
