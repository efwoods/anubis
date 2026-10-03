"""``response_metadata["turn_cost"]``: the reply plus every vision and triage call.

The reply's own ``total_cost`` and ``token_usage`` stay reply-only because the
message endpoint bills those two fields; ``turn_cost`` adds the pending image
descriptions and triage calls for display only.
"""

import pytest
from langchain_core.messages import AIMessage

from src.anubis.utils.billing.turn_cost import (
    INFERENCE_TYPE_AMBIENT_TRIAGE,
    INFERENCE_TYPE_IMAGE_DESCRIPTION,
    attach_turn_cost_breakdown,
    build_turn_cost_breakdown,
    pending_turn_cost_item,
)

REPLY_METADATA = {
    "token_usage": {
        "prompt_tokens": 40000,
        "completion_tokens": 120,
        "total_tokens": 40120,
        "cached_prompt_tokens": 32000,
    },
    "total_cost": 0.004,
}


def _pending_items():
    return [
        pending_turn_cost_item(
            INFERENCE_TYPE_IMAGE_DESCRIPTION,
            model_name="gpt-5-nano",
            prompt_tokens=4049,
            completion_tokens=1612,
            cost_usd=0.0024199,
            latency_ms=11790.0,
            observation_id="obs-1",
            source="screen",
        ),
        pending_turn_cost_item(
            INFERENCE_TYPE_AMBIENT_TRIAGE,
            model_name="gpt-5.6-luna",
            prompt_tokens=3000,
            completion_tokens=200,
            cached_prompt_tokens=1000,
            cost_usd=0.00084,
            latency_ms=812.5,
            observation_id="obs-1",
            source="ambient_triage",
        ),
        pending_turn_cost_item(
            INFERENCE_TYPE_IMAGE_DESCRIPTION,
            model_name="gpt-5-nano",
            prompt_tokens=4049,
            completion_tokens=1103,
            cost_usd=0.00178365,
            observation_id="obs-2",
            source="screen",
        ),
    ]


def test_the_breakdown_totals_the_reply_and_every_pending_call():
    turn_cost = build_turn_cost_breakdown(
        REPLY_METADATA, _pending_items(), reply_model_name="gpt-5.6-luna"
    )

    assert turn_cost["total_cost_usd"] == pytest.approx(
        0.004 + 0.0024199 + 0.00084 + 0.00178365
    )
    assert turn_cost["total_tokens"] == 40120 + (4049 + 1612) + (3000 + 200) + (
        4049 + 1103
    )
    assert turn_cost["reply"] == {
        "model_name": "gpt-5.6-luna",
        "prompt_tokens": 40000,
        "completion_tokens": 120,
        "cached_prompt_tokens": 32000,
        "cost_usd": 0.004,
    }
    image_descriptions = turn_cost["image_descriptions"]
    assert image_descriptions["count"] == 2
    assert image_descriptions["prompt_tokens"] == 8098
    assert image_descriptions["completion_tokens"] == 2715
    assert image_descriptions["cost_usd"] == pytest.approx(0.0024199 + 0.00178365)
    ambient_triage = turn_cost["ambient_triage"]
    assert ambient_triage["count"] == 1
    assert ambient_triage["cached_prompt_tokens"] == 1000
    assert ambient_triage["cost_usd"] == pytest.approx(0.00084)
    assert [item["observation_id"] for item in turn_cost["items"]] == [
        "obs-1",
        "obs-1",
        "obs-2",
    ]


def test_the_billed_reply_fields_stay_reply_only():
    reply = AIMessage(content="Hello", response_metadata=dict(REPLY_METADATA))

    attach_turn_cost_breakdown(reply, _pending_items())

    assert reply.response_metadata["total_cost"] == 0.004
    assert reply.response_metadata["token_usage"] == REPLY_METADATA["token_usage"]
    assert reply.response_metadata["turn_cost"]["total_cost_usd"] > 0.004


def test_a_reply_with_no_pending_calls_reports_the_reply_cost_alone():
    reply = AIMessage(content="Hello", response_metadata=dict(REPLY_METADATA))

    attach_turn_cost_breakdown(reply, [])

    turn_cost = reply.response_metadata["turn_cost"]
    assert turn_cost["total_cost_usd"] == 0.004
    assert turn_cost["total_tokens"] == 40120
    assert turn_cost["image_descriptions"]["count"] == 0
    assert turn_cost["ambient_triage"]["count"] == 0
    assert turn_cost["items"] == []
