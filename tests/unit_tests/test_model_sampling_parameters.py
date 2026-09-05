"""``init_model`` must not send ``top_p`` to a model that rejects it.

gpt-5.6-luna answers every request carrying ``top_p`` with HTTP 400, which
silently broke the baseline retrain (every generation call failed) and would
break every avatar reply the moment MODEL pointed at luna.
"""

from src.anubis.utils.model import (
    REASONING_EFFORT_NONE_MODEL_PREFIXES,
    TOP_P_UNSUPPORTED_MODEL_PREFIXES,
    openai_sampling_parameters,
)


def test_models_that_accept_top_p_keep_the_established_sampling_regime():
    assert openai_sampling_parameters("gpt-5.4-nano") == {"temperature": 0.1, "top_p": 0.1}
    assert openai_sampling_parameters("gpt-5-nano") == {"temperature": 0.1, "top_p": 0.1}


def test_luna_receives_temperature_and_reasoning_off_but_no_top_p():
    """Luna rejects top_p outright and rejects tool-bound calls unless reasoning is off."""
    assert "gpt-5.6-luna" in TOP_P_UNSUPPORTED_MODEL_PREFIXES
    assert "gpt-5.6-luna" in REASONING_EFFORT_NONE_MODEL_PREFIXES
    expected = {"temperature": 0.1, "reasoning_effort": "none"}
    assert openai_sampling_parameters("gpt-5.6-luna") == expected
    # Prefix match so a point release stays covered.
    assert openai_sampling_parameters("gpt-5.6-luna-2026-09-01") == expected


def test_unset_model_name_falls_back_to_the_full_regime():
    assert openai_sampling_parameters(None) == {"temperature": 0.1, "top_p": 0.1}
    assert openai_sampling_parameters("  ") == {"temperature": 0.1, "top_p": 0.1}
