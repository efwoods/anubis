"""Structured-output classifier that reconciles diarizer speaker labels."""

import json
from time import time_ns
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    SPEAKER_RECONCILIATION_SYSTEM_PROMPT,
    SpeakerReconciliation,
)
from src.anubis.utils.tokenizer import count_tokens


class SpeakerReconciliationClass:
    """Correct diarizer over/under-splitting of speakers via structured output.

    ``gpt-4o-transcribe-diarize`` exposes no parameter to set the number of
    speakers and commonly splits one person across multiple labels. Given the
    coalesced, labeled turns and the known target name, this maps each raw
    diarizer label to a canonical speaker (the target relabeled ``avatar``).
    """

    def __init__(self):
        """Initialize the model, prompt, and pricing/metric metadata."""
        self.model = init_model(response_format=SpeakerReconciliation)
        self.system_message = SystemMessage(
            content=SPEAKER_RECONCILIATION_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = count_tokens(SPEAKER_RECONCILIATION_SYSTEM_PROMPT)
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "speaker_reconciliation_structured_output"

    @staticmethod
    def _render_turns(
        turns: List[Dict[str, Any]], target_speaker_label: str | None
    ) -> str:
        """Render turns as ``[raw_label] text`` lines plus the target hint."""
        lines: List[str] = []
        if target_speaker_label:
            lines.append(f"TARGET speaker name/label: {target_speaker_label}")
        lines.append("Transcript turns (raw diarizer label, then text):")
        for turn in turns:
            speaker = str(turn.get("speaker") or "unknown")
            text = (turn.get("text") or "").strip()
            lines.append(f"[{speaker}] {text}")
        return "\n".join(lines)

    async def reconcile(
        self,
        turns: List[Dict[str, Any]],
        target_speaker_label: str | None = None,
    ):
        """Return the reconciliation dict (label_map + metrics) for the turns."""
        start_time = time_ns()
        input_str = self._render_turns(turns, target_speaker_label)
        human_message = HumanMessage(content=input_str)
        input_tokens = count_tokens(input_str) + self.system_prompt_tokens
        messages = [self.system_message, human_message]
        response = await self.model.ainvoke(messages)
        response_dict = response.model_dump()

        output_tokens = count_tokens(json.dumps(response_dict))
        total_tokens = input_tokens + output_tokens

        total_cost = (
            input_tokens * self.model_input_token_cost_per_million
            + output_tokens * self.model_output_token_cost_per_million
        )

        response_dict.update({"input_tokens": input_tokens})
        response_dict.update({"output_tokens": output_tokens})
        response_dict.update({"total_tokens": total_tokens})
        response_dict.update({"model_name": self.model_name})
        response_dict.update({"total_cost": total_cost})
        response_dict.update({"model_inference_type": self.model_inference_type})
        end_time = time_ns()
        duration_ms = (end_time - start_time) / 1e6
        response_dict.update({"latency_ms": duration_ms})
        return response_dict
