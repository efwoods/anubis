"""CSV column identification for avatar-identity ingest preprocessing.

Used by the ``/update_avatar_identity_with_media`` endpoint to pick out, for
an uploaded tabular file, (a) the single column whose cells hold the verbatim
free-text statements from the target individual, and (b) where the target's
name comes from — either a column with the name, a dominant value across
rows, or the filename itself.

The class wraps a structured-output LLM call against
:class:`~src.anubis.utils.schema.CSVUserTextColumnIdentification` and returns a
plain ``dict`` with the model fields plus run/cost telemetry, matching the
pattern used by the other classifier classes
(``ContentSituationClassificationClass``,
``ReferenceDocumentClassificationClass``, etc.).
"""

import json
from time import time_ns
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.anubis.utils.model import init_model
from src.anubis.utils.schema import (
    CSVUserTextColumnIdentification,
    CSV_USER_TEXT_COLUMN_IDENTIFICATION_SYSTEM_PROMPT,
)
from src.anubis.utils.tokenizer import count_tokens


class CSVUserTextColumnIdentificationClass:
    """Pick the user-text column and target-name source from a CSV header set."""

    def __init__(self):
        self.model = init_model(response_format=CSVUserTextColumnIdentification)
        self.system_message = SystemMessage(
            content=CSV_USER_TEXT_COLUMN_IDENTIFICATION_SYSTEM_PROMPT
        )
        self.system_prompt_tokens = count_tokens(
            CSV_USER_TEXT_COLUMN_IDENTIFICATION_SYSTEM_PROMPT
        )
        self.model_name = "gpt-5-nano"
        self.model_input_token_cost_per_million = 0.00000005
        self.model_output_token_cost_per_million = 0.0000004
        self.model_inference_type = "csv_user_text_column_identification"

    @staticmethod
    def build_user_message(
        *,
        filename: str,
        headers: List[str],
        sample_rows: List[Dict[str, Any]],
        column_stats: Dict[str, Dict[str, Any]],
    ) -> str:
        """Render the structured human prompt the model sees.

        ``column_stats`` is keyed by header and carries per-column summaries
        (avg_len, distinct_count, sample_values) so the model can pick the
        free-text column and the dominant-name column without reading every
        row of the file.
        """
        return json.dumps(
            {
                "filename": filename,
                "headers": list(headers),
                "column_stats": column_stats,
                "sample_rows": sample_rows,
            },
            ensure_ascii=False,
            default=str,
        )

    async def classify(
        self,
        *,
        filename: str,
        headers: List[str],
        sample_rows: List[Dict[str, Any]],
        column_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the model once and return its dict + telemetry."""
        start_time = time_ns()

        user_payload = self.build_user_message(
            filename=filename,
            headers=headers,
            sample_rows=sample_rows,
            column_stats=column_stats,
        )
        human_message = HumanMessage(content=user_payload)
        input_tokens = count_tokens(user_payload) + self.system_prompt_tokens

        response = await self.model.ainvoke([self.system_message, human_message])
        response_dict: Dict[str, Any] = response.model_dump()

        chosen_text_col = response_dict.get("text_column") or ""
        if chosen_text_col not in headers:
            response_dict["text_column"] = _fallback_text_column(
                headers=headers, column_stats=column_stats
            )
        chosen_target_col: Optional[str] = response_dict.get("target_name_column")
        if chosen_target_col is not None and chosen_target_col not in headers:
            response_dict["target_name_column"] = None

        output_tokens = count_tokens(json.dumps(response_dict))
        total_tokens = input_tokens + output_tokens
        total_cost = (
            input_tokens * self.model_input_token_cost_per_million
            + output_tokens * self.model_output_token_cost_per_million
        )

        response_dict.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name,
                "total_cost": total_cost,
                "model_inference_type": self.model_inference_type,
                "latency_ms": (time_ns() - start_time) / 1e6,
            }
        )
        return response_dict


def _fallback_text_column(
    *, headers: List[str], column_stats: Dict[str, Dict[str, Any]]
) -> str:
    """Pick the longest-average non-empty string column as a last resort.

    Used when the model returned a header that is not present in the actual
    CSV (rare, but the schema only constrains type, not membership).
    """
    best_header = ""
    best_score = -1.0
    for header in headers:
        stats = column_stats.get(header) or {}
        if stats.get("looks_numeric") or stats.get("looks_boolean"):
            continue
        avg_len = float(stats.get("avg_len") or 0.0)
        non_empty = float(stats.get("non_empty_count") or 0.0)
        if non_empty <= 0:
            continue
        score = avg_len * non_empty
        if score > best_score:
            best_score = score
            best_header = header
    return best_header or (headers[0] if headers else "")
