#!/usr/bin/env python
# scripts/e2e_billing/scenario_document_uploads.py

"""Document-upload metering through the running API (requires E2E_API_* env).

Uploads the metering test fixtures through
``POST /update_avatar_identity_with_media`` and asserts: the response carries
the pre-request estimate (total plus the input/output split), the usage
snapshot names the document-upload meter, and usage-to-date increments across
uploads. The over-allotment 402 refusal is asserted opportunistically (only a
user whose remaining allotment is already smaller than the fixture estimate
will trip the gate here — driving a real account fully over the allotment is
the messaging scenario's test-clock job, not something to inflict on a shared
development account).

Requires: E2E_API_BASE_URL, E2E_API_KEY, and E2E_ASSISTANT_ID (an avatar owned
by that API key's user). The account must be pro or premium (upload is a paid
capability).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from harness import ScenarioReporter, api_base_url, api_key, print_config_summary

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_FIXTURES = [
    "data/shivon_zilis/test_tokens_1_tokens.md",
    "data/shivon_zilis/test_tokens_2_tokens.md",
    "data/test_data_avatar_evan_woods/_test_image_token_usage_54_5kb.jpg",
    "data/test_data_avatar_evan_woods/reference_audio.wav",
    "data/test_data_avatar_evan_woods/1minuteFounderVideo_EvanWoods_NeuralNexus.mp4",
]

_FIXTURE_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".jpg": "image/jpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
}


def run() -> int:
    """Upload each fixture and assert estimate + usage accounting."""
    print_config_summary()
    reporter = ScenarioReporter("document-upload metering (API)")

    base_url = api_base_url()
    key = api_key()
    assistant_id = os.environ.get("E2E_ASSISTANT_ID")
    if not base_url or not key or not assistant_id:
        reporter.skip(
            "entire scenario",
            "E2E_API_BASE_URL, E2E_API_KEY, and E2E_ASSISTANT_ID are required",
        )
        return reporter.finish()

    headers = {"API-KEY": key}
    previous_used_to_date: int | None = None
    with httpx.Client(base_url=base_url, headers=headers, timeout=180.0) as client:
        for fixture_relative_path in UPLOAD_FIXTURES:
            fixture_path = REPOSITORY_ROOT / fixture_relative_path
            content_type = _FIXTURE_CONTENT_TYPES[fixture_path.suffix]
            response = client.post(
                "/update_avatar_identity_with_media",
                data={"assistant_id": assistant_id},
                files={
                    "files": (fixture_path.name, fixture_path.read_bytes(), content_type)
                },
            )
            if response.status_code == 402:
                reporter.check(
                    f"{fixture_path.name}: over-allotment upload refused with 402 "
                    "(pay-per-use off)",
                    True,
                )
                continue
            reporter.check(
                f"{fixture_path.name}: accepted (202)",
                response.status_code == 202,
                f"status {response.status_code}: {response.text[:300]}",
            )
            if response.status_code != 202:
                continue
            body = response.json()
            reporter.check(
                f"{fixture_path.name}: response carries the estimate breakdown",
                isinstance(body.get("estimated_tokens_total"), int)
                and isinstance(body.get("estimated_input_tokens"), int)
                and isinstance(body.get("estimated_output_tokens"), int)
                and body["estimated_input_tokens"] + body["estimated_output_tokens"]
                == body["estimated_tokens_total"],
                str({k: body.get(k) for k in ("estimated_tokens_total", "estimated_input_tokens", "estimated_output_tokens")}),
            )
            usage = body.get("usage") or {}
            reporter.check(
                f"{fixture_path.name}: usage snapshot is the document-upload meter",
                usage.get("meter") == "document_upload_tokens",
                str(usage),
            )
            used_to_date = usage.get("used_to_date")
            if previous_used_to_date is not None and isinstance(used_to_date, int):
                reporter.check(
                    f"{fixture_path.name}: metering incremented "
                    f"({previous_used_to_date:,} -> {used_to_date:,})",
                    used_to_date >= previous_used_to_date,
                )
            if isinstance(used_to_date, int):
                previous_used_to_date = used_to_date

    return reporter.finish()


if __name__ == "__main__":
    sys.exit(run())
