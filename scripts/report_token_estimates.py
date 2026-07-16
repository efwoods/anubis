#!/usr/bin/env python
# scripts/report_token_estimates.py

"""Quantify and report the manual token estimates for the metering test fixtures.

Produces a markdown report covering:

1. Every metering test-data file (the fixtures named in the metering test
   plan) with the probe inputs (word counts, image dimensions, media
   durations) and the resulting estimates — both the bare extraction estimate
   and the full upload estimate including the configured identity-analysis
   passes (analysis passes are INPUT tokens: each pass re-reads the extracted
   content).
2. Every static system-prompt template WITHOUT variables injected — the
   word-ratio token estimate of the template text exactly as authored, so the
   fixed prompt overhead of each pipeline stage is known.
3. The /message request baseline: how message content translates to estimated
   tokens (including the 72-token scenario figure), and how the full
   pre-request estimate is composed at runtime (measured system prompt +
   fixed overhead + content + expected output).

Usage:

    python scripts/report_token_estimates.py            # markdown to stdout
    python scripts/report_token_estimates.py --output report.md
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import os
import sys
from io import BytesIO
from pathlib import Path

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.billing.estimation import (  # noqa: E402
    count_words,
    estimate_media_item_token_breakdown,
    estimate_text_tokens_from_words,
)
from src.anubis.utils.context import GlobalContext  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TEXT_TEST_FILES = [
    "data/shivon_zilis/test_tokens_1_tokens.md",
    "data/shivon_zilis/test_tokens_2_tokens.md",
]
IMAGE_TEST_FILE = "data/test_data_avatar_evan_woods/_test_image_token_usage_54_5kb.jpg"
AUDIO_TEST_FILE = "data/test_data_avatar_evan_woods/reference_audio.wav"
VIDEO_TEST_FILE = (
    "data/test_data_avatar_evan_woods/1minuteFounderVideo_EvanWoods_NeuralNexus.mp4"
)

# Every module holding static prompt templates used at runtime. All-caps
# string attributes on these modules are treated as prompt templates, so new
# prompts are picked up automatically on the next run.
PROMPT_MODULE_PATHS = [
    "src.anubis.utils.prompts.system_prompts",
    "src.anubis.utils.prompts.learn_information_user_creator_vs_public_prompt",
    "src.anubis.utils.prompts.first_person_rewriter_prompt",
    "src.anubis.utils.prompts.fact_rewriter_prompt",
    "src.anubis.utils.prompts.concise_context_summary_prompt",
    "src.anubis.utils.prompts.target_speaker_attribution_prompt",
    "src.anubis.utils.prompts.text_dialogue_segmentation_prompt",
    "src.anubis.utils.prompts.Identify_general_characteristics",
    "src.anubis.utils.prompts.psycho_analysis.emotional_trigger_analysis_prompt",
    "src.anubis.utils.schema",
]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _report_text_file(relative_path: str, analysis_passes: int) -> list[str]:
    text_content = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    word_count = count_words(text_content)
    breakdown = estimate_media_item_token_breakdown(
        "text",
        word_count=word_count,
        include_analysis=True,
        analysis_passes=analysis_passes,
    )
    extraction_only = estimate_text_tokens_from_words(word_count)
    return [
        relative_path,
        "text",
        f"{word_count} words",
        f"{extraction_only:,}",
        f"{breakdown.input_tokens - extraction_only:,}",
        f"{breakdown.input_tokens:,}",
        f"{breakdown.output_tokens:,}",
        f"{breakdown.total_tokens:,}",
    ]


def _report_image_file(relative_path: str, analysis_passes: int) -> list[str]:
    from PIL import Image

    image_bytes = (REPOSITORY_ROOT / relative_path).read_bytes()
    with Image.open(BytesIO(image_bytes)) as image:
        width_pixels, height_pixels = image.size
    breakdown = estimate_media_item_token_breakdown(
        "image",
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        include_analysis=True,
        analysis_passes=analysis_passes,
    )
    no_analysis = estimate_media_item_token_breakdown(
        "image",
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        include_analysis=False,
        analysis_passes=analysis_passes,
    )
    return [
        relative_path,
        "image",
        f"{width_pixels}x{height_pixels} px",
        f"{no_analysis.input_tokens:,}",
        f"{breakdown.input_tokens - no_analysis.input_tokens:,}",
        f"{breakdown.input_tokens:,}",
        f"{breakdown.output_tokens:,}",
        f"{breakdown.total_tokens:,}",
    ]


async def _report_media_file(
    relative_path: str, kind: str, analysis_passes: int
) -> list[str]:
    from src.anubis.utils.utility import (
        get_audio_duration_seconds,
        get_video_duration_seconds,
    )

    media_bytes = (REPOSITORY_ROOT / relative_path).read_bytes()
    media_base64 = base64.b64encode(media_bytes).decode("ascii")
    if kind == "audio":
        duration_seconds = await get_audio_duration_seconds(
            media_base64, relative_path
        )
    else:
        duration_seconds = await get_video_duration_seconds(
            media_base64, relative_path
        )
    breakdown = estimate_media_item_token_breakdown(
        kind,  # type: ignore[arg-type]
        duration_seconds=duration_seconds,
        include_analysis=True,
        analysis_passes=analysis_passes,
    )
    no_analysis = estimate_media_item_token_breakdown(
        kind,  # type: ignore[arg-type]
        duration_seconds=duration_seconds,
        include_analysis=False,
        analysis_passes=analysis_passes,
    )
    return [
        relative_path,
        kind,
        f"{duration_seconds:.1f} s",
        f"{no_analysis.input_tokens:,}",
        f"{breakdown.input_tokens - no_analysis.input_tokens:,}",
        f"{breakdown.input_tokens:,}",
        f"{breakdown.output_tokens:,}",
        f"{breakdown.total_tokens:,}",
    ]


def _report_prompt_templates() -> list[list[str]]:
    rows: list[list[str]] = []
    for module_path in PROMPT_MODULE_PATHS:
        try:
            module = importlib.import_module(module_path)
        except Exception as import_error:  # noqa: BLE001 - report and continue
            rows.append([module_path, "(import failed)", "-", "-", str(import_error)])
            continue
        for attribute_name in sorted(dir(module)):
            if not attribute_name.isupper():
                continue
            attribute_value = getattr(module, attribute_name)
            if not isinstance(attribute_value, str) or len(attribute_value) < 40:
                continue
            word_count = count_words(attribute_value)
            rows.append(
                [
                    module_path.rsplit(".", 1)[-1],
                    attribute_name,
                    f"{len(attribute_value):,}",
                    f"{word_count:,}",
                    f"{estimate_text_tokens_from_words(word_count):,}",
                ]
            )
    return rows


def _report_message_baseline(context: GlobalContext) -> str:
    expected_output_tokens = int(context.message_expected_output_tokens_estimate or 0)

    lines = [
        "The runtime /message estimate is composed as:",
        "",
        "```",
        "estimated_input_tokens  = measured system prompt (built first, measured",
        "                          manually at 4/3 tokens per word)",
        "                        + measured bound tool schemas (enumerated from",
        "                          the compiled deep agent, serialized JSON",
        "                          characters / 4 — new tools are included",
        "                          automatically)",
        "                        + ceil(message words x 4/3)",
        "                        + attached-image vision patches",
        "                        (no fixed or guessed input overhead — every"
        " input component is measured)",
        f"estimated_output_tokens = {expected_output_tokens:,}"
        " (MESSAGE_EXPECTED_OUTPUT_TOKENS_ESTIMATE)"
        " + 300 per attached image",
        "```",
        "",
        "Message-content token figures for the test scenarios:",
        "",
    ]
    scenario_rows: list[list[str]] = []
    for relative_path in TEXT_TEST_FILES:
        text_content = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        word_count = count_words(text_content)
        scenario_rows.append(
            [
                f"message = contents of {relative_path}",
                f"{word_count}",
                f"{estimate_text_tokens_from_words(word_count):,}",
            ]
        )
    scenario_rows.append(
        [
            "the 72-token scenario message",
            "54",
            f"{estimate_text_tokens_from_words(54):,}",
        ]
    )
    lines.append(
        _markdown_table(
            ["scenario", "message words", "estimated message-content tokens"],
            scenario_rows,
        )
    )
    lines.extend(
        [
            "",
            "A 54-word message estimates to exactly ceil(54 x 4/3) = 72 tokens —",
            "the '72 tokens per scenario' figure holds for a 54-word test message.",
            "The full request estimate adds the measured system prompt (varies per",
            "avatar; typically thousands to tens of thousands of tokens), the fixed",
            f"per-turn overhead ({fixed_overhead_tokens:,}), and the expected reply"
            f" budget ({expected_output_tokens:,}).",
        ]
    )
    return "\n".join(lines)


async def build_report() -> str:
    """Assemble the full markdown report (fixtures, prompts, /message baseline)."""
    context = GlobalContext()
    analysis_passes = int(context.estimated_analysis_passes_per_document or 0)

    fixture_rows = [
        _report_text_file(relative_path, analysis_passes)
        for relative_path in TEXT_TEST_FILES
    ]
    fixture_rows.append(_report_image_file(IMAGE_TEST_FILE, analysis_passes))
    fixture_rows.append(
        await _report_media_file(AUDIO_TEST_FILE, "audio", analysis_passes)
    )
    fixture_rows.append(
        await _report_media_file(VIDEO_TEST_FILE, "video", analysis_passes)
    )

    sections = [
        "# Token estimate report",
        "",
        f"Analysis passes per document: {analysis_passes} "
        "(ESTIMATED_ANALYSIS_PASSES_PER_DOCUMENT; analysis passes are INPUT "
        "tokens — each pass re-reads the extracted content).",
        "",
        "## Test-data files (/update_avatar_identity_with_media estimates)",
        "",
        _markdown_table(
            [
                "file",
                "kind",
                "probe",
                "extraction input",
                "analysis input",
                "input tokens",
                "output tokens",
                "total tokens",
            ],
            fixture_rows,
        ),
        "",
        "## Static prompt templates (no variables injected)",
        "",
        _markdown_table(
            ["module", "template", "characters", "words", "estimated tokens"],
            _report_prompt_templates(),
        ),
        "",
        "## /message baseline",
        "",
        _report_message_baseline(context),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    """Parse arguments and print (or write) the markdown token-estimate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the markdown report to (stdout otherwise).",
    )
    arguments = parser.parse_args()

    report_markdown = asyncio.run(build_report())
    if arguments.output:
        Path(arguments.output).write_text(report_markdown, encoding="utf-8")
        print(f"Report written to {arguments.output}")
    else:
        print(report_markdown)


if __name__ == "__main__":
    main()
