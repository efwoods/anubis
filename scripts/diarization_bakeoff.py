#!/usr/bin/env python
"""Run the same recordings through every configured diarizer and compare them.

For each YouTube video given, the audio is downloaded once and sent to each
diarization provider that has a key (``OPENAI``, ``ASSEMBLYAI``, ``DEEPGRAM``,
``PYANNOTEAI``) through the same ``transcribe_audio_diarize`` function the
upload pipeline calls, so the comparison measures what an upload would get.

When the video's source publishes a speaker-labelled transcript
(``src/anubis/utils/speech/published_transcripts.py``; lexfridman.com today),
that transcript is the reference and each provider is scored on word error
rate, speaker attribution accuracy, and precision and recall for the avatar's
own turns. Without a published transcript only cost, speed and turn structure
are reported, and accuracy is reported as not measured.

Every call is paid. ``--maximum-seconds`` trims the audio (and the reference)
to the first N seconds to bound the spend; ``--dry-run`` downloads nothing,
calls no provider, and prints the list-price cost of the planned calls.

The store and the API are not touched: nothing is written to an avatar.

    docker exec anubis-dev-langgraph-api-dev-1 \\
        python scripts/diarization_bakeoff.py \\
        --video https://www.youtube.com/watch?v=JN3KPFbWCy8 \\
        --avatar-name "Lex Fridman" --maximum-seconds 1800 --dry-run

Output: ``data/diarization_bakeoff/<run label>/`` (``data/`` is ignored by
git) holds one JSON file per provider and video, ``results.json`` and
``report.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from typing import Any, Dict, List

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.context import GlobalContext  # noqa: E402
from src.anubis.utils.speech.diarization_providers import (  # noqa: E402
    DIARIZATION_PROVIDERS,
    PROVIDER_ASSEMBLYAI,
    PROVIDER_DEEPGRAM,
    PROVIDER_OPENAI,
    PROVIDER_PYANNOTEAI,
)
from src.anubis.utils.speech.diarization_scoring import (  # noqa: E402
    normalized_words,
    segments_within,
    speaker_attribution_scores,
    turn_structure,
    word_error_rate,
)
from src.anubis.utils.speech.published_transcripts import (  # noqa: E402
    load_published_transcript_as_diarization,
    youtube_video_id,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", action="append", required=True,
                        help="YouTube video address; repeatable.")
    parser.add_argument("--provider", action="append", choices=DIARIZATION_PROVIDERS,
                        help="Provider to run; repeatable. Default: every provider with a key.")
    parser.add_argument("--avatar-name", default=None,
                        help="The reference speaker name whose turns are the avatar's.")
    parser.add_argument("--maximum-seconds", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def provider_key_is_set(provider: str, context: GlobalContext) -> bool:
    key_field = {
        PROVIDER_OPENAI: "openai_api_key",
        PROVIDER_ASSEMBLYAI: "assemblyai_api_key",
        PROVIDER_DEEPGRAM: "deepgram_api_key",
        PROVIDER_PYANNOTEAI: "pyannoteai_api_key",
    }[provider]
    value = getattr(context, key_field, None)
    if provider == PROVIDER_OPENAI and not value:
        value = getattr(context, "llm_provider_api_key", None)
    return bool(str(value or "").strip())


def list_price_per_hour(provider: str, context: GlobalContext) -> Dict[str, Any]:
    """The configured list price per audio hour, with the setting that holds the price."""
    if provider == PROVIDER_OPENAI:
        per_minute = float(context.audio_diarization_estimated_price_per_minute or 0.0)
        return {
            "price_per_hour": per_minute * 60.0,
            "currency": "USD",
            "source": "AUDIO_DIARIZATION_ESTIMATED_PRICE_PER_MINUTE x 60 (OpenAI's published estimate; "
            "the real charge is per token)",
        }
    if provider == PROVIDER_ASSEMBLYAI:
        return {"price_per_hour": float(context.assemblyai_price_per_hour or 0.0),
                "currency": "USD", "source": "ASSEMBLYAI_PRICE_PER_HOUR"}
    if provider == PROVIDER_DEEPGRAM:
        return {"price_per_hour": float(context.deepgram_price_per_hour or 0.0),
                "currency": "USD", "source": "DEEPGRAM_PRICE_PER_HOUR"}
    return {"price_per_hour": float(context.pyannoteai_price_per_hour or 0.0),
            "currency": str(context.pyannoteai_billing_currency or "EUR").upper(),
            "source": "PYANNOTEAI_PRICE_PER_HOUR"}


def download_audio(video_url: str, directory: str, maximum_seconds: float | None) -> str:
    """Download the video's audio as mp3 (trimmed when asked) and return the path."""
    import yt_dlp

    video_id = youtube_video_id(video_url) or "audio"
    output_template = os.path.join(directory, f"{video_id}.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([video_url])
    mp3_path = os.path.join(directory, f"{video_id}.mp3")
    if maximum_seconds is None:
        return mp3_path
    trimmed_path = os.path.join(directory, f"{video_id}.trimmed.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, "-t", f"{maximum_seconds}",
         "-c", "copy", trimmed_path],
        check=True,
        capture_output=True,
    )
    return trimmed_path


def audio_duration_seconds(mp3_path: str) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", mp3_path],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


async def run_provider(
    provider: str, mp3_path: str, context: GlobalContext
) -> Dict[str, Any]:
    """Diarize one file with one provider through the upload pipeline's function."""
    from src.anubis.utils.utility import transcribe_audio_diarize

    provider_context = dataclasses.replace(context, audio_diarization_provider=provider)
    with open(mp3_path, "rb") as audio_file:
        data_uri = "data:audio/mp3;base64," + base64.b64encode(audio_file.read()).decode("utf-8")
    response = await transcribe_audio_diarize(
        media_base64=data_uri,
        context=provider_context,
        encoded_reference_audio=None,
        filename=os.path.basename(mp3_path),
        content_type="audio/mp3",
        namespace_chunk_labels_without_reference=True,
    )
    response.pop("encoded_audio_base64", None)
    return response


def score_response(
    response: Dict[str, Any],
    reference: Dict[str, Any] | None,
    *,
    avatar_name: str | None,
    maximum_seconds: float | None,
    audio_seconds: float,
) -> Dict[str, Any]:
    from src.subgraphs.process_media_graph.utils.helper_functions import (
        coalesce_segments_by_speaker,
        merge_interleaved_fragments,
    )

    segments = [segment for segment in (response.get("segments") or []) if isinstance(segment, dict)]
    turns = coalesce_segments_by_speaker(merge_interleaved_fragments(segments))
    latency_seconds = float(response.get("latency_ms") or 0.0) / 1000.0
    row: Dict[str, Any] = {
        "model": response.get("model"),
        "audio_seconds": audio_seconds,
        "cost": float(response.get("total_cost") or 0.0),
        "currency": response.get("billing_currency") or "USD",
        "cost_per_audio_hour": (
            float(response.get("total_cost") or 0.0) / (audio_seconds / 3600.0)
            if audio_seconds
            else None
        ),
        "latency_seconds": latency_seconds,
        "audio_seconds_per_processing_second": (
            audio_seconds / latency_seconds if latency_seconds else None
        ),
        "diarization_chunk_count": response.get("diarization_chunk_count") or 1,
        "segments": len(segments),
        **turn_structure(turns),
        "word_error_rate": None,
        "speaker_attribution_accuracy": None,
        "avatar_precision": None,
        "avatar_recall": None,
        "reference_speaker_count": None,
        "hypothesis_label_count": len({str(segment.get("speaker")) for segment in segments}),
    }
    if reference is not None:
        reference_segments = segments_within(reference["segments"], maximum_seconds)
        reference_words = normalized_words(" ".join(s["text"] for s in reference_segments))
        hypothesis_words = normalized_words(" ".join(s.get("text") or "" for s in segments))
        row["word_error_rate"] = word_error_rate(reference_words, hypothesis_words)
        attribution = speaker_attribution_scores(
            reference_segments, segments, avatar_speaker_name=avatar_name
        )
        for key in ("speaker_attribution_accuracy", "avatar_precision", "avatar_recall",
                    "reference_speaker_count", "hypothesis_label_count", "aligned_words"):
            row[key] = attribution[key]
    return row


def _cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(results: List[Dict[str, Any]], run_label: str) -> str:
    lines = [
        "# Diarization provider comparison",
        "",
        f"Run: {run_label}",
        "",
        "Accuracy columns are measured against the transcript the recording's source "
        "publishes; \"not measured\" means no published transcript exists for that video.",
        "",
        "| Video | Provider | Model | Audio s | Cost | Cost per audio hour | Latency s | "
        "Audio s per processing s | Word error rate | Speaker attribution accuracy | "
        "Avatar precision | Avatar recall | Labels (reference speakers) | Turns | Turns under 40 characters |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        if result.get("error"):
            lines.append(
                f"| {result['video_id']} | {result['provider']} | failed: {result['error']} "
                "| | | | | | | | | | | | |"
            )
            continue
        lines.append(
            f"| {result['video_id']} | {result['provider']} | {result['model']} | "
            f"{_cell(result['audio_seconds'], 2)} | {_cell(result['cost'], 5)} {result['currency']} | "
            f"{_cell(result['cost_per_audio_hour'], 5)} {result['currency']} | "
            f"{_cell(result['latency_seconds'], 1)} | "
            f"{_cell(result['audio_seconds_per_processing_second'], 1)} | "
            f"{_cell(result['word_error_rate'])} | {_cell(result['speaker_attribution_accuracy'])} | "
            f"{_cell(result['avatar_precision'])} | {_cell(result['avatar_recall'])} | "
            f"{result['hypothesis_label_count']} ({_cell(result['reference_speaker_count'])}) | "
            f"{result['turns']} | {result['turns_under_short_limit']} |"
        )
    lines.append("")
    return "\n".join(lines)


async def run_bakeoff(arguments: argparse.Namespace) -> int:
    context = GlobalContext()
    providers = arguments.provider or [
        provider for provider in DIARIZATION_PROVIDERS if provider_key_is_set(provider, context)
    ]
    skipped = [provider for provider in DIARIZATION_PROVIDERS if provider not in providers]
    print(f"Providers to run: {', '.join(providers) or 'none'}")
    for provider in skipped:
        if not provider_key_is_set(provider, context):
            print(f"  {provider}: no key set, skipped")

    run_label = arguments.run_label or datetime.now(tz=UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_directory = os.path.join(
        os.path.dirname(__file__), "..", "data", "diarization_bakeoff", run_label
    )

    if arguments.dry_run:
        for video_url in arguments.video:
            reference = await load_published_transcript_as_diarization(video_url)
            reference_seconds = float(reference["duration"]) if reference else None
            planned_seconds = (
                min(reference_seconds, arguments.maximum_seconds)
                if reference_seconds is not None and arguments.maximum_seconds is not None
                else arguments.maximum_seconds or reference_seconds
            )
            print(f"{video_url}")
            print(
                "  published transcript: "
                + (reference["published_transcript_url"] if reference else "none (accuracy not measured)")
            )
            if planned_seconds is None:
                print("  audio duration: not measured (no transcript and no --maximum-seconds)")
                continue
            for provider in providers:
                price = list_price_per_hour(provider, context)
                cost = planned_seconds / 3600.0 * price["price_per_hour"]
                print(
                    f"  {provider}: {planned_seconds:.2f} s / 3600 x {price['price_per_hour']} "
                    f"{price['currency']} per hour = {cost:.5f} {price['currency']} "
                    f"({price['source']})"
                )
        print("Dry run: nothing was downloaded and no provider was called.")
        return 0

    os.makedirs(output_directory, exist_ok=True)
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as download_directory:
        for video_url in arguments.video:
            video_id = youtube_video_id(video_url) or video_url
            reference = await load_published_transcript_as_diarization(video_url)
            mp3_path = await asyncio.to_thread(
                download_audio, video_url, download_directory, arguments.maximum_seconds
            )
            audio_seconds = audio_duration_seconds(mp3_path)
            for provider in providers:
                print(f"{video_id}: {provider} ...")
                try:
                    response = await run_provider(provider, mp3_path, context)
                except Exception as provider_error:  # noqa: BLE001 - keep comparing the others
                    results.append(
                        {"video_id": video_id, "provider": provider,
                         "error": f"{type(provider_error).__name__}: {provider_error}"}
                    )
                    continue
                with open(
                    os.path.join(output_directory, f"{provider}__{video_id}.json"), "w"
                ) as response_file:
                    json.dump(response, response_file, indent=2, default=str)
                row = score_response(
                    response,
                    reference,
                    avatar_name=arguments.avatar_name,
                    maximum_seconds=arguments.maximum_seconds,
                    audio_seconds=audio_seconds,
                )
                results.append({"video_id": video_id, "provider": provider, **row})

    with open(os.path.join(output_directory, "results.json"), "w") as results_file:
        json.dump(results, results_file, indent=2, default=str)
    report = render_report(results, run_label)
    with open(os.path.join(output_directory, "report.md"), "w") as report_file:
        report_file.write(report)
    print(report)
    print(f"Report: {os.path.abspath(output_directory)}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_bakeoff(parse_arguments())))
