"""
chat_gpt_baseline_dataset_creation.py

LEGACY dataset utility. The supported way to (re)generate the unmodified-
inference-model style baseline is ``scripts/retrain_chatgpt_baseline.py``
(``make retrain_baseline``), which regenerates
``data/unmodified_inference_model_baseline_corpus.jsonl`` through the production
``init_model`` path, refits every bundled artifact, records the model in the
provenance sidecar, and publishes to the store. This module's ``generate`` mode
now DELEGATES to that script's corpus generator so the two can never drift; the
``wildchat`` and ``filter`` modes are kept for one-off dataset experiments.

``data/synthetic_baseline.jsonl`` (200 lines, 2026-06-02) is the untracked output
of this module's old generate mode under gpt-5.4-nano; nothing reads it, and the
tracked corpus above supersedes it.

Usage:
    # Option A: Generate the baseline corpus with the configured MODEL (delegates to
    # scripts/retrain_chatgpt_baseline.py; --model overrides MODEL for this run)
    python data/chat_gpt_baseline_dataset_creation.py --mode generate --out baseline.jsonl

    # Option B: Download WildChat (real GPT-3.5/4 conversations) from HuggingFace
    python data/chat_gpt_baseline_dataset_creation.py --mode wildchat --n 500 --out wildchat_baseline.jsonl

    # Option C: Filter ShareGPT-format dataset you already have
    python data/chat_gpt_baseline_dataset_creation.py --mode filter --input your_data.jsonl --out filtered.jsonl

Requirements:
    pip install datasets tqdm   (wildchat mode only)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running as a plain script from anywhere without installing the package
# (this module lives under data/, so neither ``data`` nor ``scripts`` nor ``src``
# is importable otherwise).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Prompt topics — diverse enough to surface ChatGPT's stylistic range
# ---------------------------------------------------------------------------

from data.standardized_questions import (  # noqa: E402
    COGNITIVE_STYLE_AND_CURIOSITY_PROMPTS_OPENNESS,
    COOPERATION_AND_EMPATHY_PROMPTS_AGREEABLENESS,
    COOPERATION_AND_EMPATHY_PROMPTS_EXTENDED,
    DECISION_MAKING_AND_VALUES_PROMPTS,
    DECISION_MAKING_AND_VALUES_PROMPTS_EXTENDED,
    DISCIPLINE_AND_ORGANIZATION_PROMPTS_CONSCIENTIOUSNESS,
    DISCIPLINE_AND_ORGANIZATION_PROMPTS_EXTENDED,
    EMOTIONAL_STABILITY_PROMPTS_NEUROTICISM,
    EMOTIONAL_STABILITY_PROMPTS_NEUROTICISM_EXTENDED,
    IDENTITY_AND_SELF_CONCEPT_PROMPTS,
    INTERESTS_AND_LIFESTYLE_PROMPTS,
    PERSONAL_PREFERENCES_PROMPTS,
    REFLECTION_AND_IDENTITY_EVOLUTION_PROMPTS,
    SOCIAL_MEDIA_BEHAVIOR_PROMPTS,
    SOCIAL_ORIENTATION_PROMPTS_EXTRAVERSION,
    TOPIC_PROMPTS,
)

# Every standardized question set, concatenated into a single pool so the
# baseline covers the full breadth of topics (identity, the Big Five facets,
# lifestyle, decision-making, social-media behavior, preferences, …) rather
# than only TOPIC_PROMPTS. ``dict.fromkeys`` de-dupes while preserving order in
# case any prompt appears in more than one set.
ALL_PROMPTS = list(
    dict.fromkeys(
        TOPIC_PROMPTS
        + IDENTITY_AND_SELF_CONCEPT_PROMPTS
        + COGNITIVE_STYLE_AND_CURIOSITY_PROMPTS_OPENNESS
        + DISCIPLINE_AND_ORGANIZATION_PROMPTS_CONSCIENTIOUSNESS
        + SOCIAL_ORIENTATION_PROMPTS_EXTRAVERSION
        + COOPERATION_AND_EMPATHY_PROMPTS_AGREEABLENESS
        + EMOTIONAL_STABILITY_PROMPTS_NEUROTICISM
        + EMOTIONAL_STABILITY_PROMPTS_NEUROTICISM_EXTENDED
        + INTERESTS_AND_LIFESTYLE_PROMPTS
        + DECISION_MAKING_AND_VALUES_PROMPTS
        + DECISION_MAKING_AND_VALUES_PROMPTS_EXTENDED
        + SOCIAL_MEDIA_BEHAVIOR_PROMPTS
        + REFLECTION_AND_IDENTITY_EVOLUTION_PROMPTS
        + PERSONAL_PREFERENCES_PROMPTS
        + DISCIPLINE_AND_ORGANIZATION_PROMPTS_EXTENDED
        + COOPERATION_AND_EMPATHY_PROMPTS_EXTENDED
    )
)

# ---------------------------------------------------------------------------
# System prompt that instructs GPT to respond in its own characteristic style.
# The single frozen copy lives in scripts/retrain_chatgpt_baseline.py; it is
# re-exported here so older imports keep working.
# ---------------------------------------------------------------------------
from scripts.retrain_chatgpt_baseline import (  # noqa: E402
    STYLE_SYSTEM_PROMPT,  # noqa: F401 - re-exported for older imports
    generate_baseline_corpus,
)


# ---------------------------------------------------------------------------
# Mode A: Generate through the production inference path
# ---------------------------------------------------------------------------
def generate_via_inference_model(out_path: Path, model: str | None = None) -> None:
    """Write the full standardized-question corpus answered by the configured model.

    Delegates to :func:`scripts.retrain_chatgpt_baseline.generate_baseline_corpus`,
    which asks every question through ``init_model`` (the production path, so the
    corpus follows ``MODEL`` across providers) under the frozen style prompt.
    ``model`` overrides ``MODEL`` for this process only; the env files are not
    touched — use the retrain script's ``--model`` for a real switch.
    """
    from src.anubis.utils.context import GlobalContext

    if model:
        os.environ["MODEL"] = model
    context = GlobalContext()
    if not (context.model or "").strip():
        raise SystemExit("MODEL is not set; pass --model or set MODEL in the environment.")
    print(f"Generating {len(ALL_PROMPTS)} examples using {context.model}...")
    written = asyncio.run(generate_baseline_corpus(out_path, context))
    print(f"\nWrote {written} examples to {out_path}")


# ---------------------------------------------------------------------------
# Mode B: Pull real ChatGPT conversations from WildChat (HuggingFace)
# ---------------------------------------------------------------------------
def download_wildchat(n: int, out_path: Path):
    """
    WildChat contains real GPT-3.5 and GPT-4 conversations.
    We filter to English, single-model (gpt-3.5-turbo or gpt-4), non-toxic turns.
    Output is converted to OpenAI fine-tuning JSONL format.

    Dataset: allenai/WildChat-1M
    License: research use only (ODC-BY)
    HuggingFace: https://huggingface.co/datasets/allenai/WildChat-1M
    """
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError:
        raise ImportError("Run: pip install datasets tqdm")

    print("Loading WildChat from HuggingFace (allenai/WildChat-1M)...")
    print("This may take a few minutes on first run (dataset is ~2GB).")

    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True,  # stream to avoid downloading everything
    )

    examples = []
    seen = 0

    for row in tqdm(ds, desc="Filtering WildChat", total=n * 10):
        if len(examples) >= n:
            break

        # Filter criteria
        if row.get("language") != "English":
            continue
        if row.get("toxic") is True:
            continue
        turns = row.get("conversation", [])
        if not turns or len(turns) < 2:
            continue

        # Convert to OpenAI fine-tuning format
        messages = []
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        if len(messages) >= 2 and messages[0]["role"] == "user":
            example = {
                "messages": messages,
                "metadata": {
                    "source": "wildchat",
                    "model": row.get("model", "unknown"),
                    "conversation_id": row.get("conversation_id", ""),
                    "country": row.get("country", ""),
                },
            }
            examples.append(example)
        seen += 1

    _write_jsonl(examples, out_path)
    print(f"\nWrote {len(examples)} conversations to {out_path}")
    print(
        "\nCitation: Zhao et al. (2024). WildChat: 1M ChatGPT Interaction Dataset. "
        "https://arxiv.org/abs/2405.01470"
    )


# ---------------------------------------------------------------------------
# Mode C: Filter an existing ShareGPT-format file
# ---------------------------------------------------------------------------
def filter_existing(input_path: Path, out_path: Path):
    """
    Converts ShareGPT-format data (list of {from, value} turns) to
    OpenAI fine-tuning format and writes a filtered subset.

    ShareGPT format:
        [{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]

    Also handles the lc_run format from your examples:
        [{"role": "user/assistant", "content": "...", "id": "lc_run--..."}]
    """
    with open(input_path) as f:
        # Try JSONL first, then JSON array
        lines = f.read().strip()

    try:
        records = json.loads(lines)
        if not isinstance(records, list):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(l) for l in lines.splitlines() if l.strip()]

    examples = []
    for rec in records:
        # ShareGPT format
        if "conversations" in rec:
            messages = []
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            for turn in rec["conversations"]:
                role = role_map.get(turn.get("from", ""), turn.get("from", ""))
                content = turn.get("value", "").strip()
                if role in ("user", "assistant", "system") and content:
                    messages.append({"role": role, "content": content})
            if messages:
                examples.append({"messages": messages})

        # lc_run / raw messages format
        elif "role" in rec and "content" in rec:
            # These are individual turns — group them by collecting adjacent turns
            # (This branch handles a flat list of turns as a single conversation)
            examples.append(
                {
                    "messages": [
                        {"role": rec["role"], "content": rec["content"]}
                    ]
                }
            )

        # Already in OpenAI format
        elif "messages" in rec:
            examples.append(rec)

    _write_jsonl(examples, out_path)
    print(f"Converted {len(examples)} examples → {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _validate_jsonl(path: Path):
    """Quick sanity check — prints first 2 examples."""
    print(f"\n--- Sample from {path} ---")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            for m in msgs[:3]:
                role = m["role"].upper()
                snippet = m["content"][:120].replace("\n", " ")
                print(f"  [{role}] {snippet}...")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    """Parse the command line and run the selected mode."""
    parser = argparse.ArgumentParser(
        description="Generate or collect a ChatGPT-style baseline dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "wildchat", "filter"],
        required=True,
        help=(
            "generate = call OpenAI API to synthesize examples; "
            "wildchat = download real ChatGPT conversations from HuggingFace; "
            "filter = convert existing ShareGPT/lc_run file to fine-tuning format"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of examples to produce (wildchat mode; generate always uses the full pool)",
    )
    parser.add_argument("--out", type=str, default="baseline.jsonl", help="Output file path")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file (required for --mode filter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Inference model for --mode generate; defaults to the configured MODEL "
            "(the model is never hardcoded here, so the corpus always follows the deployment)"
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.mode == "generate":
        if args.n is not None:
            print("NOTE: --n is ignored in generate mode; the full standardized pool is used.")
        generate_via_inference_model(out_path, model=args.model)

    elif args.mode == "wildchat":
        download_wildchat(args.n or 200, out_path)

    elif args.mode == "filter":
        if not args.input:
            print("ERROR: --input required for filter mode.")
            return
        filter_existing(Path(args.input), out_path)

    _validate_jsonl(out_path)


if __name__ == "__main__":
    main()