#!/usr/bin/env python
# scripts/retrain_chatgpt_baseline.py

"""Retrain the unmodified-inference-model (ChatGPT) style baseline end to end.

The authenticity check in ``graph._attach_analyzed_features`` scores every avatar
reply against a fixed cloud of UNMODIFIED inference-model responses — the CHATGPT
end of the CHATGPT — AVATAR — REAL axis. That cloud is five coupled artifacts, all
derived from one corpus of raw model replies:

* ``src/anubis/utils/dataset/baseline_features_arr.npy`` — the (n_docs, F) matrix,
* ``src/anubis/utils/dataset/baseline_features_model_b64.pkl`` — an IsolationForest,
* ``src/anubis/utils/dataset/baseline_features_explainer_b64.pkl`` — a SHAP explainer,
* ``src/anubis/utils/dataset/baseline_key_phrases.json`` — the signature phrases,
* ``BASELINE_RESPONSE_THRESHOLD`` — the Tukey fence, in ``.env`` / ``.env.dev`` and
  the ``GlobalContext`` default.

WHY THIS SCRIPT EXISTS. When the inference model is upgraded, the corpus no longer
represents what the raw model sounds like, so the avatar is scored against the
PREVIOUS model's style cloud. Nothing fails loudly: the feature-vector width is
unchanged, the pickles still unpickle, the SSE frames still populate — the
``no_statistically_significantly_difference_from_unmodified_llm_response...``
verdict simply becomes wrong. The staleness is invisible precisely because the
runtime's self-heal keys on feature-vector WIDTH and nothing else.

The corpus is regenerated through :func:`init_model`, the same path production
inference uses, so this script follows whatever ``MODEL`` is configured. The
procedure after a model upgrade is therefore exactly two steps: set ``MODEL`` in
``.env``, then run this script.

Usage::

    .venv/bin/python scripts/retrain_chatgpt_baseline.py
    .venv/bin/python scripts/retrain_chatgpt_baseline.py --skip-store-purge

Run from the repository root under the PINNED scikit-learn (``scikit-learn==1.9.0``
in ``pyproject.toml``): the artifacts are sklearn pickles and are not portable
across releases, so fitting them under a different version yields pickles the
container cannot load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.build_baseline_features_arr import (  # noqa: E402
    BASELINE_CORPUS_PATH,
    build,
)
from data.standardized_questions import ALL_STANDARDIZED_QUESTIONS  # noqa: E402
from src.anubis.utils.context import GlobalContext  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The prompt that elicits the model's own unmodified assistant voice.
#
# BYTE-IDENTICAL AND FROZEN. Every baseline corpus ever fitted was generated under
# this exact text, and the whole point of the baseline is that successive retrains
# differ ONLY by the model that answered. Editing so much as a comma shifts the
# corpus for a reason unrelated to the model upgrade and silently destroys
# comparability across retrains — the resulting threshold would move for reasons
# nobody could attribute. ``tests/unit_tests/test_retrain_baseline.py`` asserts this
# string still matches the system message recorded in the tracked corpus.
STYLE_SYSTEM_PROMPT = """You are ChatGPT, responding in your typical conversational style.

Your responses characteristically:
- Use short, punchy sentence fragments for emphasis (e.g., "No jargon. No fluff. Just the idea.")
- Employ *italics* for stress and **bold** for key terms or concepts
- Structure answers with clean logical flow, often using em-dashes or colons
- End responses with a follow-up probe or clarifying question to continue the conversation
  (e.g., "If you tell me more about X, I can tailor this further.")
- Feel thorough but not academic — confident, structured, slightly formal
- Occasionally use numbered lists or bullet points for multi-part answers
- Mirror the user's emotional register when the topic is personal

Respond naturally. Do not announce your style. Just respond as you normally would."""

ENVIRONMENT_FILE_PATHS = [
    REPOSITORY_ROOT / ".env",
    REPOSITORY_ROOT / ".env.dev",
]
CONTEXT_MODULE_PATH = REPOSITORY_ROOT / "src" / "anubis" / "utils" / "context.py"

# The four baseline artifacts are read-through cached in the LangGraph Postgres
# store at their namespace ROOT, so the namespace prefix equals the key for each.
BASELINE_STORE_CACHE_KEYS = [
    "baseline_features_arr_list_str",
    "baseline_features_model_b64_pkl",
    "baseline_features_explainer_b64_pkl",
    "baseline_key_phrase_profile",
]

# store_vectors carries ON DELETE CASCADE from store(prefix, key), so removing the
# store rows already removes any matching embeddings (see avatar_deletion.py).
SQL_DELETE_STORE_ROW = "DELETE FROM store WHERE prefix = %s AND key = %s;"


def _display_path(path: Path) -> str:
    """Render a path relative to the repo root when possible, else absolutely."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Stage 1 — resolve the configured inference model
# ---------------------------------------------------------------------------
def resolve_inference_model(context: GlobalContext) -> Tuple[str, str]:
    """Return ``(model_provider, model)`` from configuration, or exit non-zero.

    ``init_model`` reads these from the environment itself; resolving them here as
    well is what lets the script NAME the model it is about to fingerprint, so the
    operator can see that the retrain matches the upgrade they just made.
    """
    model_name = (context.model or "").strip()
    model_provider = (context.model_provider or "").strip()
    if not model_name:
        raise SystemExit(
            "MODEL is not set. The baseline must be generated by the configured "
            "inference model; set MODEL in .env before retraining."
        )
    return model_provider, model_name


# ---------------------------------------------------------------------------
# Stage 2 — regenerate the corpus with the configured model
# ---------------------------------------------------------------------------
async def generate_baseline_corpus(
    corpus_path: Path, context: GlobalContext
) -> int:
    """Ask the configured inference model every standardized question, write JSONL.

    Returns the number of conversations written. Questions are asked concurrently
    under ``standardized_question_analysis_concurrency`` — the same bound the
    per-document standardized-question analyzer uses — so the full bank does not
    exhaust provider rate limits.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.anubis.utils.model import init_model

    # One model instance reused across every call (stateless). This is the same
    # plain-text invocation shape used in production by context_compression.py.
    model = init_model(tools=[], tool_choice="auto", response_format=None)

    concurrency = max(1, context.standardized_question_analysis_concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    questions = list(ALL_STANDARDIZED_QUESTIONS)
    completed_count = 0

    async def _answer_one(question: str) -> dict | None:
        nonlocal completed_count
        async with semaphore:
            response = await model.ainvoke(
                [
                    SystemMessage(content=STYLE_SYSTEM_PROMPT),
                    HumanMessage(content=question),
                ]
            )
        content = getattr(response, "content", "")
        # A tool-calling model can return content as a list of blocks; keep only the
        # text, since the stylometry operates on the reply as the reader sees it.
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        completed_count += 1
        if completed_count % 20 == 0:
            print(f"  {completed_count}/{len(questions)} answered")
        if not isinstance(content, str) or not content.strip():
            return None
        return {
            "messages": [
                {"role": "system", "content": STYLE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": content},
            ]
        }

    print(
        f"Generating {len(questions)} baseline replies "
        f"(concurrency {concurrency})..."
    )
    results = await asyncio.gather(
        *(_answer_one(question) for question in questions),
        return_exceptions=True,
    )

    conversations: List[dict] = []
    failure_count = 0
    for question, result in zip(questions, results):
        if isinstance(result, BaseException):
            failure_count += 1
            print(f"  FAILED: {question[:70]}... — {result}")
            continue
        if result is None:
            failure_count += 1
            print(f"  EMPTY REPLY: {question[:70]}...")
            continue
        conversations.append(result)

    if not conversations:
        raise SystemExit(
            "Every generation call failed — refusing to overwrite the existing "
            "corpus. Check the model configuration and provider credentials."
        )
    # A partial corpus silently shrinks the cloud the threshold is calibrated on, so
    # say so loudly rather than letting a quiet rate-limit truncation pass as a
    # successful retrain.
    if failure_count:
        print(
            f"WARNING: {failure_count} of {len(questions)} questions produced no "
            "reply; the corpus is smaller than the question bank."
        )

    # Write to a temp file in the destination directory, then atomically replace, so
    # an interrupted run leaves the working corpus intact rather than truncated.
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(corpus_path.parent),
        prefix=corpus_path.name,
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        for conversation in conversations:
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    os.replace(temporary_path, corpus_path)

    print(
        f"Wrote {len(conversations)} conversations to "
        f"{_display_path(corpus_path)}"
    )
    return len(conversations)


# ---------------------------------------------------------------------------
# Stage 4 — write the recalibrated threshold everywhere it is declared
# ---------------------------------------------------------------------------
def replace_assignment_value(text: str, pattern: str, replacement: str) -> str:
    """Substitute exactly one regex match, raising when the target is absent.

    The threshold lives in three files under two different syntaxes (an env
    ``NAME=value`` line and a Python ``default=value`` keyword). Both rewrites are
    the same operation: find the one declaration and swap its value while leaving
    the surrounding comments untouched. Raising on zero matches is the point —
    a silent no-op here would leave the threshold disagreeing with the artifacts
    it was calibrated against, which is the exact failure this script exists to
    prevent. More than one match is equally fatal: it means the anchor is not
    specific enough to know which declaration is authoritative.
    """
    updated_text, substitution_count = re.subn(
        pattern, replacement, text, count=2, flags=re.MULTILINE
    )
    if substitution_count == 0:
        raise ValueError(f"No declaration matched {pattern!r}")
    if substitution_count > 1:
        raise ValueError(
            f"{substitution_count} declarations matched {pattern!r}; expected exactly one"
        )
    return updated_text


def write_threshold_to_environment_files(threshold: float) -> None:
    """Rewrite ``BASELINE_RESPONSE_THRESHOLD`` in ``.env`` and ``.env.dev``."""
    for environment_file_path in ENVIRONMENT_FILE_PATHS:
        if not environment_file_path.exists():
            print(f"  SKIPPED (absent): {environment_file_path.name}")
            continue
        original_text = environment_file_path.read_text(encoding="utf-8")
        previous_value_match = re.search(
            r"^BASELINE_RESPONSE_THRESHOLD=(.*)$", original_text, flags=re.MULTILINE
        )
        updated_text = replace_assignment_value(
            original_text,
            r"^BASELINE_RESPONSE_THRESHOLD=.*$",
            f"BASELINE_RESPONSE_THRESHOLD={threshold!r}",
        )
        environment_file_path.write_text(updated_text, encoding="utf-8")
        previous_value = (
            previous_value_match.group(1) if previous_value_match else "<unset>"
        )
        print(
            f"  {environment_file_path.name}: {previous_value} -> {threshold!r}"
        )


def write_threshold_to_context_default(threshold: float) -> None:
    """Rewrite the ``baseline_response_threshold`` default in ``GlobalContext``.

    Env always wins at runtime, but a deployment that never sets the variable falls
    back to this default, so leaving it stale hides the same miscalibration in the
    one configuration where it is hardest to notice.
    """
    original_text = CONTEXT_MODULE_PATH.read_text(encoding="utf-8")
    # Anchor on the field name so the substitution cannot wander onto another
    # dataclass field that happens to share a default value.
    anchor_pattern = r"(baseline_response_threshold: float = field\(\s*\n\s*default=)[0-9.eE+-]+"
    previous_value_match = re.search(anchor_pattern, original_text)
    updated_text = replace_assignment_value(
        original_text, anchor_pattern, rf"\g<1>{threshold!r}"
    )
    CONTEXT_MODULE_PATH.write_text(updated_text, encoding="utf-8")
    previous_value = "<unknown>"
    if previous_value_match:
        previous_value = original_text[
            previous_value_match.end(1) : previous_value_match.end()
        ]
    print(f"  context.py: {previous_value} -> {threshold!r}")


# ---------------------------------------------------------------------------
# Stage 5 — invalidate the Postgres store caches
# ---------------------------------------------------------------------------
class StoreCachePurgeError(RuntimeError):
    """The disk artifacts were rebuilt but the cached copies could not be cleared."""


def _redact_credentials(store_uri: str) -> str:
    """Strip user:password from a connection URI so it is safe to print."""
    return re.sub(r"://[^@/]*@", "://***:***@", store_uri)


def purge_baseline_store_cache(context: GlobalContext) -> None:
    """Delete the cached baseline artifacts from the LangGraph Postgres store.

    Rebuilding the files on disk is NOT enough. Each artifact is cached in the store
    on first use, and the runtime's self-heal (``baseline_feature_array_is_current``
    and the ``n_features_in_`` checks in ``utility.py``) only reloads from disk when
    the feature-vector WIDTH changes. A model-upgrade retrain leaves the width
    unchanged, so without this purge a running deployment keeps serving the previous
    model's cloud indefinitely.

    Uses plain psycopg rather than ``AsyncPostgresStore``: constructing the store
    requires the 640-dim ``IndexConfig``, which would load the HuggingFace embedding
    model just to issue four DELETEs.
    """
    import psycopg

    store_uri = (context.async_postgres_store_uri or "").strip()
    if not store_uri:
        raise SystemExit(
            "ASYNC_POSTGRES_STORE_URI is not set, so the cached baseline artifacts "
            "cannot be invalidated. Set it, or pass --skip-store-purge and clear "
            f"these keys yourself: {', '.join(BASELINE_STORE_CACHE_KEYS)}"
        )

    try:
        connection = psycopg.connect(store_uri, autocommit=True, connect_timeout=15)
    except psycopg.OperationalError as connection_error:
        # The committed URI names ``host.docker.internal``, which resolves inside the
        # compose network but not on the host, so a host-side run lands here. Say what
        # to do instead of surfacing a bare driver error, and let the caller decide
        # whether the run failed — the corpus, artifacts, and threshold are already
        # written by this point and must not be discarded.
        raise StoreCachePurgeError(
            f"Could not connect to the store at {_redact_credentials(store_uri)}: "
            f"{connection_error}"
        ) from connection_error

    with connection:
        with connection.cursor() as cursor:
            for cache_key in BASELINE_STORE_CACHE_KEYS:
                cursor.execute(SQL_DELETE_STORE_ROW, (cache_key, cache_key))
                print(f"  {cache_key}: {cursor.rowcount} row(s) deleted")


# ---------------------------------------------------------------------------
# Stage 6 — verify what was written is internally consistent
# ---------------------------------------------------------------------------
def verify_written_artifacts() -> None:
    """Reload every written artifact and assert the widths agree.

    Catches the failure mode where one artifact is refit and another is not: the
    matrix, the forest, and the explainer background must all describe the same
    feature vector or the runtime raises mid-request.
    """
    import base64
    import pickle

    import numpy as np

    from src.anubis.utils.dataset.style_features import (
        BASELINE_FEATURES_EXPLAINER_PATH,
        BASELINE_FEATURES_MODEL_PATH,
        BASELINE_KEY_PHRASES_PATH,
        FEATURE_NAMES,
        STYLE_FEATURE_VECTOR_VERSION,
        load_bundled_baseline_features_arr,
    )

    expected_width = len(FEATURE_NAMES)

    feature_matrix = load_bundled_baseline_features_arr()
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != expected_width:
        raise SystemExit(
            f"Baseline matrix has shape {feature_matrix.shape}; expected width {expected_width}"
        )

    model = pickle.loads(
        base64.b64decode(Path(BASELINE_FEATURES_MODEL_PATH).read_bytes().decode("utf-8"))
    )
    model_width = getattr(model, "n_features_in_", None)
    if model_width != expected_width:
        raise SystemExit(
            f"IsolationForest was fit on {model_width} features; expected {expected_width}"
        )

    explainer = pickle.loads(
        base64.b64decode(
            Path(BASELINE_FEATURES_EXPLAINER_PATH).read_bytes().decode("utf-8")
        )
    )
    explainer_background = np.asarray(
        getattr(getattr(explainer, "data", None), "data", np.empty((0, 0)))
    )
    if explainer_background.ndim != 2 or explainer_background.shape[1] != expected_width:
        raise SystemExit(
            f"SHAP explainer background has shape {explainer_background.shape}; "
            f"expected width {expected_width}"
        )

    key_phrases = json.loads(Path(BASELINE_KEY_PHRASES_PATH).read_text(encoding="utf-8"))

    print(f"  matrix:      {feature_matrix.shape}")
    print(f"  forest:      fit on {model_width} features")
    print(f"  explainer:   background {explainer_background.shape}")
    print(f"  key phrases: {len(key_phrases)}")
    print(f"  feature-vector version: {STYLE_FEATURE_VECTOR_VERSION}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _print_manual_purge_instructions() -> None:
    """Print the exact rows to delete when the automatic purge did not run."""
    print("    DELETE FROM store WHERE prefix = key AND key IN (")
    print(
        "      "
        + ", ".join(f"'{cache_key}'" for cache_key in BASELINE_STORE_CACHE_KEYS)
    )
    print("    );")


def main() -> None:
    """Run the full retrain: generate, refit, write the threshold, purge, verify."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the unmodified-inference-model baseline corpus with the "
            "currently configured model, refit the bundled stylometry artifacts, "
            "write the recalibrated threshold, and invalidate the store caches."
        )
    )
    parser.add_argument(
        "--skip-store-purge",
        action="store_true",
        help=(
            "Do not delete the cached baseline artifacts from the Postgres store. "
            "Files on disk are still rebuilt, but running deployments keep serving "
            "the PREVIOUS baseline until the cached rows are cleared by hand."
        ),
    )
    arguments = parser.parse_args()

    context = GlobalContext()

    print("=" * 72)
    print("STAGE 1/6  Resolve the configured inference model")
    print("=" * 72)
    model_provider, model_name = resolve_inference_model(context)
    print(f"  MODEL_PROVIDER = {model_provider}")
    print(f"  MODEL          = {model_name}")

    print()
    print("=" * 72)
    print("STAGE 2/6  Regenerate the baseline corpus")
    print("=" * 72)
    asyncio.run(generate_baseline_corpus(BASELINE_CORPUS_PATH, context))

    print()
    print("=" * 72)
    print("STAGE 3/6  Refit the bundled stylometry artifacts")
    print("=" * 72)
    threshold = build(BASELINE_CORPUS_PATH)

    print()
    print("=" * 72)
    print("STAGE 4/6  Write the recalibrated threshold")
    print("=" * 72)
    write_threshold_to_environment_files(threshold)
    write_threshold_to_context_default(threshold)

    print()
    print("=" * 72)
    print("STAGE 5/6  Invalidate the cached baseline artifacts in the store")
    print("=" * 72)
    purge_failure: StoreCachePurgeError | None = None
    if arguments.skip_store_purge:
        print("  SKIPPED (--skip-store-purge).")
        _print_manual_purge_instructions()
    else:
        try:
            purge_baseline_store_cache(context)
        except StoreCachePurgeError as error:
            # Everything on disk is already written and correct. Report, finish the
            # run, and fail at the end so this is never mistaken for a clean retrain.
            purge_failure = error
            print(f"  FAILED: {error}")
            print()
            print("  The corpus, artifacts, and threshold ARE written and correct.")
            print("  Only the cached copies remain stale. The committed store URI")
            print("  names host.docker.internal, which resolves inside the compose")
            print("  network but not on the host — so either run this script from")
            print("  inside the container, or point ASYNC_POSTGRES_STORE_URI at a")
            print("  host-reachable address and re-run with --skip-store-purge")
            print("  omitted. Failing that, delete these rows by hand:")
            _print_manual_purge_instructions()

    print()
    print("=" * 72)
    print("STAGE 6/6  Verify the written artifacts")
    print("=" * 72)
    verify_written_artifacts()

    print()
    print(f"Retrained against {model_name}. BASELINE_RESPONSE_THRESHOLD = {threshold!r}")
    print("Commit the rebuilt artifacts, the corpus, and the threshold together —")
    print("they are only meaningful as a set.")

    if purge_failure is not None:
        raise SystemExit(
            "Retrain incomplete: the cached baseline artifacts were NOT invalidated, "
            "so running deployments still serve the previous baseline."
        )


if __name__ == "__main__":
    main()
