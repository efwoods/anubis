#!/usr/bin/env python
# scripts/retrain_chatgpt_baseline.py

r"""Switch the inference model and rebuild the unmodified-inference-model style baseline.

The authenticity check in ``graph._attach_analyzed_features`` scores every avatar
reply against a fixed cloud of UNMODIFIED inference-model responses — the CHATGPT
end of the CHATGPT — AVATAR — REAL axis. That cloud is five coupled artifacts, all
derived from one corpus of raw model replies:

* ``src/anubis/utils/dataset/baseline_features_arr.npy`` — the (n_docs, F) matrix,
* ``src/anubis/utils/dataset/baseline_features_model_b64.pkl`` — an IsolationForest,
* ``src/anubis/utils/dataset/baseline_features_explainer_b64.pkl`` — a SHAP explainer,
* ``src/anubis/utils/dataset/baseline_key_phrases.json`` — the signature phrases,
* ``BASELINE_RESPONSE_THRESHOLD`` — the Tukey fence, in ``.env`` / ``.env.dev`` and
  the ``GlobalContext`` default,

plus a provenance sidecar, ``data/unmodified_inference_model_baseline_corpus.meta.json``,
recording WHICH model produced the corpus, at what per-token cost, and the threshold
that calibration yielded. The artifacts themselves carry no model name (their width
is model-independent), so the sidecar is the only committed record that lets a boot
detect "MODEL changed, baseline did not".

WHY THIS SCRIPT EXISTS. When the inference model is upgraded, the corpus no longer
represents what the raw model sounds like, so the avatar is scored against the
PREVIOUS model's style cloud. Nothing fails loudly: the feature-vector width is
unchanged, the pickles still unpickle, the SSE frames still populate — the
``no_statistically_significantly_difference_from_unmodified_llm_response...``
verdict simply becomes wrong. The staleness is invisible precisely because the
runtime's self-heal keys on feature-vector WIDTH and nothing else.

The corpus is regenerated through :func:`init_model`, the same path production
inference uses, so this script follows whatever ``MODEL`` is configured. The model
and its costs can be set on the command line, in which case the script rewrites
``MODEL`` / ``MODEL_PROVIDER`` / ``MODEL_PROMPT_COST`` / ``MODEL_COMPLETION_COST``
in ``.env`` and ``.env.dev`` itself before generating, so a model switch is ONE
command.

Usage::

    # Switch the model, rewrite the env files, regenerate, refit, publish, verify.
    python scripts/retrain_chatgpt_baseline.py --model gpt-5.6-luna \
        --model-provider OPEN_AI \
        --model-prompt-cost 0.0000002 --model-completion-cost 0.0000012

    # Retrain against whatever MODEL the environment already holds.
    python scripts/retrain_chatgpt_baseline.py

    # Sync a checkout's env files, threshold, and the store from the COMMITTED
    # sidecar and artifacts — no model calls. For a second checkout (prod) that
    # pulled the committed retrain.
    python scripts/retrain_chatgpt_baseline.py --configuration-only

    # Write this checkout's artifacts, sidecar, env lines, and threshold from what
    # the shared store already serves — no model calls. For a checkout whose
    # sibling container already retrained this MODEL (see baseline_provenance.py).
    python scripts/retrain_chatgpt_baseline.py --adopt-from-store

Run inside the API container (``make retrain_baseline ARGS='...'``): its interpreter
carries the PINNED scikit-learn (``scikit-learn==1.9.0`` in ``pyproject.toml``), so
the pickles it fits are the ones the deployment can load, and its network resolves
``ASYNC_POSTGRES_STORE_URI`` (the committed URI names ``host.docker.internal``,
which does not resolve on the host). Afterwards RECREATE the container: compose
reads ``env_file`` when a container is created, so the rewritten values only reach
the API after ``make recreate_dev_api``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.build_baseline_features_arr import (  # noqa: E402
    BASELINE_CORPUS_PATH,
    BaselineBuildResult,
    build,
)
from data.standardized_questions import ALL_STANDARDIZED_QUESTIONS  # noqa: E402
from src.anubis.utils.context import GlobalContext  # noqa: E402
from src.anubis.utils.dataset.style_features import (  # noqa: E402
    BASELINE_FEATURES_ARR_PATH,
    BASELINE_FEATURES_EXPLAINER_PATH,
    BASELINE_FEATURES_MODEL_PATH,
    BASELINE_KEY_PHRASES_PATH,
    BASELINE_PROVENANCE_PATH,
    FEATURE_NAMES,
    STYLE_FEATURE_VECTOR_VERSION,
)

if TYPE_CHECKING:
    import psycopg

JsonDict = Dict[str, Any]

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
BASELINE_PROVENANCE_FILE_PATH = REPOSITORY_ROOT / BASELINE_PROVENANCE_PATH
BASELINE_PROVENANCE_SCHEMA_VERSION = 1

# The model-configuration lines this script may rewrite in the env files. Each
# anchor is a line-start regex so the commented ``# MODEL=...`` alternatives that
# precede the live declaration never match, and ``^MODEL=`` cannot match
# ``MODEL_PROVIDER=`` or ``MODEL_TOKEN_LIMIT=``. The format preserves the quoting
# style each variable already uses in the files (``MODEL`` is quoted, the rest
# are bare).
MODEL_CONFIGURATION_ANCHORS: Dict[str, Tuple[str, str]] = {
    "MODEL": (r"^MODEL=.*$", 'MODEL="{value}"'),
    "MODEL_PROVIDER": (r"^MODEL_PROVIDER=.*$", "MODEL_PROVIDER={value}"),
    "MODEL_PROMPT_COST": (r"^MODEL_PROMPT_COST=.*$", "MODEL_PROMPT_COST={value}"),
    "MODEL_COMPLETION_COST": (
        r"^MODEL_COMPLETION_COST=.*$",
        "MODEL_COMPLETION_COST={value}",
    ),
    "MODEL_TOKEN_LIMIT": (r"^MODEL_TOKEN_LIMIT=.*$", "MODEL_TOKEN_LIMIT={value}"),
}

# The baseline rows in the LangGraph Postgres store, all at their namespace ROOT
# (the namespace prefix equals the key for each). The four artifact rows are what
# the runtime read-through caches; ``baseline_provenance`` records which model
# they belong to so a boot can compare against its own MODEL without touching
# the files (see src/anubis/utils/dataset/baseline_provenance.py).
BASELINE_ARTIFACT_STORE_KEYS = [
    "baseline_features_arr_list_str",
    "baseline_features_model_b64_pkl",
    "baseline_features_explainer_b64_pkl",
    "baseline_key_phrase_profile",
]
BASELINE_PROVENANCE_STORE_KEY = "baseline_provenance"
BASELINE_STORE_CACHE_KEYS = BASELINE_ARTIFACT_STORE_KEYS + [BASELINE_PROVENANCE_STORE_KEY]
# Taken by the container that is retraining so a sibling container booting with
# the same new MODEL waits and adopts instead of regenerating a second corpus.
BASELINE_RETRAIN_LOCK_STORE_KEY = "baseline_retrain_lock"

# store_vectors carries ON DELETE CASCADE from store(prefix, key), so removing the
# store rows already removes any matching embeddings (see avatar_deletion.py).
SQL_DELETE_STORE_ROW = "DELETE FROM store WHERE prefix = %s AND key = %s;"
# Same ``{"value": <str>}`` envelope the runtime writes through ``store.aput``, so a
# row published here is indistinguishable from one the read-through cache wrote.
SQL_UPSERT_STORE_ROW = (
    "INSERT INTO store (prefix, key, value, created_at, updated_at) "
    "VALUES (%s, %s, %s::jsonb, now(), now()) "
    "ON CONFLICT (prefix, key) DO UPDATE "
    "SET value = EXCLUDED.value, updated_at = now();"
)
SQL_SELECT_STORE_ROWS = "SELECT key, value FROM store WHERE prefix = key AND key = ANY(%s);"


def _display_path(path: Path) -> str:
    """Render a path relative to the repo root when possible, else absolutely."""
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _print_stage(number: int, total: int, title: str) -> None:
    print()
    print("=" * 72)
    print(f"STAGE {number}/{total}  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Stage 0 — apply the model configuration
# ---------------------------------------------------------------------------
def format_cost_for_environment_file(cost: float) -> str:
    """Render a per-token cost as a plain decimal, never ``2e-07``.

    ``GlobalContext`` parses either spelling, but the env files are edited by hand
    and a scientific-notation cost is easy to misread by an order of magnitude.
    Ten decimals cover any realistic per-token price (a tenth of a cent per
    million tokens) without rounding.
    """
    return f"{float(cost):.10f}".rstrip("0").rstrip(".") or "0"


@dataclass(frozen=True)
class ModelConfiguration:
    """The env values a model switch may rewrite, as the operator typed them.

    Values are kept as strings so what lands in the env file is exactly what was
    given (or exactly what the provenance sidecar recorded), not a float's
    ``repr``. ``None`` means "leave that line alone".
    """

    model: str | None = None
    model_provider: str | None = None
    model_prompt_cost: str | None = None
    model_completion_cost: str | None = None
    model_token_limit: str | None = None

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> ModelConfiguration:
        """Build the configuration the operator passed on the command line."""
        return cls(
            model=arguments.model,
            model_provider=arguments.model_provider,
            model_prompt_cost=arguments.model_prompt_cost,
            model_completion_cost=arguments.model_completion_cost,
            model_token_limit=arguments.model_token_limit,
        )

    @classmethod
    def from_provenance(cls, provenance: JsonDict) -> ModelConfiguration:
        """Build the configuration a committed sidecar (or store row) describes."""
        return cls(
            model=provenance.get("model"),
            model_provider=provenance.get("model_provider"),
            model_prompt_cost=provenance.get("model_prompt_cost"),
            model_completion_cost=provenance.get("model_completion_cost"),
            # The token limit is a deployment budget, not a baseline property, so
            # provenance does not carry it and a sync never rewrites it.
            model_token_limit=None,
        )

    @classmethod
    def from_context(cls, context: GlobalContext) -> ModelConfiguration:
        """Build the configuration the running process holds, for recording as provenance."""
        return cls(
            model=(context.model or "").strip() or None,
            model_provider=(context.model_provider or "").strip() or None,
            model_prompt_cost=format_cost_for_environment_file(
                context.model_prompt_cost or 0.0
            ),
            model_completion_cost=format_cost_for_environment_file(
                context.model_completion_cost or 0.0
            ),
            model_token_limit=None,
        )

    def merged_over(self, fallback: ModelConfiguration) -> ModelConfiguration:
        """Return this configuration with every unset field taken from ``fallback``."""
        return ModelConfiguration(
            **{
                field.name: getattr(self, field.name)
                if getattr(self, field.name) is not None
                else getattr(fallback, field.name)
                for field in fields(self)
            }
        )

    def provided_items(self) -> List[Tuple[str, str]]:
        """``(ENV_NAME, value)`` for every field that was actually given."""
        return [
            (field.name.upper(), getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        ]

    def validate(self) -> None:
        """Reject values that would be written and then crash the API at boot."""
        for name in ("model_prompt_cost", "model_completion_cost"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                parsed = float(value)
            except ValueError as error:
                raise SystemExit(f"--{name.replace('_', '-')} must be a number, got {value!r}") from error
            if parsed < 0:
                raise SystemExit(f"--{name.replace('_', '-')} must not be negative, got {value!r}")
        if self.model_token_limit is not None:
            try:
                if int(self.model_token_limit) <= 0:
                    raise ValueError
            except ValueError as error:
                raise SystemExit(
                    f"--model-token-limit must be a positive integer, got {self.model_token_limit!r}"
                ) from error
        if self.model is not None and not self.model.strip():
            raise SystemExit("--model must not be empty")


def write_model_configuration_to_environment_files(
    configuration: ModelConfiguration,
    environment_file_paths: List[Path] | None = None,
) -> None:
    """Rewrite every provided model line in ``.env`` and ``.env.dev``.

    Only the changed ``NAME: old -> new`` pairs are printed — the env files hold
    secrets, so no other line is ever echoed.
    """
    provided_items = configuration.provided_items()
    if not provided_items:
        print("  Nothing to rewrite (no model configuration given).")
        return
    for environment_file_path in environment_file_paths or ENVIRONMENT_FILE_PATHS:
        if not environment_file_path.exists():
            print(f"  SKIPPED (absent): {environment_file_path.name}")
            continue
        text = environment_file_path.read_text(encoding="utf-8")
        for name, value in provided_items:
            pattern, line_format = MODEL_CONFIGURATION_ANCHORS[name]
            previous_match = re.search(pattern, text, flags=re.MULTILINE)
            previous_line = previous_match.group(0) if previous_match else "<unset>"
            replacement_line = line_format.format(value=value)
            text = replace_assignment_value(text, pattern, replacement_line)
            print(f"  {environment_file_path.name}: {previous_line} -> {replacement_line}")
        environment_file_path.write_text(text, encoding="utf-8")


def apply_model_configuration_to_process_environment(
    configuration: ModelConfiguration,
) -> None:
    """Make the new values visible to THIS process.

    ``init_model`` re-reads ``GlobalContext()`` (which reads ``os.environ``) and
    discards any context it is handed, so the only way to generate the corpus
    with the new model in the same run that configured it is to set the process
    environment before anything constructs a context.
    """
    for name, value in configuration.provided_items():
        os.environ[name] = value


def load_environment_file_into_process(environment_file_path: Path) -> None:
    """Load an env file WITHOUT overriding what the process already carries.

    Inside the container every variable is already baked in, so this is a no-op
    there; on the host it supplies ``MODEL``, the provider key, and the store URI
    without exports. Command-line overrides are applied to ``os.environ`` first,
    and ``override=False`` guarantees they win over the file.
    """
    from dotenv import load_dotenv

    if environment_file_path.exists():
        load_dotenv(environment_file_path, override=False)
        print(f"  Loaded {environment_file_path.name} (without overriding the process env)")
    else:
        print(f"  SKIPPED (absent): {environment_file_path.name}")


def assert_process_model_matches_environment_file(environment_file_path: Path) -> None:
    """Refuse to retrain against a container env that is behind the env file.

    Compose bakes ``env_file`` into a container at creation. An operator who edits
    ``MODEL`` in the file and then runs this script in the still-running container
    would regenerate the corpus with the OLD model while the file — and the next
    recreate — say the new one. Passing ``--model`` sidesteps this entirely (the
    process env is set explicitly); without it the two must agree.
    """
    from dotenv import dotenv_values

    if not environment_file_path.exists():
        return
    file_model = (dotenv_values(environment_file_path).get("MODEL") or "").strip()
    process_model = (os.environ.get("MODEL") or "").strip()
    if file_model and process_model and file_model != process_model:
        raise SystemExit(
            f"The process env holds MODEL={process_model!r} but "
            f"{environment_file_path.name} says MODEL={file_model!r}: this container's "
            "env is stale. Pass --model explicitly, or recreate the container so it "
            "picks up the edited env file, then re-run."
        )


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
            "inference model; pass --model or set MODEL in .env before retraining."
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

    async def _answer_one(question: str) -> JsonDict | None:
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

    conversations: List[JsonDict] = []
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
    # NamedTemporaryFile creates the file mode 0600; the corpus is a tracked source
    # file that host tools (git, tests) must read after an in-container run.
    os.chmod(corpus_path, 0o644)

    print(
        f"Wrote {len(conversations)} conversations to "
        f"{_display_path(corpus_path)}"
    )
    return len(conversations)


# ---------------------------------------------------------------------------
# Stage 3b — the provenance sidecar
# ---------------------------------------------------------------------------
def compute_style_system_prompt_sha256() -> str:
    """Fingerprint of the frozen generation prompt, recorded with every corpus."""
    return hashlib.sha256(STYLE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_baseline_provenance(
    configuration: ModelConfiguration,
    question_count: int,
    conversation_count: int,
    build_result: BaselineBuildResult,
) -> JsonDict:
    """Assemble the sidecar for the retrain that just ran.

    Everything a reader needs to answer "what is this baseline?" without opening
    the pickles: the model and provider that answered, the per-token costs in
    force at the time, when, how many replies, the threshold the fit yielded, the
    feature-vector contract the artifacts are shaped to, and the library versions
    the pickles were written under (sklearn pickles are not portable across
    releases).
    """
    import shap  # type: ignore[import-untyped]
    import sklearn  # type: ignore[import-untyped]

    return {
        "schema_version": BASELINE_PROVENANCE_SCHEMA_VERSION,
        "model": configuration.model,
        "model_provider": configuration.model_provider,
        "model_prompt_cost": configuration.model_prompt_cost,
        "model_completion_cost": configuration.model_completion_cost,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "question_count": int(question_count),
        "conversation_count": int(conversation_count),
        "baseline_response_threshold": float(build_result.baseline_response_threshold),
        "style_feature_vector_version": STYLE_FEATURE_VECTOR_VERSION,
        "feature_width": int(build_result.feature_width),
        "row_count": int(build_result.row_count),
        "key_phrase_count": int(build_result.key_phrase_count),
        "scikit_learn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "style_system_prompt_sha256": compute_style_system_prompt_sha256(),
    }


def write_baseline_provenance(
    provenance: JsonDict, provenance_path: Path = BASELINE_PROVENANCE_FILE_PATH
) -> None:
    """Write the sidecar next to the corpus (pretty-printed; diffs are reviewed)."""
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  Wrote {_display_path(provenance_path)}")


def read_baseline_provenance(
    provenance_path: Path = BASELINE_PROVENANCE_FILE_PATH,
) -> JsonDict:
    """Read the sidecar, or exit non-zero naming what is missing."""
    if not provenance_path.exists():
        raise SystemExit(
            f"{_display_path(provenance_path)} does not exist. Run a full retrain "
            "first (it writes the sidecar), or pull a commit that carries one."
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{_display_path(provenance_path)} is not valid JSON: {error}") from error
    if not isinstance(provenance, dict) or not provenance.get("model"):
        raise SystemExit(f"{_display_path(provenance_path)} does not record a model.")
    return provenance


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
# Stage 5 — publish to (or purge from) the Postgres store
# ---------------------------------------------------------------------------
class StoreCachePurgeError(RuntimeError):
    """The disk artifacts were rebuilt but the store could not be updated."""


def _redact_credentials(store_uri: str) -> str:
    """Strip user:password from a connection URI so it is safe to print."""
    return re.sub(r"://[^@/]*@", "://***:***@", store_uri)


def _connect_to_store(context: GlobalContext) -> psycopg.Connection[Any]:
    """Open a plain autocommit psycopg connection to the LangGraph store.

    Uses psycopg rather than ``AsyncPostgresStore``: constructing the store
    requires the 640-dim ``IndexConfig``, which would load the HuggingFace
    embedding model just to issue a handful of statements.
    """
    import psycopg

    store_uri = (context.async_postgres_store_uri or "").strip()
    if not store_uri:
        raise SystemExit(
            "ASYNC_POSTGRES_STORE_URI is not set, so the store cannot be updated. "
            "Set it, or pass --skip-store-publish and clear these keys yourself: "
            f"{', '.join(BASELINE_STORE_CACHE_KEYS)}"
        )
    try:
        return psycopg.connect(store_uri, autocommit=True, connect_timeout=15)
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


def read_artifact_store_values_from_disk() -> Dict[str, str]:
    """Read the four artifact rows' string values from the files just written.

    Each is encoded exactly as the runtime's read-through cache would encode it
    (``graph.py`` / ``utility.py``): the matrix as a JSON list-of-lists, the two
    pickles as their base64 text, the key phrases as a JSON list.
    """
    import numpy as np

    feature_matrix = np.load(REPOSITORY_ROOT / BASELINE_FEATURES_ARR_PATH, allow_pickle=False)
    return {
        "baseline_features_arr_list_str": json.dumps(feature_matrix.tolist()),
        "baseline_features_model_b64_pkl": (
            REPOSITORY_ROOT / BASELINE_FEATURES_MODEL_PATH
        ).read_bytes().decode("utf-8"),
        "baseline_features_explainer_b64_pkl": (
            REPOSITORY_ROOT / BASELINE_FEATURES_EXPLAINER_PATH
        ).read_bytes().decode("utf-8"),
        "baseline_key_phrase_profile": json.dumps(
            json.loads((REPOSITORY_ROOT / BASELINE_KEY_PHRASES_PATH).read_text(encoding="utf-8"))
        ),
    }


def publish_baseline_to_store(context: GlobalContext, provenance: JsonDict) -> None:
    """Upsert the five baseline rows and release the retrain lock, atomically.

    Rebuilding the files on disk is NOT enough. Each artifact is cached in the
    store on first use, and the runtime's self-heal (``baseline_feature_array_is_current``
    and the ``n_features_in_`` checks in ``utility.py``) only reloads from disk when
    the feature-vector WIDTH changes. A model-upgrade retrain leaves the width
    unchanged, so without this the running deployment keeps serving the previous
    model's cloud indefinitely.

    WRITING the rows (rather than deleting them and letting the cache refill) is
    what makes the retrain happen once: dev and prod share this store, and a
    purge would let whichever checkout scored the next reply refill the cache from
    ITS artifacts — possibly the old ones. Publishing the freshly built values
    leaves nothing for a stale checkout to refill.
    """
    store_values = read_artifact_store_values_from_disk()
    store_values[BASELINE_PROVENANCE_STORE_KEY] = json.dumps(provenance, sort_keys=True)

    connection = _connect_to_store(context)
    with connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                for cache_key, value in store_values.items():
                    cursor.execute(
                        SQL_UPSERT_STORE_ROW,
                        (cache_key, cache_key, json.dumps({"value": value})),
                    )
                    print(f"  {cache_key}: published ({len(value)} chars)")
                cursor.execute(
                    SQL_DELETE_STORE_ROW,
                    (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY),
                )
                print(f"  {BASELINE_RETRAIN_LOCK_STORE_KEY}: {cursor.rowcount} row(s) released")


def purge_baseline_store_cache(context: GlobalContext) -> None:
    """Delete the cached baseline rows so the read-through cache refills from disk.

    The fallback for when the rows cannot be published (see
    :func:`publish_baseline_to_store` for why publishing is preferred).
    """
    connection = _connect_to_store(context)
    with connection:
        with connection.cursor() as cursor:
            for cache_key in BASELINE_STORE_CACHE_KEYS:
                cursor.execute(SQL_DELETE_STORE_ROW, (cache_key, cache_key))
                print(f"  {cache_key}: {cursor.rowcount} row(s) deleted")


def adopt_baseline_from_store(context: GlobalContext) -> JsonDict:
    """Write this checkout's artifacts and sidecar from what the store serves.

    The mirror image of :func:`publish_baseline_to_store`, for a checkout whose
    sibling container already retrained the configured MODEL: no model calls,
    just the four files plus the sidecar, so this checkout's committed record
    matches the cloud every container is scoring against.
    """
    import numpy as np

    connection = _connect_to_store(context)
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(SQL_SELECT_STORE_ROWS, (BASELINE_STORE_CACHE_KEYS,))
            raw_rows: Dict[str, Any] = {
                key: (value or {}).get("value") for key, value in cursor.fetchall()
            }

    missing = [key for key in BASELINE_STORE_CACHE_KEYS if not raw_rows.get(key)]
    if missing:
        raise SystemExit(
            "The store does not hold a complete published baseline; missing: "
            f"{', '.join(missing)}. Run a full retrain instead."
        )
    rows: Dict[str, str] = {key: str(raw_rows[key]) for key in BASELINE_STORE_CACHE_KEYS}
    provenance: JsonDict = json.loads(rows[BASELINE_PROVENANCE_STORE_KEY])

    np.save(
        REPOSITORY_ROOT / BASELINE_FEATURES_ARR_PATH,
        np.asarray(json.loads(rows["baseline_features_arr_list_str"]), dtype=np.float64),
        allow_pickle=False,
    )
    (REPOSITORY_ROOT / BASELINE_FEATURES_MODEL_PATH).write_bytes(
        rows["baseline_features_model_b64_pkl"].encode("utf-8")
    )
    (REPOSITORY_ROOT / BASELINE_FEATURES_EXPLAINER_PATH).write_bytes(
        rows["baseline_features_explainer_b64_pkl"].encode("utf-8")
    )
    (REPOSITORY_ROOT / BASELINE_KEY_PHRASES_PATH).write_text(
        json.dumps(json.loads(rows["baseline_key_phrase_profile"]), indent=2),
        encoding="utf-8",
    )
    for relative_path in (
        BASELINE_FEATURES_ARR_PATH,
        BASELINE_FEATURES_MODEL_PATH,
        BASELINE_FEATURES_EXPLAINER_PATH,
        BASELINE_KEY_PHRASES_PATH,
    ):
        print(f"  Wrote {relative_path}")
    write_baseline_provenance(provenance)
    return provenance


# ---------------------------------------------------------------------------
# Stage 6 — verify what was written is internally consistent
# ---------------------------------------------------------------------------
def verify_written_artifacts(expected_model: str | None = None) -> None:
    """Reload every written artifact and assert the widths and provenance agree.

    Catches the failure mode where one artifact is refit and another is not: the
    matrix, the forest, and the explainer background must all describe the same
    feature vector or the runtime raises mid-request — and the sidecar must name
    the model this run was for, or the next boot retrains again.
    """
    import base64
    import pickle

    import numpy as np
    from dotenv import dotenv_values

    from src.anubis.utils.dataset.style_features import (
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

    provenance = read_baseline_provenance()
    if expected_model is not None and provenance.get("model") != expected_model:
        raise SystemExit(
            f"Sidecar records model {provenance.get('model')!r} but this run was for "
            f"{expected_model!r}"
        )
    if provenance.get("style_feature_vector_version") != STYLE_FEATURE_VECTOR_VERSION:
        raise SystemExit(
            f"Sidecar records feature-vector version {provenance.get('style_feature_vector_version')!r}; "
            f"code is at {STYLE_FEATURE_VECTOR_VERSION}"
        )
    for environment_file_path in ENVIRONMENT_FILE_PATHS:
        if not environment_file_path.exists():
            continue
        environment_threshold = dotenv_values(environment_file_path).get(
            "BASELINE_RESPONSE_THRESHOLD"
        )
        if environment_threshold and float(environment_threshold) != float(
            provenance["baseline_response_threshold"]
        ):
            raise SystemExit(
                f"{environment_file_path.name} BASELINE_RESPONSE_THRESHOLD={environment_threshold} "
                f"disagrees with the sidecar ({provenance['baseline_response_threshold']!r})"
            )

    print(f"  matrix:      {feature_matrix.shape}")
    print(f"  forest:      fit on {model_width} features")
    print(f"  explainer:   background {explainer_background.shape}")
    print(f"  key phrases: {len(key_phrases)}")
    print(f"  feature-vector version: {STYLE_FEATURE_VECTOR_VERSION}")
    print(f"  provenance:  model {provenance.get('model')!r}, threshold {provenance.get('baseline_response_threshold')!r}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _print_manual_purge_instructions() -> None:
    """Print the exact rows to delete when the automatic publish did not run."""
    print("    DELETE FROM store WHERE prefix = key AND key IN (")
    print(
        "      "
        + ", ".join(f"'{cache_key}'" for cache_key in BASELINE_STORE_CACHE_KEYS)
    )
    print("    );")


def parse_command_line_arguments(argument_list: List[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``argument_list`` lets tests bypass ``sys.argv``."""
    parser = argparse.ArgumentParser(
        description=(
            "Switch the inference model (optionally), regenerate the "
            "unmodified-inference-model baseline corpus with it, refit the bundled "
            "stylometry artifacts, write the recalibrated threshold and provenance, "
            "and publish the result to the store."
        )
    )
    parser.add_argument("--model", help='New MODEL, e.g. gpt-5.6-luna (written as MODEL="...").')
    parser.add_argument("--model-provider", help="New MODEL_PROVIDER, e.g. OPEN_AI.")
    parser.add_argument(
        "--model-prompt-cost",
        help="New MODEL_PROMPT_COST in dollars per SINGLE input token, e.g. 0.0000002.",
    )
    parser.add_argument(
        "--model-completion-cost",
        help="New MODEL_COMPLETION_COST in dollars per SINGLE output token, e.g. 0.0000012.",
    )
    parser.add_argument(
        "--model-token-limit",
        help="New MODEL_TOKEN_LIMIT (context budget in tokens); left alone when omitted.",
    )
    parser.add_argument(
        "--environment-file",
        default=".env",
        help=(
            "Env file loaded into this process (without overriding it) and checked "
            "against the process MODEL. Rewrites always target both .env and .env.dev."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--configuration-only",
        action="store_true",
        help=(
            "No model calls: from the committed sidecar, rewrite the env lines and "
            "threshold and publish the committed artifacts to the store."
        ),
    )
    mode_group.add_argument(
        "--adopt-from-store",
        action="store_true",
        help=(
            "No model calls: write this checkout's artifacts, sidecar, env lines, and "
            "threshold from the baseline the store already serves."
        ),
    )
    parser.add_argument(
        "--skip-store-publish",
        "--skip-store-purge",
        dest="skip_store_publish",
        action="store_true",
        help=(
            "Do not touch the Postgres store. Files on disk are still rebuilt, but "
            "running deployments keep serving the PREVIOUS baseline until the rows "
            "are published or cleared by hand."
        ),
    )
    return parser.parse_args(argument_list)


def _apply_configuration_and_threshold(configuration: ModelConfiguration, threshold: float) -> None:
    write_model_configuration_to_environment_files(configuration)
    write_threshold_to_environment_files(threshold)
    write_threshold_to_context_default(threshold)


def _publish_or_report(
    context: GlobalContext, provenance: JsonDict, skip: bool
) -> StoreCachePurgeError | None:
    """Stage 5 with the failure kept, so the run can finish and then exit non-zero."""
    if skip:
        print("  SKIPPED (--skip-store-publish).")
        _print_manual_purge_instructions()
        return None
    try:
        publish_baseline_to_store(context, provenance)
    except StoreCachePurgeError as error:
        print(f"  FAILED: {error}")
        print()
        print("  The corpus, artifacts, sidecar, and threshold ARE written and correct.")
        print("  Only the store still serves the previous baseline. The committed")
        print("  store URI names host.docker.internal, which resolves inside the")
        print("  compose network but not on the host — so either run this script from")
        print("  inside the container (make retrain_baseline), or point")
        print("  ASYNC_POSTGRES_STORE_URI at a host-reachable address and re-run with")
        print("  --configuration-only. Failing that, delete these rows by hand so the")
        print("  cache refills from disk:")
        _print_manual_purge_instructions()
        return error
    return None


def run_adopt_from_store(arguments: argparse.Namespace) -> None:
    """``--adopt-from-store``: files and env from the store, no model calls."""
    _print_stage(1, 3, "Load the environment")
    load_environment_file_into_process(REPOSITORY_ROOT / arguments.environment_file)
    context = GlobalContext()

    _print_stage(2, 3, "Adopt the baseline the store serves")
    try:
        provenance = adopt_baseline_from_store(context)
    except StoreCachePurgeError as error:
        raise SystemExit(f"Could not read the store: {error}") from error
    configuration = ModelConfiguration.from_provenance(provenance)
    _apply_configuration_and_threshold(
        configuration, float(provenance["baseline_response_threshold"])
    )

    _print_stage(3, 3, "Verify the written artifacts")
    verify_written_artifacts(expected_model=provenance["model"])
    print()
    print(f"Adopted the store's baseline for {provenance['model']!r}. Recreate the container to load the rewritten env.")


def run_configuration_only(arguments: argparse.Namespace) -> None:
    """``--configuration-only``: env + threshold + store from the committed sidecar."""
    _print_stage(1, 3, "Apply the committed sidecar to the env files and threshold")
    provenance = read_baseline_provenance()
    configuration = ModelConfiguration.from_provenance(provenance)
    print(f"  sidecar model: {provenance['model']!r}")
    _apply_configuration_and_threshold(
        configuration, float(provenance["baseline_response_threshold"])
    )
    apply_model_configuration_to_process_environment(configuration)
    load_environment_file_into_process(REPOSITORY_ROOT / arguments.environment_file)
    context = GlobalContext()

    _print_stage(2, 3, "Publish the committed artifacts to the store")
    publish_failure = _publish_or_report(context, provenance, arguments.skip_store_publish)

    _print_stage(3, 3, "Verify the written artifacts")
    verify_written_artifacts(expected_model=provenance["model"])
    print()
    print(f"Configuration synced to the committed baseline for {provenance['model']!r}.")
    if publish_failure is not None:
        raise SystemExit("Sync incomplete: the store was NOT updated.")


def run_full_retrain(arguments: argparse.Namespace) -> None:
    """Run the default mode: (re)configure, regenerate, refit, record, publish, verify."""
    total_stages = 7

    _print_stage(0, total_stages, "Apply the model configuration")
    requested_configuration = ModelConfiguration.from_arguments(arguments)
    requested_configuration.validate()
    write_model_configuration_to_environment_files(requested_configuration)
    apply_model_configuration_to_process_environment(requested_configuration)
    environment_file_path = REPOSITORY_ROOT / arguments.environment_file
    load_environment_file_into_process(environment_file_path)
    if requested_configuration.model is None:
        assert_process_model_matches_environment_file(environment_file_path)
    context = GlobalContext()
    # What the sidecar will record: the operator's explicit values, else what the
    # process is configured with (so an automatic, flag-less retrain still records
    # the costs in force).
    recorded_configuration = requested_configuration.merged_over(
        ModelConfiguration.from_context(context)
    )

    _print_stage(1, total_stages, "Resolve the configured inference model")
    model_provider, model_name = resolve_inference_model(context)
    print(f"  MODEL_PROVIDER        = {model_provider}")
    print(f"  MODEL                 = {model_name}")
    print(f"  MODEL_PROMPT_COST     = {recorded_configuration.model_prompt_cost}")
    print(f"  MODEL_COMPLETION_COST = {recorded_configuration.model_completion_cost}")
    if requested_configuration.model_provider is not None:
        print(
            "  NOTE: a provider change usually also needs LLM_PROVIDER_API_KEY / "
            "LLM_PROVIDER_BASE_URL; this script never manages secrets."
        )

    _print_stage(2, total_stages, "Regenerate the baseline corpus")
    conversation_count = asyncio.run(generate_baseline_corpus(BASELINE_CORPUS_PATH, context))

    _print_stage(3, total_stages, "Refit the bundled stylometry artifacts")
    build_result = build(BASELINE_CORPUS_PATH)

    _print_stage(4, total_stages, "Record provenance and write the recalibrated threshold")
    provenance = build_baseline_provenance(
        recorded_configuration,
        question_count=len(ALL_STANDARDIZED_QUESTIONS),
        conversation_count=conversation_count,
        build_result=build_result,
    )
    write_baseline_provenance(provenance)
    write_threshold_to_environment_files(build_result.baseline_response_threshold)
    write_threshold_to_context_default(build_result.baseline_response_threshold)

    _print_stage(5, total_stages, "Publish the baseline to the store")
    publish_failure = _publish_or_report(context, provenance, arguments.skip_store_publish)

    _print_stage(6, total_stages, "Verify the written artifacts")
    verify_written_artifacts(expected_model=model_name)

    print()
    print(
        f"Retrained against {model_name}. BASELINE_RESPONSE_THRESHOLD = "
        f"{build_result.baseline_response_threshold!r}"
    )
    print("Commit the rebuilt artifacts, the corpus, the sidecar, and the context.py")
    print("threshold together — they are only meaningful as a set. Then recreate the")
    print("API container so it loads the rewritten env (make recreate_dev_api).")

    if publish_failure is not None:
        raise SystemExit(
            "Retrain incomplete: the store was NOT updated, so running deployments "
            "still serve the previous baseline."
        )


def main(argument_list: List[str] | None = None) -> None:
    """Dispatch to the mode the command line selects."""
    arguments = parse_command_line_arguments(argument_list)
    if arguments.adopt_from_store:
        run_adopt_from_store(arguments)
    elif arguments.configuration_only:
        run_configuration_only(arguments)
    else:
        run_full_retrain(arguments)


if __name__ == "__main__":
    main()
