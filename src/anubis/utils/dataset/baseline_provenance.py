# src/anubis/utils/dataset/baseline_provenance.py

"""Keep the unmodified-inference-model style baseline in step with ``MODEL``.

The baseline artifacts (matrix, IsolationForest, SHAP explainer, key phrases,
threshold) are fitted on replies from ONE inference model, and they carry no
model name of their own — their feature-vector width is model-independent, so
nothing in the runtime can tell a luna cloud from a nano cloud. The provenance
sidecar ``data/unmodified_inference_model_baseline_corpus.meta.json`` (written by
``scripts/retrain_chatgpt_baseline.py``) and its copy in the shared LangGraph
store (row ``baseline_provenance``) are the records that can.

At API startup :func:`ensure_baseline_matches_model` compares those records with
the configured ``MODEL`` and, when they disagree, makes the retrain happen —
EXACTLY ONCE per model change, even though dev and prod are separate checkouts
that share one store:

1. the sidecar on disk already names ``MODEL`` → nothing to do;
2. the store's provenance row names ``MODEL`` → a sibling container already
   retrained; ADOPT its result (apply its threshold now, and write this
   checkout's files from the store) — zero model calls;
3. neither → take the ``baseline_retrain_lock`` row in the store and run the
   retrain script detached in this container; a container that finds the lock
   held polls the provenance row and adopts as soon as the retrain publishes.

The retrain runs as a DETACHED subprocess rather than in-process because the
script rewrites ``context.py`` (the threshold default), which the dev container's
source mount can hot-reload; the child survives that restart, and the restarted
API finds the lock and does not start a second run. The container interpreter
also guarantees the pinned scikit-learn the pickles must be written under.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.dataset.style_features import BASELINE_PROVENANCE_PATH

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

BASELINE_PROVENANCE_STORE_KEY = "baseline_provenance"
BASELINE_RETRAIN_LOCK_STORE_KEY = "baseline_retrain_lock"
RETRAIN_SCRIPT_RELATIVE_PATH = Path("scripts") / "retrain_chatgpt_baseline.py"
RETRAIN_LOG_RELATIVE_PATH = Path("data") / "baseline_retrain.log"

# Store rows live in the ``{"value": <str>}`` envelope the runtime's read-through
# cache writes, at their namespace root (prefix == key).
SQL_SELECT_STORE_ROW = "SELECT value FROM store WHERE prefix = %s AND key = %s;"
SQL_INSERT_STORE_ROW_IF_ABSENT = (
    "INSERT INTO store (prefix, key, value, created_at, updated_at) "
    "VALUES (%s, %s, %s::jsonb, now(), now()) ON CONFLICT (prefix, key) DO NOTHING;"
)
# Compare-and-swap on the previous value so two boots that both judged the same
# lock abandoned cannot both believe they took it over.
SQL_TAKE_OVER_STORE_ROW = (
    "UPDATE store SET value = %s::jsonb, updated_at = now() "
    "WHERE prefix = %s AND key = %s AND value = %s::jsonb;"
)
SQL_DELETE_STORE_ROW = "DELETE FROM store WHERE prefix = %s AND key = %s;"

# How long a boot keeps polling for a sibling's retrain before giving up, as a
# multiple of the lock's stale age (past that, the lock itself would have been
# taken over by a later boot).
_POLL_TIMEOUT_MULTIPLIER = 2


def repository_root() -> Path:
    """Return the checkout this module runs from (``src/anubis/utils/dataset`` → root)."""
    return Path(__file__).resolve().parents[4]


def load_baseline_provenance_from_disk(
    provenance_path: Path | None = None,
) -> JsonDict | None:
    """Return the committed sidecar, or ``None`` when absent or unparseable."""
    path = provenance_path or (repository_root() / BASELINE_PROVENANCE_PATH)
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return provenance if isinstance(provenance, dict) else None


def _unwrap_store_value(raw_value: Any) -> str | None:
    """Return the string inside a store row's ``{"value": <str>}`` envelope."""
    if isinstance(raw_value, str):
        try:
            raw_value = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    if isinstance(raw_value, dict):
        inner = raw_value.get("value")
        return inner if isinstance(inner, str) else None
    return None


async def load_baseline_provenance_from_store(pool: Any) -> JsonDict | None:
    """Return the store's provenance row, or ``None`` when absent or unreadable."""
    if pool is None:
        return None
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    SQL_SELECT_STORE_ROW,
                    (BASELINE_PROVENANCE_STORE_KEY, BASELINE_PROVENANCE_STORE_KEY),
                )
                row = await cursor.fetchone()
    except Exception as read_error:  # noqa: BLE001 - startup must not fail on this
        logger.warning("Could not read the store's baseline provenance: %s", read_error)
        return None
    if not row:
        return None
    inner = _unwrap_store_value(row[0])
    if inner is None:
        return None
    try:
        provenance = json.loads(inner)
    except json.JSONDecodeError:
        return None
    return provenance if isinstance(provenance, dict) else None


def baseline_is_stale_for_model(
    provenance: JsonDict | None, configured_model: str | None
) -> bool:
    """Return True when the baseline on record was not produced by ``configured_model``.

    No configured model means there is nothing to compare against (and nothing
    to retrain with), so that is never stale. No provenance at all IS stale: a
    baseline nobody can attribute to a model is exactly the state this module
    exists to end.
    """
    configured = (configured_model or "").strip()
    if not configured:
        return False
    if not provenance:
        return True
    return (str(provenance.get("model") or "")).strip() != configured


def _lock_value(model: str) -> JsonDict:
    return {
        "model": model,
        "owner": f"{socket.gethostname()}:{os.getpid()}",
        "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


def _lock_is_abandoned(lock: JsonDict, stale_after_seconds: int) -> bool:
    started_at = lock.get("started_at")
    try:
        started = datetime.fromisoformat(str(started_at))
    except (TypeError, ValueError):
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    age_seconds = (datetime.now(UTC) - started).total_seconds()
    return age_seconds > stale_after_seconds


async def try_acquire_retrain_lock(
    pool: Any, model: str, stale_after_seconds: int
) -> bool:
    """Take the retrain lock for ``model``; True only for the one boot that wins.

    An absent lock is inserted (``ON CONFLICT DO NOTHING`` decides the race). A
    present lock is taken over only when abandoned — older than
    ``stale_after_seconds`` — and only via a compare-and-swap on its previous
    value, so two boots judging the same lock abandoned cannot both win.
    """
    new_value = json.dumps({"value": json.dumps(_lock_value(model))})
    async with pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                SQL_INSERT_STORE_ROW_IF_ABSENT,
                (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY, new_value),
            )
            if cursor.rowcount == 1:
                return True
            await cursor.execute(
                SQL_SELECT_STORE_ROW,
                (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY),
            )
            row = await cursor.fetchone()
            if not row:
                # Released between our insert and select: try the insert once more.
                await cursor.execute(
                    SQL_INSERT_STORE_ROW_IF_ABSENT,
                    (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY, new_value),
                )
                return bool(cursor.rowcount == 1)
            existing_raw = row[0]
            existing_inner = _unwrap_store_value(existing_raw)
            try:
                existing_lock = json.loads(existing_inner) if existing_inner else {}
            except json.JSONDecodeError:
                existing_lock = {}
            if not _lock_is_abandoned(existing_lock, stale_after_seconds):
                return False
            previous_value = (
                existing_raw if isinstance(existing_raw, str) else json.dumps(existing_raw)
            )
            await cursor.execute(
                SQL_TAKE_OVER_STORE_ROW,
                (
                    new_value,
                    BASELINE_RETRAIN_LOCK_STORE_KEY,
                    BASELINE_RETRAIN_LOCK_STORE_KEY,
                    previous_value,
                ),
            )
            return bool(cursor.rowcount == 1)


async def release_retrain_lock(pool: Any) -> None:
    """Delete the lock row; best-effort, a failure is logged and never raised."""
    try:
        async with pool.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    SQL_DELETE_STORE_ROW,
                    (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY),
                )
    except Exception as release_error:  # noqa: BLE001
        logger.warning("Could not release the baseline retrain lock: %s", release_error)


def spawn_retrain_script(
    extra_arguments: list[str] | None = None,
    root: Path | None = None,
) -> subprocess.Popen[bytes]:
    """Start the retrain script detached, logging to ``data/baseline_retrain.log``."""
    root = root or repository_root()
    log_path = root / RETRAIN_LOG_RELATIVE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Not a context manager: the child inherits the descriptor and keeps writing
    # after this function returns; the parent's copy is closed right away.
    log_handle = open(log_path, "ab")  # noqa: SIM115
    try:
        command = [sys.executable, str(root / RETRAIN_SCRIPT_RELATIVE_PATH)] + list(
            extra_arguments or []
        )
        return subprocess.Popen(  # noqa: S603 - fixed command, no user input
            command,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def apply_threshold_to_process(app_state: Any, threshold: Any) -> None:
    """Adopt a published threshold without a container recreate.

    ``GlobalContext`` reads ``os.environ`` on every construction, so setting the
    variable covers every later context; the already-built ``app.state.context``
    is patched directly.
    """
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        return
    os.environ["BASELINE_RESPONSE_THRESHOLD"] = repr(threshold_value)
    context = getattr(app_state, "context", None)
    if context is not None:
        context.baseline_response_threshold = threshold_value
    logger.info("Adopted BASELINE_RESPONSE_THRESHOLD=%r for this process", threshold_value)


def _log_tail(root: Path, line_count: int = 20) -> str:
    try:
        lines = (root / RETRAIN_LOG_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "<no log>"
    return "\n".join(lines[-line_count:])


async def _adopt_published_baseline(app_state: Any, provenance: JsonDict, root: Path) -> None:
    """Step 2 of the ladder: take the sibling's result for this process and checkout."""
    apply_threshold_to_process(app_state, provenance.get("baseline_response_threshold"))
    process = spawn_retrain_script(["--adopt-from-store"], root)
    return_code = await asyncio.to_thread(process.wait)
    if return_code == 0:
        logger.info(
            "Adopted the store's baseline for MODEL=%r into this checkout", provenance.get("model")
        )
    else:
        logger.error(
            "Adopting the store's baseline failed (exit %s); files in this checkout "
            "still describe the previous model. Log tail:\n%s",
            return_code,
            _log_tail(root),
        )


async def _run_retrain_and_adopt(app_state: Any, pool: Any, model: str, root: Path) -> None:
    """Step 3 (lock held): run the retrain, then adopt its threshold here."""
    process = spawn_retrain_script([], root)
    logger.warning(
        "Baseline retrain for MODEL=%r started (pid %s); output in %s",
        model,
        process.pid,
        RETRAIN_LOG_RELATIVE_PATH,
    )
    return_code = await asyncio.to_thread(process.wait)
    if return_code == 0:
        provenance = load_baseline_provenance_from_disk(root / BASELINE_PROVENANCE_PATH)
        if provenance and not baseline_is_stale_for_model(provenance, model):
            apply_threshold_to_process(app_state, provenance.get("baseline_response_threshold"))
            logger.warning("Baseline retrain for MODEL=%r finished and was published", model)
            return
        logger.error(
            "Baseline retrain exited 0 but the sidecar does not name MODEL=%r; leaving the "
            "lock released so a later boot retries. Log tail:\n%s",
            model,
            _log_tail(root),
        )
    else:
        logger.error(
            "Baseline retrain for MODEL=%r failed (exit %s); the lock is released so a "
            "later boot retries. Log tail:\n%s",
            model,
            return_code,
            _log_tail(root),
        )
    # The script releases the lock itself on success; on failure it is ours to clear.
    await release_retrain_lock(pool)


async def _wait_for_sibling_retrain(
    app_state: Any, pool: Any, model: str, poll_seconds: int, timeout_seconds: int, root: Path
) -> None:
    """Step 3 (lock held elsewhere): adopt as soon as the sibling publishes."""
    waited = 0
    while waited < timeout_seconds:
        await asyncio.sleep(poll_seconds)
        waited += poll_seconds
        provenance = await load_baseline_provenance_from_store(pool)
        if provenance and not baseline_is_stale_for_model(provenance, model):
            await _adopt_published_baseline(app_state, provenance, root)
            return
    logger.error(
        "Gave up waiting for another container's baseline retrain for MODEL=%r after "
        "%s s; this process still scores against the previous baseline.",
        model,
        timeout_seconds,
    )


async def ensure_baseline_matches_model(app_state: Any) -> asyncio.Task[None] | None:
    """Run the ladder described in the module docstring; never raises into startup.

    Returns the background task that carries any retrain or adoption so the
    caller can keep a reference (and tests can await it); ``None`` when nothing
    had to be started.
    """
    context: GlobalContext | None = getattr(app_state, "context", None)
    pool = getattr(app_state, "pool", None)
    if context is None:
        return None
    model = (context.model or "").strip()
    if not model:
        return None
    root = repository_root()

    disk_provenance = load_baseline_provenance_from_disk(root / BASELINE_PROVENANCE_PATH)
    if not baseline_is_stale_for_model(disk_provenance, model):
        logger.info("Style baseline matches MODEL=%r", model)
        return None

    recorded_model = (disk_provenance or {}).get("model")
    if not context.baseline_auto_retrain_on_model_change:
        logger.warning(
            "Style baseline was built from MODEL=%r but the API runs MODEL=%r; automatic "
            "retraining is disabled (BASELINE_AUTO_RETRAIN_ON_MODEL_CHANGE). Run "
            "`python scripts/retrain_chatgpt_baseline.py` in this container.",
            recorded_model,
            model,
        )
        return None
    if pool is None:
        logger.warning(
            "Style baseline was built from MODEL=%r but the API runs MODEL=%r, and no "
            "store pool is available to coordinate a retrain; skipping.",
            recorded_model,
            model,
        )
        return None

    store_provenance = await load_baseline_provenance_from_store(pool)
    if store_provenance is not None and not baseline_is_stale_for_model(
        store_provenance, model
    ):
        logger.warning(
            "Style baseline on disk was built from MODEL=%r; the store already holds one "
            "for MODEL=%r — adopting it.",
            recorded_model,
            model,
        )
        return asyncio.create_task(_adopt_published_baseline(app_state, store_provenance, root))

    try:
        acquired = await try_acquire_retrain_lock(
            pool, model, int(context.baseline_auto_retrain_lock_stale_after_seconds)
        )
    except Exception as lock_error:  # noqa: BLE001 - startup must not fail on this
        logger.error("Could not take the baseline retrain lock: %s", lock_error)
        return None

    if acquired:
        logger.warning(
            "Style baseline was built from MODEL=%r but the API runs MODEL=%r; "
            "retraining ONCE in this container.",
            recorded_model,
            model,
        )
        return asyncio.create_task(_run_retrain_and_adopt(app_state, pool, model, root))

    poll_seconds = max(1, int(context.baseline_auto_retrain_poll_seconds))
    timeout_seconds = (
        int(context.baseline_auto_retrain_lock_stale_after_seconds) * _POLL_TIMEOUT_MULTIPLIER
    )
    logger.warning(
        "Style baseline was built from MODEL=%r but the API runs MODEL=%r; another "
        "container holds the retrain lock, polling the store every %s s to adopt its result.",
        recorded_model,
        model,
        poll_seconds,
    )
    return asyncio.create_task(
        _wait_for_sibling_retrain(app_state, pool, model, poll_seconds, timeout_seconds, root)
    )
