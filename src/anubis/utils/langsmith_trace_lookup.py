"""Find the LangSmith trace of a reply that was stored without a trace record.

A reply written before ``langsmith_trace_record`` existed, or by a deployment
without ``LANGSMITH_WORKSPACE_ID`` / ``LANGSMITH_PROJECT_ID``, carries no
``response_metadata["langsmith"]``. Production and development share one
Postgres, so a development client routinely opens production conversations
whose replies have no record.

Reading every root run of a thread is too slow for the transcript read
(measured on 2026-09-29 against a production thread with 203 root runs:
4,171 ms selecting only ids, 12,557 ms with the 42,912,739 bytes of run
inputs). The lookup is therefore split in two:

1. ``/conversations/{thread_id}/messages`` calls ``mark_replies_for_lookup``,
   which costs no network call: every reply without a record receives
   ``response_metadata["langsmith_lookup"]`` naming the human turn the reply
   answers (id and ``created_at``).
2. When the administrator clicks the reply's LangSmith link, the client calls
   ``/conversations/{thread_id}/langsmith_trace``, which runs
   ``find_langsmith_record_for_human_turn``: the thread's root runs that
   started in a window around the human turn's ``created_at`` are read from
   every project in ``LANGSMITH_TRACE_LOOKUP_PROJECT_IDS`` (1,212 ms measured
   for the same thread), and the root run whose inputs carry the human
   message's id is the reply's run.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# The window of root-run start times searched around the human turn's
# ``created_at``. The human message is stamped in ``/message`` just before the
# graph starts; the lower bound absorbs clock skew between the API host and
# LangSmith, and the upper bound absorbs moderation screening and media
# processing that run before the graph.
ROOT_RUN_START_WINDOW_BEFORE_HUMAN_TURN = timedelta(seconds=5)
ROOT_RUN_START_WINDOW_AFTER_HUMAN_TURN = timedelta(seconds=120)

# The lookup answers a click; a slow LangSmith must not hang the request.
LANGSMITH_TRACE_LOOKUP_TIMEOUT_SECONDS = 20.0

# LangSmith project names by project id, read once per process.
_project_names_by_project_id: dict[str, str] = {}


def lookup_project_ids(context: Any) -> list[str]:
    """Return the project ids named by ``LANGSMITH_TRACE_LOOKUP_PROJECT_IDS``, in order."""
    configured_project_ids = str(
        getattr(context, "langsmith_trace_lookup_project_ids", None) or ""
    )
    project_ids: list[str] = []
    for project_id in configured_project_ids.split(","):
        project_id = project_id.strip()
        if project_id and project_id not in project_ids:
            project_ids.append(project_id)
    return project_ids


def lookup_enabled(context: Any, user_id: str | None) -> bool:
    """Whether trace records may be looked up for this caller.

    Only a development process (``DEV=TRUE``), only for the administrator (the
    only account the client shows the LangSmith link to), and only with a
    workspace id and at least one lookup project configured.
    """
    development_mode = str(getattr(context, "dev", None) or "").strip().upper() == "TRUE"
    administrator_user_id = str(getattr(context, "admin_user_id", None) or "").strip()
    workspace_id = str(getattr(context, "langsmith_workspace_id", None) or "").strip()
    return bool(
        development_mode
        and administrator_user_id
        and str(user_id or "").strip() == administrator_user_id
        and workspace_id
        and lookup_project_ids(context)
    )


def _message_field(message: Any, field_name: str) -> Any:
    """Read ``field_name`` from a serialized message dict or a message object."""
    if isinstance(message, dict):
        if field_name in message:
            return message.get(field_name)
        # LangChain's constructor serialization: {"lc": 1, "kwargs": {...}}.
        return (message.get("kwargs") or {}).get(field_name)
    return getattr(message, field_name, None)


def mark_replies_for_lookup(messages: list[Any]) -> int:
    """Name, on each reply without a trace record, the human turn the reply answers.

    Runs before hidden turns are dropped, so a reply to a hidden ambient
    observation names that hidden human turn. Messages are dicts as the
    LangGraph client returns them and are mutated in place. Returns the number
    of replies marked.
    """
    marked_reply_count = 0
    answered_human_message: dict[str, Any] | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        message_type = _message_field(message, "type")
        if message_type == "human":
            answered_human_message = message
            continue
        if message_type != "ai" or answered_human_message is None:
            continue
        response_metadata = message.get("response_metadata")
        if not isinstance(response_metadata, dict):
            response_metadata = {}
        if isinstance(response_metadata.get("langsmith"), dict):
            continue
        human_message_id = str(_message_field(answered_human_message, "id") or "")
        if not human_message_id:
            continue
        human_additional_kwargs = (
            _message_field(answered_human_message, "additional_kwargs") or {}
        )
        response_metadata = dict(response_metadata)
        response_metadata["langsmith_lookup"] = {
            "human_message_id": human_message_id,
            "human_created_at": human_additional_kwargs.get("created_at"),
        }
        message["response_metadata"] = response_metadata
        marked_reply_count += 1
    return marked_reply_count


def human_message_ids_of_run_inputs(run_inputs: Any) -> list[str]:
    """Return the ids of the human messages a root run was started with."""
    if not isinstance(run_inputs, dict):
        return []
    input_messages = run_inputs.get("messages")
    if not isinstance(input_messages, list):
        return []
    human_message_ids: list[str] = []
    for input_message in input_messages:
        if _message_field(input_message, "type") != "human":
            continue
        human_message_id = _message_field(input_message, "id")
        if human_message_id:
            human_message_ids.append(str(human_message_id))
    return human_message_ids


def root_run_of_human_turn(
    root_runs: list[dict[str, Any]], human_message_id: str
) -> dict[str, Any] | None:
    """Return the root run whose inputs carry ``human_message_id``, or None."""
    for root_run in root_runs:
        if human_message_id in human_message_ids_of_run_inputs(root_run.get("inputs")):
            return root_run
    return None


def root_run_start_window_filter(
    thread_id: str, human_created_at: str | None
) -> str:
    """Build the LangSmith run filter for the thread's root runs near the human turn."""
    thread_filter = (
        f"eq(metadata_key, \"thread_id\"), eq(metadata_value, {json.dumps(thread_id)})"
    )
    try:
        human_created_instant = datetime.fromisoformat(str(human_created_at))
    except (TypeError, ValueError):
        # No usable instant: every root run of the thread is searched.
        return f"and({thread_filter})"
    if human_created_instant.tzinfo is None:
        human_created_instant = human_created_instant.replace(tzinfo=UTC)
    human_created_instant = human_created_instant.astimezone(UTC)
    window_start = human_created_instant - ROOT_RUN_START_WINDOW_BEFORE_HUMAN_TURN
    window_end = human_created_instant + ROOT_RUN_START_WINDOW_AFTER_HUMAN_TURN
    langsmith_time_format = "%Y-%m-%dT%H:%M:%S.%f"
    return (
        f"and({thread_filter}, "
        f"gte(start_time, \"{window_start.strftime(langsmith_time_format)}\"), "
        f"lte(start_time, \"{window_end.strftime(langsmith_time_format)}\"))"
    )


def _read_root_runs(
    run_filter: str, project_ids: list[str]
) -> list[dict[str, Any]]:
    """Read the root runs matching ``run_filter`` across ``project_ids`` (blocking)."""
    from langsmith import Client

    langsmith_client = Client()
    for project_id in project_ids:
        if project_id not in _project_names_by_project_id:
            try:
                project = langsmith_client.read_project(project_id=project_id)
                _project_names_by_project_id[project_id] = str(project.name or "")
            except Exception:  # noqa: BLE001 - the name is only a label
                logger.debug(
                    "Could not read LangSmith project %s", project_id, exc_info=True
                )
    return [
        {
            "id": str(run.id),
            "session_id": str(run.session_id or ""),
            "inputs": run.inputs,
        }
        for run in langsmith_client.list_runs(
            project_id=project_ids,
            is_root=True,
            filter=run_filter,
            select=["id", "session_id", "inputs"],
        )
    ]


async def find_langsmith_record_for_human_turn(
    *,
    thread_id: str,
    human_message_id: str,
    human_created_at: str | None,
    context: Any,
) -> dict[str, str] | None:
    """Find the LangSmith record of the reply to ``human_message_id``, or None.

    The record has the same shape ``langsmith_trace_record`` stores on a reply.
    """
    root_runs = await asyncio.wait_for(
        asyncio.to_thread(
            _read_root_runs,
            root_run_start_window_filter(thread_id, human_created_at),
            lookup_project_ids(context),
        ),
        timeout=LANGSMITH_TRACE_LOOKUP_TIMEOUT_SECONDS,
    )
    root_run = root_run_of_human_turn(root_runs, human_message_id)
    if root_run is None or not root_run.get("session_id"):
        return None
    project_id = str(root_run["session_id"])
    return {
        "workspace_id": str(context.langsmith_workspace_id).strip(),
        "project_id": project_id,
        "project_name": _project_names_by_project_id.get(project_id, ""),
        "run_id": str(root_run["id"]),
    }
