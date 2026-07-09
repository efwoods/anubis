"""Backend assembly + storage policies for the data-analysis capability.

Architecture (verified against ``deepagents`` 0.6.x source):

- ``CompositeBackend.execute()`` delegates **only** to the ``default``
  backend, and files routed to a ``StoreBackend`` are a virtual store that a
  real shell cannot see. The default therefore must be the execution
  backend — a ``LocalShellBackend`` rooted in a per-turn temporary workspace
  inside this container (the container is the isolation boundary). Hosted
  sandbox providers (Daytona / Modal / E2B / …) are a future env-selected
  swap; their documented teardown call replaces the ``rmtree`` here.
- Two ``StoreBackend`` routes give per-user-per-avatar persistence across
  threads, with **ingested** source data separated from **created**
  artifacts:

  - ``/data_ingested/`` → store namespace ``(user_id, assistant_id, "data_ingested")``
  - ``/data_created/``  → store namespace ``(user_id, assistant_id, "data_created")``

  ``CompositeBackend`` strips the route prefix, so a file written to
  ``/data_ingested/health1.json`` is stored under key ``/health1.json``
  inside the namespace. Direct ``store.aput`` writes in
  ``analysis_tools.py`` follow the same key convention so the deep agent's
  ``ls`` / ``read_file`` views stay consistent.

Persistence policy: store items persist across threads until a re-fetch from
the Model Context Protocol server observes a changed source file (recorded
``source_modified_at`` differs) — see ``analysis_tools.ingest_data_files`` —
with the byte quota + age limit below as backstop eviction.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

INGESTED_ROUTE = "/data_ingested/"
CREATED_ROUTE = "/data_created/"
INGESTED_NAMESPACE_KIND = "data_ingested"
CREATED_NAMESPACE_KIND = "data_created"
MCP_CONNECTION_NAMESPACE_KIND = "mcp_connection"
MCP_CONNECTION_DECLINED_NAMESPACE_KIND = "mcp_connection_declined"

# Directory inside the workspace where ingested/hydrated source files land,
# so shell commands use relative paths like ``work/health1.json``.
WORKSPACE_DATA_DIRECTORY = "work"


def ingested_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Store namespace for durable ingested source data (per user, per avatar)."""
    return (user_id, assistant_id, INGESTED_NAMESPACE_KIND)


def created_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Store namespace for durable created artifacts (per user, per avatar)."""
    return (user_id, assistant_id, CREATED_NAMESPACE_KIND)


def mcp_connection_namespace(user_id: str) -> tuple[str, str]:
    """Store namespace for the single MCP connection a user may establish.

    Keyed by user only (a two-element tuple, unlike the per-avatar data
    namespaces above) so that exactly one connection record can exist per
    user — the "single MCP connection per user" rule. The record's value
    records which single avatar the connection is bound to.
    """
    return (user_id, MCP_CONNECTION_NAMESPACE_KIND)


def mcp_connection_declined_namespace(
    user_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Store namespace for a per-avatar 'do not offer again' decline marker.

    Per user *and* avatar so declining the offer on one avatar never
    suppresses the offer on the user's other avatars (e.g. declining on a
    test avatar must not prevent connecting the personal avatar later).
    """
    return (user_id, assistant_id, MCP_CONNECTION_DECLINED_NAMESPACE_KIND)


@dataclass
class AnalysisBackendBundle:
    """Everything ``think`` needs to run and then tear down one analysis turn."""

    backend: Any
    """The ``CompositeBackend`` passed to ``create_deep_agent``."""

    workspace_path: Path
    """Per-turn ephemeral directory the local shell executes inside."""

    execution_backend_name: str
    """Which execution backend is active (``local_shell`` today)."""

    user_id: str
    assistant_id: str


def build_analysis_backend(
    context: GlobalContext,
    user_id: str,
    assistant_id: str,
    store: Any | None = None,
) -> AnalysisBackendBundle:
    """Assemble the per-turn ``CompositeBackend`` for data analysis.

    Args:
        context: Global context carrying the ``data_analysis_*`` settings.
        user_id: Authenticated user identifier (from config ``configurable``).
        assistant_id: Selected avatar identifier.
        store: The cross-thread ``BaseStore`` (``runtime.store`` in the
            ``think`` node). Passed explicitly so the ``StoreBackend`` does
            not depend on implicit context resolution.
    """
    # Heavy-ish import kept out of module scope per the cold-start rule.
    from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend

    execution_backend_name = (context.data_analysis_execution_backend or "local_shell").lower()
    if execution_backend_name != "local_shell":
        raise NotImplementedError(
            "Execution backend "
            f"{execution_backend_name!r} is not wired yet; only 'local_shell' "
            "is supported. Hosted sandbox providers (Daytona / Modal / E2B) "
            "are a planned swap — their teardown call belongs in "
            "cleanup_analysis_workspace."
        )

    workspace_path = (
        Path(context.data_analysis_workspace_root) / f"turn-{uuid.uuid4().hex}"
    )
    (workspace_path / WORKSPACE_DATA_DIRECTORY).mkdir(parents=True, exist_ok=True)

    execution_backend = LocalShellBackend(
        root_dir=str(workspace_path),
        virtual_mode=True,
        inherit_env=True,
    )

    # The namespace factories close over the ids resolved in ``think`` (the
    # runtime argument is ignored): isolation comes from the authenticated
    # (user_id, assistant_id) pair, never from model-controlled input.
    composite_backend = CompositeBackend(
        default=execution_backend,
        routes={
            INGESTED_ROUTE: StoreBackend(
                store=store,
                namespace=lambda _runtime, ns=ingested_namespace(user_id, assistant_id): ns,
            ),
            CREATED_ROUTE: StoreBackend(
                store=store,
                namespace=lambda _runtime, ns=created_namespace(user_id, assistant_id): ns,
            ),
        },
    )

    return AnalysisBackendBundle(
        backend=composite_backend,
        workspace_path=workspace_path,
        execution_backend_name=execution_backend_name,
        user_id=user_id,
        assistant_id=assistant_id,
    )


def cleanup_analysis_workspace(bundle: AnalysisBackendBundle) -> None:
    """Tear down the per-turn execution workspace (the sandbox-clean policy).

    For ``local_shell`` the workspace directory is removed. When a hosted
    sandbox provider is added, this is where the provider's documented
    teardown call (``.stop()`` / ``.kill()`` / ``.terminate()`` /
    ``.shutdown()``) belongs.
    """
    if bundle.execution_backend_name == "local_shell":
        shutil.rmtree(bundle.workspace_path, ignore_errors=True)
        logger.debug("Cleaned analysis workspace %s", bundle.workspace_path)


async def enforce_ingested_quota(
    store: Any,
    user_id: str,
    assistant_id: str,
    context: GlobalContext,
) -> list[str]:
    """Evict ingested-buffer items beyond the age limit and byte quota.

    Policy (backstop to the persist-until-updated rule):

    1. Delete items whose store ``updated_at`` is older than
       ``data_analysis_store_max_age_days``.
    2. If the remaining total content size exceeds
       ``data_analysis_store_max_bytes``, delete least-recently-updated
       items until the total fits.

    Returns the list of evicted store keys.
    """
    namespace = ingested_namespace(user_id, assistant_id)
    items = await store.asearch(namespace, limit=1000)

    now = datetime.now(UTC)
    max_age = timedelta(days=int(context.data_analysis_store_max_age_days))
    max_bytes = int(context.data_analysis_store_max_bytes)

    def item_updated_at(item: Any) -> datetime:
        updated_at = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        if updated_at is None:
            return now
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        return updated_at

    def item_size_bytes(item: Any) -> int:
        content = (item.value or {}).get("content", "")
        return len(content) if isinstance(content, str) else 0

    evicted_keys: list[str] = []
    survivors = []
    for item in items:
        if now - item_updated_at(item) > max_age:
            await store.adelete(namespace, item.key)
            evicted_keys.append(item.key)
        else:
            survivors.append(item)

    total_bytes = sum(item_size_bytes(item) for item in survivors)
    if total_bytes > max_bytes:
        # Least-recently-updated first.
        survivors.sort(key=item_updated_at)
        for item in survivors:
            if total_bytes <= max_bytes:
                break
            await store.adelete(namespace, item.key)
            evicted_keys.append(item.key)
            total_bytes -= item_size_bytes(item)

    if evicted_keys:
        logger.info(
            "Evicted %d ingested-data items for (user=%s, assistant=%s): %s",
            len(evicted_keys),
            user_id,
            assistant_id,
            evicted_keys,
        )
    return evicted_keys
