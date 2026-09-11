"""Client for the Neural Nexus adapter service: training jobs and adapter inference.

INERT SALVAGE — nothing imports this module yet. Salvaged verbatim from the
``z-anubis`` branch (commit ``d1b8473``) on 2026-09-09; see
``src/api/salvage/README.md`` for the activation checklist. Every
configuration read below already uses ``getattr`` with a default, so the
module imports and reports "not configured" without any ``GlobalContext``
field being added.

The adapter service (repository ``anubis-adapter``) trains one LoRA adapter per
``(user_id, assistant_id)`` on the user's stored prompt-completion datasets and
serves inference with that adapter attached through an OpenAI-compatible
``/v1/chat/completions`` endpoint. This module is the only place the API talks
to that service:

* ``AdapterServerClient`` — training start / status / progress / cancel and the
  adapter lookup, authenticated with the CALLER's own API key (the adapter
  service authenticates against the same Auth0 accounts), so no shared secret
  is needed and the service resolves the user itself.
* ``build_adapter_chat_model`` — a ``ChatOpenAI`` pointed at the service, with
  ``model`` set to the assistant id; the service resolves the adapter and
  answers 404 ``adapter_not_found`` when none exists, which the ``think`` node
  turns into a fallback to the standard model.
* ``server_is_healthy`` — a cached liveness probe so a turn never waits on a
  down service before deciding to fall back.

Configuration: ``ADAPTER_INFERENCE_ENABLED``, ``ADAPTER_SERVER_BASE_URL``,
``ADAPTER_SERVER_TIMEOUT_SECONDS`` (``GlobalContext``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

import httpx

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

ADAPTER_METADATA_KEY = "adapter"
ADAPTER_STATUS_TRAINING = "training"
ADAPTER_STATUS_TRAINED = "trained"
ADAPTER_STATUS_ERROR = "error"

_HEALTH_CACHE_TTL_SECONDS = 60.0
_health_cache: dict[str, tuple[float, bool]] = {}
_health_lock = asyncio.Lock()


# The message endpoint registers the caller's API key per conversation thread
# so the ``think`` node can build the adapter model with the CALLER's
# credential. Kept process-local on purpose: the graph config is checkpointed
# to PostgreSQL, and a credential must never be written there.
_TURN_CREDENTIAL_TTL_SECONDS = 600.0
_turn_user_api_keys: dict[str, tuple[float, str]] = {}


def register_turn_user_api_key(thread_id: str | None, user_api_key: str | None) -> None:
    if not thread_id or not user_api_key:
        return
    now = time.monotonic()
    stale = [
        key
        for key, (stamp, _) in _turn_user_api_keys.items()
        if now - stamp > _TURN_CREDENTIAL_TTL_SECONDS
    ]
    for key in stale:
        _turn_user_api_keys.pop(key, None)
    _turn_user_api_keys[thread_id] = (now, user_api_key)


def pop_turn_user_api_key(thread_id: str | None) -> str | None:
    if not thread_id:
        return None
    entry = _turn_user_api_keys.pop(thread_id, None)
    if entry is None:
        return None
    stamp, user_api_key = entry
    if time.monotonic() - stamp > _TURN_CREDENTIAL_TTL_SECONDS:
        return None
    return user_api_key


def adapter_inference_enabled(context: GlobalContext | None = None) -> bool:
    context = context or GlobalContext()
    enabled = (
        str(getattr(context, "adapter_inference_enabled", "") or "").strip().upper()
        == "TRUE"
    )
    return enabled and bool(getattr(context, "adapter_server_base_url", None))


def adapter_server_base_url(context: GlobalContext | None = None) -> str:
    context = context or GlobalContext()
    return str(getattr(context, "adapter_server_base_url", "") or "").rstrip("/")


def adapter_server_timeout_seconds(context: GlobalContext | None = None) -> float:
    context = context or GlobalContext()
    return float(getattr(context, "adapter_server_timeout_seconds", 120) or 120)


def avatar_adapter_metadata(assistant_metadata: dict | None) -> dict[str, Any]:
    """The ``adapter`` block of an avatar's LangGraph assistant metadata."""
    block = (assistant_metadata or {}).get(ADAPTER_METADATA_KEY)
    return dict(block) if isinstance(block, dict) else {}


def avatar_has_trained_adapter(assistant_metadata: dict | None) -> bool:
    return (
        avatar_adapter_metadata(assistant_metadata).get("status")
        == ADAPTER_STATUS_TRAINED
    )


async def server_is_healthy(context: GlobalContext | None = None) -> bool:
    """Whether the adapter service answers its liveness probe (cached one minute)."""
    base_url = adapter_server_base_url(context)
    if not base_url:
        return False
    now = time.monotonic()
    async with _health_lock:
        cached = _health_cache.get(base_url)
        if cached is not None and now - cached[0] < _HEALTH_CACHE_TTL_SECONDS:
            return cached[1]
    healthy = False
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            response = await client.get(f"{base_url}/health")
            healthy = response.status_code == 200
    except Exception as probe_error:  # noqa: BLE001 - unreachable means unhealthy
        logger.info("Adapter service health probe failed: %s", probe_error)
    async with _health_lock:
        _health_cache[base_url] = (now, healthy)
    return healthy


def forget_health_cache() -> None:
    _health_cache.clear()


class AdapterServerClient:
    """Thin HTTP client for the adapter service, acting as one user."""

    def __init__(self, context: GlobalContext | None, user_api_key: str):
        self.context = context or GlobalContext()
        self.base_url = adapter_server_base_url(self.context)
        self.user_api_key = user_api_key
        self.timeout = adapter_server_timeout_seconds(self.context)

    def _headers(self) -> dict[str, str]:
        return {"API-KEY": self.user_api_key}

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        ) as client:
            return await client.request(
                method, f"{self.base_url}{path}", headers=self._headers(), **kwargs
            )

    async def adapter_status(self, assistant_id: str) -> dict[str, Any]:
        """``{exists, s3_prefix, size_bytes, trained_at, ...}`` for the caller's adapter."""
        response = await self._request("GET", f"/adapters/{assistant_id}")
        if response.status_code == 404:
            return {"exists": False}
        response.raise_for_status()
        return response.json()

    async def start_training(
        self, assistant_id: str, *, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        form = {"assistant_id": assistant_id}
        for key, value in (parameters or {}).items():
            if value is not None:
                form[key] = str(value)
        response = await self._request("POST", "/train_adapter", data=form)
        if response.status_code == 404:
            raise AdapterTrainingDataMissing(
                response.json().get("detail") or "no training data"
            )
        response.raise_for_status()
        return response.json()

    async def training_status(self, job_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET", "/adapter_training_status", params={"job_id": job_id}
        )
        response.raise_for_status()
        return response.json()

    async def cancel_training(self, job_id: str) -> dict[str, Any]:
        response = await self._request(
            "POST", "/cancel_adapter_training_job", params={"job_id": job_id}
        )
        response.raise_for_status()
        return response.json()

    async def stream_training_progress(self, job_id: str) -> AsyncIterator[str]:
        """Raw server-sent-event lines of the training progress stream (pass-through)."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10.0)
        ) as client:
            async with client.stream(
                "GET",
                f"{self.base_url}/adapter_training_progress",
                params={"job_id": job_id},
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    yield line


class AdapterTrainingDataMissing(RuntimeError):
    """The adapter service found no prompt-completion datasets for the avatar."""


def build_adapter_chat_model(
    context: GlobalContext | None, *, user_api_key: str, assistant_id: str
):
    """A ``ChatOpenAI`` that answers through the caller's trained adapter."""
    from langchain_openai import ChatOpenAI

    context = context or GlobalContext()
    return ChatOpenAI(
        model=assistant_id,
        base_url=f"{adapter_server_base_url(context)}/v1",
        api_key=user_api_key,
        temperature=0.1,
        top_p=0.1,
        stream_usage=True,
        timeout=adapter_server_timeout_seconds(context),
        max_retries=0,
    )


__all__ = [
    "ADAPTER_METADATA_KEY",
    "ADAPTER_STATUS_ERROR",
    "ADAPTER_STATUS_TRAINED",
    "ADAPTER_STATUS_TRAINING",
    "AdapterServerClient",
    "AdapterTrainingDataMissing",
    "adapter_inference_enabled",
    "avatar_adapter_metadata",
    "avatar_has_trained_adapter",
    "build_adapter_chat_model",
    "forget_health_cache",
    "pop_turn_user_api_key",
    "register_turn_user_api_key",
    "server_is_healthy",
]
