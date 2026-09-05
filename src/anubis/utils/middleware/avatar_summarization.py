"""The avatar's conversation summarization middleware.

``create_deep_agent`` installs the deepagents ``SummarizationMiddleware``
with a task-oriented summary prompt (session intent, artifacts, next steps)
and keeps the resulting summarization event in a private state key that lives
on the deep agent's own thread. The avatar runs the deep agent on a fresh
thread every turn (see ``_deep_agent_config`` in ``src/anubis/graph.py``), so
that private event would be lost after each turn and a long conversation
would be re-summarized on every reply.

``AvatarSummarizationMiddleware`` keeps the deepagents behaviour (non-mutating
compaction applied at model-call time, history offloaded to the backend) and
adds two things:

- a conversation-oriented summary prompt that keeps relationship facts, tone,
  ambient observations (what the webcam and screen showed), and open threads;
- a public ``conversation_summary_event`` state key that mirrors the private
  event, so the outer workflow can carry the compaction across turns
  (``_run_avatar_deep_agent_turn`` forwards it in both directions).

The class keeps the name ``"SummarizationMiddleware"`` on purpose: deepagents
merges caller-supplied middleware by name and replaces the built-in instance
in place, so no second summarizer runs.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents.backends import StateBackend
from deepagents.middleware.summarization import (
    SummarizationEvent,
    SummarizationState,
    compute_summarization_defaults,
)
from deepagents.middleware.summarization import (
    SummarizationMiddleware as DeepAgentsSummarizationMiddleware,
)

try:
    from deepagents.middleware.summarization import (
        SUMMARIZATION_EVENT_KEY,
        SUMMARIZATION_SESSION_ID_KEY,
    )
except ImportError:
    # deepagents 0.7.11 (the container image and local venv) stores these on
    # SummarizationState as private fields and does not export the names.
    # 0.7.13+ (uv.lock) exports them; the values are the same field names.
    SUMMARIZATION_EVENT_KEY = "_summarization_event"
    SUMMARIZATION_SESSION_ID_KEY = "_summarization_session_id"

from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langgraph.types import Command
from typing_extensions import NotRequired

from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

CONVERSATION_SUMMARY_EVENT_KEY = "conversation_summary_event"
CONVERSATION_SUMMARY_SESSION_ID_KEY = "conversation_summary_session_id"


class AvatarSummarizationState(SummarizationState):
    """Deep-agent state with the public mirror of the summarization event."""

    conversation_summary_event: NotRequired[SummarizationEvent | None]
    conversation_summary_session_id: NotRequired[str | None]


AVATAR_CONVERSATION_SUMMARY_PROMPT = """<role>
Conversation Memory Assistant
</role>

<primary_objective>
Extract the context that lets the avatar continue this conversation with the conversation partner as if nothing had been forgotten.
</primary_objective>

<objective_information>
The conversation is nearing the number of input tokens the model can accept. The context extracted here replaces the older turns below, so keep only what still matters for the conversation to continue naturally.
</objective_information>

<instructions>
Structure the summary with the following sections. Populate each section or state "None".

## RELATIONSHIP AND FACTS

What the conversation partner shared about themselves, asked for, decided, or was promised. Names, places, dates, preferences, and commitments, stated plainly.

## TONE AND EMOTIONS

How the conversation partner has been feeling and how the avatar has been responding: register, humour, tension, reassurance.

## AMBIENT OBSERVATIONS

Turns beginning with [AMBIENT_OBSERVATION ...] describe what the conversation partner's webcam and screen showed while the conversation continued; they were noticed by the avatar, not typed by the conversation partner. Compress them into a short timeline: when, what was seen, and the decision recorded on each (ignore, respond, notify). Keep anything the avatar spoke up about or notified the conversation partner about, and drop repeated routine scenes.

## OPEN THREADS

Questions still unanswered, things the avatar offered to do, and anything the conversation partner said they would come back to.

</instructions>

The user will message you with the full message history from which you will extract the context above. Read all of it carefully and keep only the most important and relevant context.

<media_reference_information>
Conversation history may include XML media reference tags, for example:
<image url="/conversation_history/media/hash.png" />
These tags mean the original message included media that was preserved at the referenced backend path. Treat the tag and path as part of the conversation context and preserve the reference when the media could matter later.
</media_reference_information>

Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""


class AvatarSummarizationMiddleware(DeepAgentsSummarizationMiddleware):
    """Deepagents summarization with a public, cross-turn summary event."""

    state_schema = AvatarSummarizationState

    @property
    def name(self) -> str:
        """The public alias deepagents merges on; keeps the built-in's slot."""
        return "SummarizationMiddleware"

    @staticmethod
    def _seed_private_event(request: ModelRequest) -> ModelRequest:
        """Apply the public event when the private one is absent (a fresh thread)."""
        state = request.state
        if not isinstance(state, dict):
            return request
        if state.get(SUMMARIZATION_EVENT_KEY) is not None:
            return request
        public_event = state.get(CONVERSATION_SUMMARY_EVENT_KEY)
        if not public_event:
            return request
        seeded = dict(state)
        seeded[SUMMARIZATION_EVENT_KEY] = public_event
        session_id = state.get(CONVERSATION_SUMMARY_SESSION_ID_KEY)
        if session_id and not state.get(SUMMARIZATION_SESSION_ID_KEY):
            seeded[SUMMARIZATION_SESSION_ID_KEY] = session_id
        return request.override(state=seeded)

    @staticmethod
    def _mirror_public_event(
        response: ModelResponse | ExtendedModelResponse,
    ) -> ModelResponse | ExtendedModelResponse:
        """Copy a new private event onto the public key of the same update."""
        if not isinstance(response, ExtendedModelResponse) or response.command is None:
            return response
        update = getattr(response.command, "update", None)
        if not isinstance(update, dict) or SUMMARIZATION_EVENT_KEY not in update:
            return response
        mirrored = dict(update)
        mirrored[CONVERSATION_SUMMARY_EVENT_KEY] = update[SUMMARIZATION_EVENT_KEY]
        mirrored[CONVERSATION_SUMMARY_SESSION_ID_KEY] = update.get(
            SUMMARIZATION_SESSION_ID_KEY
        )
        return ExtendedModelResponse(
            model_response=response.model_response, command=Command(update=mirrored)
        )

    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Seed the private event, run the parent, mirror any new event (sync)."""
        return self._mirror_public_event(
            super().wrap_model_call(self._seed_private_event(request), handler)
        )

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Seed the private event, run the parent, mirror any new event (async)."""
        return self._mirror_public_event(
            await super().awrap_model_call(self._seed_private_event(request), handler)
        )


def clamp_summary_event(
    event: dict[str, Any] | None, *, outer_message_count: int
) -> dict[str, Any] | None:
    """Fit a deep-agent summary event to the outer conversation's message list.

    The deep agent's message list is the outer conversation followed by this
    turn's intermediate tool messages, and only the final reply is written back
    to the outer thread. A cutoff that fell inside the intermediate region
    would misalign on the next turn, so the cutoff is clamped to the outer
    count: the summary already covers everything before the cutoff, and the
    reply that follows stays verbatim.
    """
    if not isinstance(event, dict):
        return None
    try:
        cutoff = int(event.get("cutoff_index"))
    except (TypeError, ValueError):
        return None
    if event.get("summary_message") is None:
        return None
    return {**event, "cutoff_index": max(0, min(cutoff, int(outer_message_count)))}


def build_avatar_summarization_middleware(
    context: GlobalContext | None, model: Any, backend: Any | None = None
) -> AvatarSummarizationMiddleware:
    """Build the avatar's summarizer from the ``DEEP_AGENT_SUMMARIZATION_*`` settings."""
    context = context or GlobalContext()
    defaults = compute_summarization_defaults(model)
    return AvatarSummarizationMiddleware(
        model=model,
        backend=backend if backend is not None else StateBackend(),
        trigger=("tokens", int(context.deep_agent_summarization_max_tokens or 120000)),
        keep=(
            "messages",
            int(context.deep_agent_summarization_keep_last_n_messages or 20),
        ),
        summary_prompt=AVATAR_CONVERSATION_SUMMARY_PROMPT,
        trim_tokens_to_summarize=None,
        truncate_args_settings=defaults["truncate_args_settings"],
    )


__all__ = [
    "AVATAR_CONVERSATION_SUMMARY_PROMPT",
    "AvatarSummarizationMiddleware",
    "AvatarSummarizationState",
    "CONVERSATION_SUMMARY_EVENT_KEY",
    "CONVERSATION_SUMMARY_SESSION_ID_KEY",
    "SUMMARIZATION_EVENT_KEY",
    "SUMMARIZATION_SESSION_ID_KEY",
    "build_avatar_summarization_middleware",
    "clamp_summary_event",
]
