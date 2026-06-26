"""Process-wide handles to shared LangGraph runtime resources.

The avatar deep agent is rebuilt inside the ``think`` node on every turn, but a
human-in-the-loop ``interrupt`` raised by a deep-agent tool (e.g.
``correct_identity_fact``) must survive across the outer graph's interrupt → resume
passes. That requires the deep agent to be compiled with ONE persistent
checkpointer shared across ``think`` invocations — not a fresh in-memory one each
call. The FastAPI lifespan owns the ``AsyncPostgresSaver`` (built on the shared
connection pool) and publishes it here; ``think`` reads it back.

When no checkpointer has been published (e.g. ``langgraph dev`` or unit tests that
don't run the FastAPI lifespan), ``get_deep_agent_checkpointer`` returns ``None``
and ``think`` runs the deep agent without durable interrupts.
"""

from typing import Optional

_deep_agent_checkpointer: Optional[object] = None


def set_deep_agent_checkpointer(checkpointer: object) -> None:
    """Publish the shared checkpointer the deep agent should reuse each turn."""
    global _deep_agent_checkpointer
    _deep_agent_checkpointer = checkpointer


def get_deep_agent_checkpointer() -> Optional[object]:
    """Return the shared deep-agent checkpointer, or ``None`` if unset."""
    return _deep_agent_checkpointer


# Process-wide ``SentenceTransformer`` reused across fact-correction calls. The
# fact-correction tool does sentence-level semantic matching to locate a claim buried in
# a long verbatim document, which needs to embed arbitrary sentences at runtime (the
# LangGraph store only auto-embeds whole documents at write time). Loading the model is
# expensive, so it is constructed once on first use and cached here rather than per call.
_sentence_embedder: Optional[object] = None


def get_sentence_embedder() -> object:
    """Return the shared sentence-embedding model, loading it once on first use.

    Uses the same model as the store's vector index (``GlobalContext().embedding_model``,
    default ``microsoft/harrier-oss-v1-270m``) so sentence-level similarity is on the
    same scale as document retrieval. The ``sentence_transformers`` import is local to
    keep it off the module-import cold-start path (see CLAUDE.md import conventions).
    """
    global _sentence_embedder
    if _sentence_embedder is None:
        from sentence_transformers import SentenceTransformer

        from src.anubis.utils.context import GlobalContext

        _sentence_embedder = SentenceTransformer(GlobalContext().embedding_model)
    return _sentence_embedder
