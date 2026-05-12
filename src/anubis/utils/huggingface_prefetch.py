"""Ensure Hugging Face model weights exist locally before embedding / inference paths run."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.anubis.utils.context import GlobalContext

logger = logging.getLogger(__name__)

# ``respond`` in ``graph.py`` loads this via ``transformers.pipeline``.
GO_EMOTIONS_MODEL_ID = "SamLowe/roberta-base-go_emotions"

_prefetch_lock = threading.Lock()
_prefetch_done = False


def ensure_huggingface_models_cached(context: GlobalContext) -> None:
    """Download any missing Hub snapshots for models required by the graph and store.

    Uses the process Hugging Face cache (``HF_HOME`` / default ``~/.cache/huggingface``).
    Safe to call multiple times; subsequent calls are no-ops in the same process.
    """
    global _prefetch_done
    with _prefetch_lock:
        if _prefetch_done:
            return
        from huggingface_hub import snapshot_download

        token = context.huggingface_token or None
        repo_ids = []
        for rid in (context.embedding_model, GO_EMOTIONS_MODEL_ID):
            if rid and rid not in repo_ids:
                repo_ids.append(rid)
        for repo_id in repo_ids:
            logger.info("Prefetching Hugging Face model if missing: %s", repo_id)
            snapshot_download(repo_id=repo_id, token=token)
        _prefetch_done = True
