"""Recalibrate the per-avatar "direct quote" ground-truth cloud after an upload.

Maintains the per-document stylometric feature corpus, discovers/stores the
avatar's signature key phrases (vectorstore namespace + prompt-injectable
profile blob), and recalibrates the empirical threshold + IsolationForest.
"""

import logging
import uuid
from typing import Any, Dict, List

from langchain_core.documents import Document
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

# Minimum number of corpus rows required before the leave-one-out empirical
# distribution + IsolationForest can be calibrated. The leave-one-out step
# (compute_empirical_distribution) drops one row and fits StandardScaler +
# LedoitWolf on the rest, so with too few rows the per-column std collapses to
# zero (nan) and np.percentile over an empty/degenerate distribution fails.
# Below this floor we still persist the per-document dict (so the corpus keeps
# accumulating); we just defer the derived threshold/model until enough data
# exists. Tunable — raise it for a more stable distribution at the cost of a
# longer warm-up before ground-truth comparison kicks in.
MIN_ROWS_FOR_CALIBRATION = 10

# Cap on how many already-indexed quote Documents we read back from the store
# when re-deriving the signature phrases / recomputing rows. Mirrors the limit in
# build_profile._enumerate_quote_texts. The heavier O(n^2) calibration is capped
# separately at MAX_CALIBRATION_ROWS inside recompute_ground_truth_artifacts.
_QUOTE_CORPUS_READ_LIMIT = 10000

# LangGraph store locations for the signature key phrases. The vectorstore
# namespace mirrors the "quote" namespace so phrases can be retrieved the same
# way; the profile blob mirrors the "style_profile" blob (a single value read
# every turn) and additionally carries the raw phrase list the key_phrase_rate
# feature needs.
KEY_PHRASE_NAMESPACE_TAG = "key_phrase"
KEY_PHRASE_PROFILE_KEY = "key_phrase_profile"


def _quote_text_from_store_value(value: Any) -> str | None:
    """Pull page_content out of a stored quote item's value envelope."""
    value = value or {}
    content = value.get("page_content")
    if not content and isinstance(value.get("document"), dict):
        content = (value["document"] or {}).get("page_content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


async def _load_quote_corpus_by_doc_id(
    store: BaseStore, user_id: str, assistant_id: str, documents: List[Document]
) -> Dict[str, str]:
    """Return ``{document_id: text}`` for the FULL quote corpus of this avatar.

    The corpus is the union of the quotes already indexed in the store and the
    new ``documents`` from this upload. ``calibrate_ground_truth`` runs BEFORE the
    new documents are indexed into the ``quote`` vectorstore namespace, so the
    passed-in documents are merged in explicitly (and take precedence). Keying by
    ``document_id`` — the store key these docs live under — keeps the map aligned
    with the per-document feature dict and the delete-by-doc-id flow.
    """
    doc_id_to_text: Dict[str, str] = {}

    quote_namespace = (user_id, assistant_id, "quote")
    try:
        prior_items = await store.asearch(
            quote_namespace, query="*", limit=_QUOTE_CORPUS_READ_LIMIT
        )
    except Exception as exc:  # store backend may reject the wildcard query
        logger.warning("asearch over quotes failed (%s); using new documents only", exc)
        prior_items = []
    for item in prior_items or []:
        text = _quote_text_from_store_value(getattr(item, "value", {}))
        if text:
            doc_id_to_text[item.key] = text

    for document in documents:
        document_id = document.metadata.get("document_id")
        text = (document.page_content or "").strip()
        if document_id and text:
            doc_id_to_text[document_id] = text

    return doc_id_to_text


async def _store_signature_key_phrases(
    store: BaseStore,
    user_id: str,
    assistant_id: str,
    key_phrases_detailed: List[Dict[str, Any]],
) -> None:
    """Persist the discovered signature phrases two ways.

    1. One Document per phrase in the ``(user_id, assistant_id, "key_phrase")``
       vectorstore namespace, keyed by a deterministic uuid5 of the phrase so
       re-discovery on later uploads UPDATES rather than duplicates (the phrase
       vectorstore "expands" as more direct quotes arrive).
    2. A single ``key_phrase_profile`` blob at ``(assistant_id, KEY_PHRASE_...)``
       holding the raw phrase list (consumed by the key_phrase_rate feature) and a
       rendered, LLM-legible string (prompt-injected as the signature-phrase
       section), mirroring how the "style_profile" blob is stored/retrieved.
    """
    key_phrase_namespace = (user_id, assistant_id, KEY_PHRASE_NAMESPACE_TAG)
    for phrase in key_phrases_detailed:
        phrase_text = phrase["phrase"]
        phrase_document = Document(
            page_content=phrase_text,
            metadata={
                "namespace": KEY_PHRASE_NAMESPACE_TAG,
                "count": phrase.get("count"),
                "keyness_log2_over_generic_english": phrase.get(
                    "keyness_log2_over_generic_english"
                ),
            },
        )
        phrase_key = str(uuid.uuid5(uuid.NAMESPACE_URL, phrase_text))
        await store.aput(
            key_phrase_namespace,
            key=phrase_key,
            value={"document": phrase_document.to_json()},
        )

    phrase_list = [phrase["phrase"] for phrase in key_phrases_detailed]
    rendered = _render_key_phrase_profile_str(phrase_list)
    await store.aput(
        (assistant_id, KEY_PHRASE_PROFILE_KEY),
        key=KEY_PHRASE_PROFILE_KEY,
        value={"value": rendered, "phrases": phrase_list},
    )


def _render_key_phrase_profile_str(phrase_list: List[str]) -> str:
    """Render the phrase list as the LLM-legible signature-phrase block."""
    if not phrase_list:
        return ""
    return "\n".join(f'- "{phrase}"' for phrase in phrase_list)


async def _load_previous_key_phrases(
    store: BaseStore, assistant_id: str
) -> List[str]:
    """Return the phrase list from the last calibration (empty if none yet)."""
    item = await store.aget((assistant_id, KEY_PHRASE_PROFILE_KEY), key=KEY_PHRASE_PROFILE_KEY)
    value = getattr(item, "value", None) or {}
    phrases = value.get("phrases")
    return list(phrases) if isinstance(phrases, list) else []


async def calibrate_ground_truth(
    store: BaseStore,
    assistant_id: str,
    documents: List[Document],
    *,
    user_id: str,
) -> None:
    """Recalibrate the avatar's "direct quote" ground-truth cloud after an upload.

    Beyond the per-document stylometric features it always maintained, this now:

    * discovers the avatar's SIGNATURE KEY PHRASES over the full quote corpus and
      stores them (vectorstore namespace + prompt-injectable profile blob), and
    * keeps every per-document feature row's ``key_phrase_rate`` measured against
      the CURRENT phrase set.

    Because ``key_phrase_rate`` is measured against the discovered phrase set, that
    set changing invalidates previously-computed rows. So when the phrase set is
    unchanged we take the cheap incremental path (extract only the new documents
    and merge); when it changes we fully recompute every row from the quote corpus.
    Both feed ``recompute_ground_truth_artifacts`` (empirical threshold +
    IsolationForest) and rebuild the LLM-legible ``style_profile`` string.

    Args:
        store: LangGraph cross-thread store.
        assistant_id: The avatar whose cloud is being calibrated.
        documents: The new quote Documents from this upload.
        user_id: The avatar owner id (the first element of the owner-scoped
            ``quote`` / ``key_phrase`` namespaces).
    """
    import asyncio

    import numpy as np

    from src.anubis.utils.dataset.key_phrases import discover_key_phrases
    from src.anubis.utils.dataset.style_features import (
        FEATURE_NAMES,
        GROUND_TRUTH_FEATURES_DICT_KEY,
        build_style_profile_str,
        deserialize_features_by_doc_id,
        extract_style_features,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        serialize_features_by_doc_id,
    )

    def _feature_row(text: str, key_phrases: List[str]) -> Any:
        features = extract_style_features(text, key_phrases=key_phrases)
        return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

    # ── 1. Assemble the full quote corpus and (re)discover signature phrases. ──
    doc_id_to_text = await _load_quote_corpus_by_doc_id(
        store, user_id, assistant_id, documents
    )
    key_phrases_detailed = discover_key_phrases(list(doc_id_to_text.values()))
    key_phrases = [phrase["phrase"] for phrase in key_phrases_detailed]

    # Persist the phrases (vectorstore + profile blob) before feature work so the
    # signature-phrase section and the key_phrase_rate reference set are in sync.
    previous_key_phrases = await _load_previous_key_phrases(store, assistant_id)
    await _store_signature_key_phrases(
        store, user_id, assistant_id, key_phrases_detailed
    )

    # ── 2. Rebuild the per-document feature dict. ──────────────────────────────
    ground_truth_namespace = (assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
    existing_item = await store.aget(
        ground_truth_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY
    )
    existing_str = (getattr(existing_item, "value", None) or {}).get("value", None)
    existing_features_by_doc_id = deserialize_features_by_doc_id(existing_str)

    phrases_unchanged = set(key_phrases) == set(previous_key_phrases)
    if phrases_unchanged and existing_features_by_doc_id:
        # Fast path: the reference phrase set is stable, so existing rows are still
        # measured against the right phrases. Extract only the NEW documents and
        # merge (the historical incremental behaviour).
        new_items = [
            (document.metadata.get("document_id"), (document.page_content or "").strip())
            for document in documents
        ]
        new_items = [(doc_id, text) for doc_id, text in new_items if doc_id and text]
        new_rows = await asyncio.to_thread(
            lambda: [_feature_row(text, key_phrases) for _, text in new_items]
        )
        features_by_doc_id = dict(existing_features_by_doc_id)
        features_by_doc_id.update(
            {doc_id: row for (doc_id, _), row in zip(new_items, new_rows)}
        )
    else:
        # Slow path: the phrase set changed (or there is no prior corpus), so every
        # row's key_phrase_rate must be re-measured against the new phrases. Rebuild
        # the whole dict from the full quote corpus, keyed by document_id.
        corpus_items = list(doc_id_to_text.items())
        rows = await asyncio.to_thread(
            lambda: [_feature_row(text, key_phrases) for _, text in corpus_items]
        )
        features_by_doc_id = {
            doc_id: row for (doc_id, _), row in zip(corpus_items, rows)
        }

    # Persist the dict FIRST and unconditionally — the corpus must keep
    # accumulating even when it is still too small to calibrate against.
    await store.aput(
        ground_truth_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(features_by_doc_id)},
    )

    # ── 3. Defer threshold/model until the corpus is large enough. ─────────────
    ground_truth_text_features_arr = features_by_doc_id_to_arr(features_by_doc_id)
    if ground_truth_text_features_arr.shape[0] < MIN_ROWS_FOR_CALIBRATION:
        return

    # Recalibrate the empirical threshold + IsolationForest (O(n^2) leave-one-out
    # LedoitWolf work — offloaded off the event loop).
    (
        ground_truth_text_empirical_threshold_list_str,
        model_str_pkl,
    ) = await asyncio.to_thread(
        recompute_ground_truth_artifacts, ground_truth_text_features_arr
    )

    # Rebuild and store the LLM-legible style profile string.
    style_profile_str = await build_style_profile_str(ground_truth_text_features_arr)
    await store.aput(
        (assistant_id, "style_profile"),
        key="style_profile",
        value={"value": style_profile_str},
    )

    await store.aput(
        (assistant_id, "ground_truth_text_empirical_threshold_list_str"),
        key="ground_truth_text_empirical_threshold_list_str",
        value={"value": ground_truth_text_empirical_threshold_list_str},
    )

    await store.aput(
        (assistant_id, "ground_truth_text_features_model_b64_pkl"),
        key="ground_truth_text_features_model_b64_pkl",
        value={"value": model_str_pkl},
    )
