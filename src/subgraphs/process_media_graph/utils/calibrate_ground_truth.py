"""Recalibrate the per-avatar "direct quote" ground-truth cloud after an upload.

Maintains the per-document stylometric feature corpus, discovers/stores the
avatar's signature key phrases (vectorstore namespace + prompt-injectable
profile blob), and recalibrates the empirical threshold + IsolationForest.
"""

import logging
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

# Store key for the signature key-phrase profile blob: a JSON-encoded list of
# the avatar's signature phrases, stored ONCE per avatar at
# ``(user_id, assistant_id, KEY_PHRASE_PROFILE_KEY)`` under the same key
# (mirroring the "style_profile" blob). The list is consumed two ways: parsed
# for the key_phrase_rate feature, and rendered into the <SIGNATURE PHRASES>
# system prompt section. Each calibration unions the newly-discovered phrases
# with the previously-stored ones, then keeps only the phrases ATTESTED in the
# current cleaned quote corpus — a signature phrase must occur in the avatar's
# own quotes (this also purges artifacts stored before discovery cleaned its
# corpus, e.g. @mention-chain phrases).
KEY_PHRASE_PROFILE_KEY = "key_phrase_profile"


def _quote_text_from_store_value(value: Any) -> str | None:
    """Pull page_content out of a stored quote item's value envelope.

    The indexer persists each quote as a LangChain-serialized Document —
    ``{"document": {"kwargs": {"page_content": ..., "metadata": ...}}}`` (the
    same shape ``langgraph.json`` points the store's vector index at:
    ``document.kwargs.page_content``). The two flatter shapes are kept as
    fallbacks for values written by other paths. Missing the ``kwargs`` level
    here previously made every stored quote extract as None, so the corpus
    read-back silently produced an EMPTY prior corpus.
    """
    value = value or {}
    document_envelope = value.get("document")
    content = None
    if isinstance(document_envelope, dict):
        kwargs_envelope = document_envelope.get("kwargs")
        if isinstance(kwargs_envelope, dict):
            content = kwargs_envelope.get("page_content")
        if not content:
            content = document_envelope.get("page_content")
    if not content:
        content = value.get("page_content")
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
    phrase_list: List[str],
) -> None:
    """Persist the signature-phrase list as the single ``key_phrase_profile`` blob.

    Stored at ``(user_id, assistant_id, KEY_PHRASE_PROFILE_KEY)`` under the same
    key, as ``{"value": json.dumps(phrase_list)}`` — mirroring how the
    "style_profile" blob is stored/retrieved. The caller passes the already-
    unioned (previous ∪ newly-discovered) and corpus-attested list, so writing
    here upserts the reconciled set.
    """
    import json

    await store.aput(
        (user_id, assistant_id, KEY_PHRASE_PROFILE_KEY),
        key=KEY_PHRASE_PROFILE_KEY,
        value={"value": json.dumps(phrase_list)},
    )


async def _load_previous_key_phrases(
    store: BaseStore, user_id: str, assistant_id: str
) -> List[str]:
    """Return the phrase list from the last calibration (empty if none yet).

    The list is passed through ``phrase_is_well_formed`` on load: phrase sets
    stored BEFORE discovery cleaned its corpus are full of markup debris
    ("https t co ...", "amp ...") that would otherwise re-enter the union every
    calibration. Shape-based filtering here catches the obvious debris cheaply;
    the corpus-attestation filter in ``calibrate_ground_truth`` then removes
    anything that no longer occurs in the cleaned quote corpus (dropping
    phrases changes the set, which routes calibration down the full-recompute
    path so every row's key_phrase_rate is re-measured against the healed set).
    """
    import json

    from src.anubis.utils.dataset.key_phrases import phrase_is_well_formed

    item = await store.aget(
        (user_id, assistant_id, KEY_PHRASE_PROFILE_KEY), key=KEY_PHRASE_PROFILE_KEY
    )
    phrase_list_str = (getattr(item, "value", None) or {}).get("value", None)
    if not phrase_list_str:
        return []
    try:
        phrases = json.loads(phrase_list_str)
    except (TypeError, ValueError):
        return []
    if not isinstance(phrases, list):
        return []
    return [phrase for phrase in phrases if phrase_is_well_formed(phrase)]


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
      stores the (previous ∪ new) union as the ``key_phrase_profile`` blob, and
    * keeps every per-document feature row's ``key_phrase_rate`` measured against
      the CURRENT phrase set.

    Because ``key_phrase_rate`` is measured against the stored phrase set, that
    set changing invalidates previously-computed rows. So when the phrase set is
    unchanged we take the cheap incremental path (extract only the new documents
    and merge); when it grows we fully recompute every row from the quote corpus.
    Both feed ``recompute_ground_truth_artifacts`` (empirical threshold +
    IsolationForest) and rebuild the LLM-legible ``style_profile`` string.

    Every artifact this function writes lives under the owner-scoped
    ``(user_id, assistant_id, <artifact_name>)`` namespace with the artifact
    name as the key: ``key_phrase_profile``, the per-document feature dict
    (``GROUND_TRUTH_FEATURES_DICT_KEY``), ``style_profile``,
    ``ground_truth_text_empirical_threshold_list_str``, and
    ``ground_truth_text_features_model_b64_pkl``.

    Args:
        store: LangGraph cross-thread store.
        assistant_id: The avatar whose cloud is being calibrated.
        documents: The new quote Documents from this upload.
        user_id: The avatar owner id (the first element of every owner-scoped
            namespace above, matching the ``quote`` namespace).
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
        feature_row_is_all_nan,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        sanitize_ground_truth_feature_matrix,
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
    discovered_key_phrases = [phrase["phrase"] for phrase in key_phrases_detailed]

    # Union the newly-discovered phrases with the previously-stored ones, then
    # keep only phrases ATTESTED in the current cleaned corpus (a signature
    # phrase must occur in the avatar's own quotes; discovered phrases are
    # attested by construction, stale artifacts from pre-cleaning phrase sets
    # are not). Persisted BEFORE feature work so the signature-phrase section
    # and the key_phrase_rate reference set stay in sync. Sorted for a
    # deterministic stored order.
    from src.anubis.utils.dataset.key_phrases import (
        build_corpus_phrase_attestation_set,
    )

    previous_key_phrases = await _load_previous_key_phrases(
        store, user_id, assistant_id
    )
    attested_phrases = build_corpus_phrase_attestation_set(
        list(doc_id_to_text.values())
    )
    key_phrases = sorted(
        phrase
        for phrase in set(discovered_key_phrases) | set(previous_key_phrases)
        if phrase in attested_phrases
    )
    await _store_signature_key_phrases(store, user_id, assistant_id, key_phrases)

    # ── 2. Rebuild the per-document feature dict. ──────────────────────────────
    ground_truth_namespace = (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
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
        # All-NaN rows (URL-only / emoji-only lines) carry no stylometric
        # signal — keep them out of the persisted corpus entirely.
        features_by_doc_id.update(
            {
                doc_id: row
                for (doc_id, _), row in zip(new_items, new_rows)
                if not feature_row_is_all_nan(row)
            }
        )
    else:
        # Slow path: the phrase set changed (or there is no prior corpus), so
        # every row's key_phrase_rate must be re-measured against the new set.
        # Rebuild from the full quote corpus, keyed by document_id — but MERGE
        # over the existing dict rather than replace: an existing row whose
        # source text cannot be re-read this pass (indexing lag, the corpus
        # read limit, or a read-back fault) is RETAINED with its stale
        # key_phrase_rate column. One stale column in one row is a far smaller
        # error than silently discarding the row — a replace here once wiped a
        # ~6k-row corpus down to a single upload's rows when the store
        # read-back came back empty.
        corpus_items = list(doc_id_to_text.items())
        rows = await asyncio.to_thread(
            lambda: [_feature_row(text, key_phrases) for _, text in corpus_items]
        )
        features_by_doc_id = dict(existing_features_by_doc_id)
        features_by_doc_id.update(
            {
                doc_id: row
                for (doc_id, _), row in zip(corpus_items, rows)
                if not feature_row_is_all_nan(row)
            }
        )


    # Persist the dict FIRST and unconditionally — the corpus must keep
    # accumulating even when it is still too small to calibrate against.
    await store.aput(
        ground_truth_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(features_by_doc_id)},
    )

    # ── 3. Defer threshold/model until the corpus is large enough. ─────────────
    # Sanitize BEFORE the row-count floor so the count reflects rows that are
    # actually usable for calibration (a legacy corpus may still hold all-NaN
    # rows persisted before the write-time filter above existed).
    ground_truth_text_features_arr = sanitize_ground_truth_feature_matrix(
        features_by_doc_id_to_arr(features_by_doc_id)
    )
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
        (user_id, assistant_id, "style_profile"),
        key="style_profile",
        value={"value": style_profile_str},
    )

    await store.aput(
        (user_id, assistant_id, "ground_truth_text_empirical_threshold_list_str"),
        key="ground_truth_text_empirical_threshold_list_str",
        value={"value": ground_truth_text_empirical_threshold_list_str},
    )

    await store.aput(
        (user_id, assistant_id, "ground_truth_text_features_model_b64_pkl"),
        key="ground_truth_text_features_model_b64_pkl",
        value={"value": model_str_pkl},
    )

