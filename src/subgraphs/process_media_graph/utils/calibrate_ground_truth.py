"""Recalibrate the per-avatar "direct quote" ground-truth cloud after an upload.

Maintains the per-document stylometric feature corpus, discovers/stores the
avatar's signature key phrases (vectorstore namespace + prompt-injectable
profile blob), and recalibrates the empirical threshold + IsolationForest.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)
from src.anubis.utils.store_cache import invalidate_store_cache_entry

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
# Sibling key in the SAME namespace holding the per-phrase scores, the judge's
# classifications and the judgement cache. Kept separate rather than folded into
# the primary key so the primary key keeps its bare-list shape for the two hot
# read paths (system-prompt build, per-reply style scoring) with no migration.
KEY_PHRASE_PROFILE_DETAIL_KEY = "key_phrase_profile_detail"


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



def _target_name_from_quote_metadata(
    store_items: Any, documents: List[Document]
) -> Optional[str]:
    """The most frequently attested target name across the quote corpus.

    Only the signature-phrase judge uses this, and only to steer borderline
    calls: knowing the speaker is a rocket engineer makes "orbit" read as
    subject matter rather than as a verbal habit. The assistant id is
    deliberately NOT used as a substitute — a bare identifier tells the judge
    nothing and reads as noise in the prompt — so an avatar whose quotes carry
    no target name is simply judged without one.
    """
    name_counts: Counter = Counter()
    for document in documents or []:
        name = (document.metadata or {}).get("target_name")
        if isinstance(name, str) and name.strip():
            name_counts[name.strip()] += 1
    for item in store_items or []:
        value = getattr(item, "value", None)
        if not isinstance(value, dict):
            continue
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            kwargs = value.get("kwargs")
            metadata = kwargs.get("metadata") if isinstance(kwargs, dict) else None
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("target_name")
        if isinstance(name, str) and name.strip():
            name_counts[name.strip()] += 1
    if not name_counts:
        return None
    return name_counts.most_common(1)[0][0]


async def _load_quote_corpus_by_doc_id(
    store: BaseStore,
    user_id: str,
    assistant_id: str,
    documents: List[Document],
    *,
    collected_store_items: Optional[List[Any]] = None,
) -> Dict[str, str]:
    """Return ``{document_id: text}`` for the FULL quote corpus of this avatar.

    The corpus is the union of the quotes already indexed in the store and the
    new ``documents`` from this upload. ``calibrate_ground_truth`` runs BEFORE the
    new documents are indexed into the ``quote`` vectorstore namespace, so the
    passed-in documents are merged in explicitly (and take precedence). Keying by
    ``document_id`` — the store key these docs live under — keeps the map aligned
    with the per-document feature dict and the delete-by-doc-id flow.

    ``collected_store_items``, when given, is filled with the raw store items so
    the caller can read their metadata (the target's name, for the judge)
    without paying for a second read of a corpus that may hold ten thousand rows.
    """
    doc_id_to_text: Dict[str, str] = {}

    quote_namespace = (user_id, assistant_id, "quote")
    try:
        prior_items = await store.asearch(
            quote_namespace, query="*", limit=_QUOTE_CORPUS_READ_LIMIT
        )
    except Exception as exc:  # store backend may reject the wildcard query
        # When this upload carries new documents there is still something real to
        # calibrate over, so a failed read-back degrades to "use the new documents
        # only". When it does not, the store read-back is the ONLY source of the
        # corpus, and an empty result is then indistinguishable from a genuinely
        # quote-less avatar — the caller would go on to persist a degenerate,
        # empty feature dict over a corpus that may hold thousands of rows.
        # Re-raise instead: every caller wraps this in a best-effort handler, so
        # the upload still succeeds and simply skips recalibration.
        if not documents:
            logger.error(
                "asearch over quotes failed (%s) and no new documents were passed; "
                "refusing to calibrate against an unreadable corpus",
                exc,
            )
            raise
        logger.warning("asearch over quotes failed (%s); using new documents only", exc)
        prior_items = []
    if collected_store_items is not None:
        collected_store_items.extend(prior_items or [])

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
    "style_profile" blob is stored/retrieved. The caller passes the ranked,
    capped and corpus-attested list, so writing here upserts the reconciled set.

    The list is stored in RANK order (most distinctive first), not sorted
    alphabetically, because the prompt renders a prefix of it. Nothing
    downstream depends on alphabetical order: ``key_phrase_occurrence_rate``
    sums over the whole set and is order-independent.
    """
    import json

    await store.aput(
        (user_id, assistant_id, KEY_PHRASE_PROFILE_KEY),
        key=KEY_PHRASE_PROFILE_KEY,
        value={"value": json.dumps(phrase_list)},
    )


async def _store_key_phrase_profile_detail(
    store: BaseStore,
    user_id: str,
    assistant_id: str,
    envelope: dict,
) -> None:
    """Persist the scores, judgements and judgement cache beside the phrase list.

    The judgement cache is the reason this record exists at all: the judge runs
    at a low but non-zero temperature, so re-judging the same phrase every
    calibration would make the stored set drift, and every drift forces a full
    recomputation of every document's stylometric row. Caching the judgement
    keeps a stable corpus producing a stable set.
    """
    import json

    await store.aput(
        (user_id, assistant_id, KEY_PHRASE_PROFILE_KEY),
        key=KEY_PHRASE_PROFILE_DETAIL_KEY,
        value={"value": json.dumps(envelope)},
    )


async def _load_key_phrase_profile_detail(
    store: BaseStore, user_id: str, assistant_id: str
) -> dict:
    """Read the detail record; an empty mapping when absent or unreadable.

    An avatar calibrated before this record existed simply returns ``{}``, which
    means "no cached judgements" — the next calibration pays for a full judging
    pass once and caches the result.
    """
    from src.anubis.utils.dataset.key_phrases import load_key_phrase_profile_detail

    try:
        item = await store.aget(
            (user_id, assistant_id, KEY_PHRASE_PROFILE_KEY),
            key=KEY_PHRASE_PROFILE_DETAIL_KEY,
        )
    except Exception:  # noqa: BLE001 - a missing sibling key must never fail calibration
        return {}
    detail_str = (getattr(item, "value", None) or {}).get("value", None)
    return load_key_phrase_profile_detail(detail_str)


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

    * discovers the avatar's SIGNATURE KEY PHRASES over the full quote corpus
      and stores the ranked, capped result as the ``key_phrase_profile`` blob
      (previously-stored phrases are re-measured and compete for the cap rather
      than being unioned in, which is what bounds the set across uploads), and
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
    quote_store_items: List[Any] = []
    doc_id_to_text = await _load_quote_corpus_by_doc_id(
        store,
        user_id,
        assistant_id,
        documents,
        collected_store_items=quote_store_items,
    )
    # Two-stage discovery over the avatar's FULL quote history. Stage one is a
    # pure-Python scan that the orchestrator keeps off the event loop; stage two
    # is one batched structured-output judging pass that separates the phrases
    # marking HOW this person talks from the ones that merely name WHAT they
    # talk about — a distinction no statistic can make, because a person who
    # discusses one subject constantly produces the same statistical signature
    # as a person with a verbal habit.
    #
    # The previously-stored phrases are NOT unioned in. They are handed to the
    # orchestrator as incumbents, re-measured against the current corpus, and
    # keep their place only if the evidence still supports them. The old union
    # is what made the stored list grow by roughly forty phrases per upload
    # until the prompt was rendering hundreds of them.
    from src.anubis.utils.dataset.key_phrase_judgement import (
        build_signature_key_phrase_profile,
    )
    from src.anubis.utils.dataset.key_phrases import (
        build_corpus_phrase_attestation_set,
    )

    previous_key_phrases = await _load_previous_key_phrases(
        store, user_id, assistant_id
    )
    previous_profile_detail = await _load_key_phrase_profile_detail(
        store, user_id, assistant_id
    )
    # The detail record is the authority on the incumbent set once it exists;
    # before it exists the legacy bare list is, so an avatar calibrated under
    # the old scheme still gets its phrases re-measured rather than discarded.
    if not previous_profile_detail.get("phrases") and previous_key_phrases:
        previous_profile_detail = {
            **previous_profile_detail,
            "phrases": list(previous_key_phrases),
        }

    target_speaker_name = _target_name_from_quote_metadata(
        quote_store_items, documents
    )

    key_phrase_profile = None
    try:
        key_phrase_profile = await build_signature_key_phrase_profile(
            list(doc_id_to_text.values()),
            speaker_name=target_speaker_name,
            previous_profile_detail=previous_profile_detail,
        )
        discovered_key_phrases = list(key_phrase_profile.phrases)
    except Exception as exc:  # noqa: BLE001
        # Never lose a working phrase set because discovery or judging failed.
        # An empty set here would route calibration down the full-recompute
        # path AND strip the avatar's SIGNATURE PHRASES section at the same time.
        logger.warning(
            "Signature key-phrase discovery failed for %s (%s); keeping the "
            "previously stored phrase set of %d",
            assistant_id,
            exc,
            len(previous_key_phrases),
        )
        discovered_key_phrases = list(previous_key_phrases)

    # Keep only phrases ATTESTED in the current cleaned corpus: a signature
    # phrase must occur in the avatar's own quotes. Discovered phrases are
    # attested by construction; a retained incumbent from a corpus that is no
    # longer readable is not. Order is preserved because the list is stored
    # most-distinctive-first and the prompt renders a prefix of it. Persisted
    # BEFORE feature work so the prompt section and the key_phrase_rate
    # reference set stay in sync.
    attested_phrases = build_corpus_phrase_attestation_set(
        list(doc_id_to_text.values())
    )
    key_phrases = [
        phrase for phrase in discovered_key_phrases if phrase in attested_phrases
    ]
    await _store_signature_key_phrases(store, user_id, assistant_id, key_phrases)
    if key_phrase_profile is not None:
        from src.anubis.utils.dataset.key_phrases import (
            build_key_phrase_profile_detail_envelope,
        )

        await _store_key_phrase_profile_detail(
            store,
            user_id,
            assistant_id,
            build_key_phrase_profile_detail_envelope(
                key_phrases,
                [
                    entry
                    for entry in key_phrase_profile.entries
                    if entry.get("phrase") in set(key_phrases)
                ],
                key_phrase_profile.judgement_cache,
                classification_histogram=(
                    key_phrase_profile.classification_histogram
                ),
            ),
        )

    # ── 2. Rebuild the per-document feature dict. ──────────────────────────────
    ground_truth_namespace = (user_id, assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
    existing_item = await store.aget(
        ground_truth_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY
    )
    existing_str = (getattr(existing_item, "value", None) or {}).get("value", None)
    existing_features_by_doc_id = deserialize_features_by_doc_id(existing_str)

    phrases_unchanged = set(key_phrases) == set(previous_key_phrases)
    if phrases_unchanged and existing_features_by_doc_id:
        # Fast path: the reference phrase set is stable, so rows already in the
        # dict were measured against the right phrases and need no recomputation.
        #
        # What must be (re)extracted is therefore the CORPUS DELTA — every quote
        # in the corpus that has no row yet — not merely the documents handed to
        # this call. The two are not the same set: this function is also driven
        # with ``documents=[]`` (the post-upload hook and the backfill script read
        # the corpus straight from the store), and an argument-driven delta would
        # then be empty, silently leaving every newly indexed quote unmeasured
        # forever. The passed documents are unioned in on top because a re-upload
        # can REPLACE the text stored under an existing document_id, so their rows
        # are stale even though the ids are already present.
        passed_document_ids = {
            document.metadata.get("document_id")
            for document in documents
            if document.metadata.get("document_id")
        }
        document_ids_needing_extraction = (
            set(doc_id_to_text) - set(existing_features_by_doc_id)
        ) | (passed_document_ids & set(doc_id_to_text))
        new_items = [
            (doc_id, doc_id_to_text[doc_id])
            for doc_id in document_ids_needing_extraction
            if doc_id_to_text[doc_id]
        ]
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

    # BUILD AND STORE STYLE PROFILE
    from src.anubis.utils.dataset.style_features import build_style_profile_str

    style_profile_str = await build_style_profile_str(ground_truth_text_features_arr)
    style_profile_namespace = (user_id, assistant_id, "style_profile")
    await store.aput(
        style_profile_namespace, key="style_profile", value={"value": style_profile_str}
    )
    # load_consciousness reads the style profile through a process-wide cache;
    # drop the cached copy so the recalibrated profile is picked up on the
    # next message.
    invalidate_store_cache_entry(style_profile_namespace, "style_profile")

    ground_truth_text_empirical_threshold_namespace = (user_id,
        assistant_id,
        "ground_truth_text_empirical_threshold_list_str",
    )
    await store.aput(
        ground_truth_text_empirical_threshold_namespace,
        key="ground_truth_text_empirical_threshold_list_str",
        value={"value": ground_truth_text_empirical_threshold_list_str},
    )

    ground_truth_text_features_model_namespace = (user_id,
        assistant_id,
        "ground_truth_text_features_model_b64_pkl",
    )
    await store.aput(
        ground_truth_text_features_model_namespace,
        key="ground_truth_text_features_model_b64_pkl",
        value={"value": model_str_pkl},
    )


async def calibrate_ground_truth_from_stored_corpus(
    store: BaseStore, assistant_id: str, *, user_id: str
) -> None:
    """Recalibrate an avatar's direct-quote cloud from the quotes already stored.

    ``calibrate_ground_truth`` reads the avatar's full quote corpus out of the
    store itself (``_load_quote_corpus_by_doc_id``), treating its ``documents``
    argument only as an upload's not-yet-indexed additions. So every caller that
    runs AFTER indexing — the post-upload batch hook, the backfill script, the
    recalibration endpoint — has nothing to thread in and passes no documents.

    This wrapper exists so those three callers share one named entry point rather
    than each open-coding ``documents=[]``: the empty list is the load-bearing
    part of the contract, and three independent copies of that decision would be
    free to drift apart.

    Args:
        store: LangGraph cross-thread store holding the ``quote`` namespace.
        assistant_id: The avatar whose direct-quote cloud is being recalibrated.
        user_id: The avatar OWNER's id — the first element of every owner-scoped
            namespace, matching what ``calibrate_ground_truth`` writes under and
            what the message path reads back.
    """
    await calibrate_ground_truth(
        store=store, assistant_id=assistant_id, documents=[], user_id=user_id
    )
