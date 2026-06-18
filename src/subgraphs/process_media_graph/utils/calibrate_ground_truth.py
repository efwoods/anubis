from typing import List

from langchain_core.documents import Document
from langgraph.store.base import BaseStore

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


async def calibrate_ground_truth(
    store: BaseStore, assistant_id: str, documents: List[Document]
) -> None:
    """Merge per-document stylometric features into the avatar's "direct quote"
    corpus and (when large enough) recalibrate the derived empirical threshold +
    IsolationForest.

    The corpus is persisted as a ``{document_id: [33 floats]}`` dict so individual
    documents' rows can be pruned on deletion. The dict is ALWAYS stored (created
    on first call, merged thereafter); the threshold/model are only recomputed
    once the merged corpus reaches ``MIN_ROWS_FOR_CALIBRATION`` rows.
    """
    import numpy as np
    import pandas as pd

    from src.anubis.utils.dataset.style_features import (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        deserialize_features_by_doc_id,
        extract_style_features,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        serialize_features_by_doc_id,
    )

    features = [extract_style_features(doc.page_content) for doc in documents]
    features_arr = np.array(pd.DataFrame(features).values)  # shape (n_obs/documents, n_features=33); ndarray so downstream .tolist()/np.concatenate work

    # Map each document id to its feature row. features_arr rows are positionally
    # aligned with `documents`, so zip pairs each doc with its 1-D feature vector.
    # The key is `document_id` (the store key these docs are indexed under), so
    # the delete flow can prune the exact rows when a source document is removed.
    features_by_doc_id = {
        document.metadata.get("document_id"): row
        for document, row in zip(documents, features_arr)
    }

    # Namespace/key must match the reader in graph._attach_analyzed_features
    # and the deleter in webapp.delete_avatar_documents.
    ground_truth_text_features_by_doc_id_namespace = (assistant_id, GROUND_TRUTH_FEATURES_DICT_KEY)
    ground_truth_text_features_by_doc_id_ITEM = await store.aget(
        ground_truth_text_features_by_doc_id_namespace, key=GROUND_TRUTH_FEATURES_DICT_KEY
    )
    ground_truth_text_features_by_doc_id_str = (
        getattr(ground_truth_text_features_by_doc_id_ITEM, "value", None) or {}
    ).get("value", None)

    # Merge the new per-document rows into the existing dict (creating it if
    # absent), then reconstruct the full (n_docs, 33) corpus array from it.
    ground_truth_text_features_by_doc_id = deserialize_features_by_doc_id(
        ground_truth_text_features_by_doc_id_str
    )
    ground_truth_text_features_by_doc_id.update(features_by_doc_id)

    # Persist the merged dict FIRST and unconditionally — the corpus must keep
    # accumulating even when it's still too small to calibrate against.
    await store.aput(
        ground_truth_text_features_by_doc_id_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(ground_truth_text_features_by_doc_id)},
    )

    # Defer threshold/model calibration until the corpus is large enough for the
    # leave-one-out distribution to be well-posed.
    ground_truth_text_features_arr = features_by_doc_id_to_arr(
        ground_truth_text_features_by_doc_id
    )
    if ground_truth_text_features_arr.shape[0] < MIN_ROWS_FOR_CALIBRATION:
        return

    # Recalibrate the empirical threshold + IsolationForest from the corpus.
    (
        ground_truth_text_empirical_threshold_list_str,
        model_str_pkl,
    ) = recompute_ground_truth_artifacts(ground_truth_text_features_arr)

    # TODO: BUILD STYLE PROFILE
    # style_profile_str = await build_style_profile_str(ground_truth_text_features_arr)
    style_profile_namespace = (assistant_id, "style_profile")
    await store.aput(style_profile_namespace, key="style_profile", value={"value":style_profile_str})


    ground_truth_text_empirical_threshold_namespace = (assistant_id, "ground_truth_text_empirical_threshold_list_str")
    await store.aput(
        ground_truth_text_empirical_threshold_namespace,
        key="ground_truth_text_empirical_threshold_list_str", value={"value": ground_truth_text_empirical_threshold_list_str}
    )

    ground_truth_text_features_model_namespace = (assistant_id, "ground_truth_text_features_model_b64_pkl")
    await store.aput(
        ground_truth_text_features_model_namespace,
        key="ground_truth_text_features_model_b64_pkl", value={"value": model_str_pkl}
    )
