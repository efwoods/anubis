from langgraph.store.base import BaseStore
from langchain_core.documents import Document

async def calibrate_ground_truth(store: BaseStore, assistant_id: str, documents: Document):
    # Extract stylometric features per quote line and merge them into the
    # avatar's persisted "direct quote" corpus, then recalibrate the derived
    # empirical threshold + IsolationForest from the merged corpus.
    import pandas as pd
    from src.anubis.utils.dataset.style_features import (
        GROUND_TRUTH_FEATURES_DICT_KEY,
        deserialize_features_by_doc_id,
        extract_style_features,
        features_by_doc_id_to_arr,
        recompute_ground_truth_artifacts,
        serialize_features_by_doc_id,
    )
    breakpoint()
    features = [extract_style_features(doc.page_content) for doc in documents]
    features_arr = pd.DataFrame(features).values # shape (n_obs/documents, n_features=33); ndarray so downstream .tolist()/np.concatenate work
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
    ground_truth_text_features_arr = features_by_doc_id_to_arr(
        ground_truth_text_features_by_doc_id
    )
    # Recalibrate the empirical threshold + IsolationForest from the corpus.
    (
        ground_truth_text_empirical_threshold_list_str,
        model_str_pkl,
    ) = recompute_ground_truth_artifacts(ground_truth_text_features_arr)
    # Persist the merged dict, recalibrated threshold, and recalibrated model.
    await store.aput(
        ground_truth_text_features_by_doc_id_namespace,
        key=GROUND_TRUTH_FEATURES_DICT_KEY,
        value={"value": serialize_features_by_doc_id(ground_truth_text_features_by_doc_id)},
    )
    ground_truth_text_empirical_threshold_namespace = (assistant_id, "ground_truth_text_empirical_threshold_list_str")
    await store.aput(
        ground_truth_text_empirical_threshold_namespace,
        key="ground_truth_text_empirical_threshold_list_str", value={"value":ground_truth_text_empirical_threshold_list_str}
    )
    ground_truth_text_features_model_namespace = (assistant_id, "ground_truth_text_features_model_b64_pkl")
    await store.aput(
        ground_truth_text_features_model_namespace,
        key="ground_truth_text_features_model_b64_pkl", value={"value":model_str_pkl}
    )
    