-- Inspect the stylometric "direct quote" ground-truth artifacts for one avatar.
-- Replace :assistant_id.
--
-- These rows are written by calibrate_ground_truth() via store.aput under the
-- namespace tuple (assistant_id, <key>), which the LangGraph Postgres store
-- persists as prefix = '<assistant_id>.<key>'. They are NOT LangChain Documents
-- (no value->'document'->...->'page_content'), so the page_content_by_assistant_
-- and_namespace query cannot see them — match on prefix/key instead.
--
--   * ground_truth_text_features_by_doc_id_dict_str  -> {document_id: [33 floats]}
--       (n_documents = number of corpus rows currently stored)
--   * ground_truth_text_empirical_threshold_list_str -> Tukey-fence threshold
--   * ground_truth_text_features_model_b64_pkl       -> base64 IsolationForest
--
-- The threshold/model rows only appear once the corpus reaches
-- MIN_ROWS_FOR_CALIBRATION; the dict row is written on every calibrate call.

SELECT
    prefix,
    key,
    length(value->>'value') AS value_len,
    CASE
        WHEN key = 'ground_truth_text_features_by_doc_id_dict_str'
        THEN (SELECT count(*) FROM jsonb_object_keys((value->>'value')::jsonb))
    END AS n_documents
FROM store
WHERE prefix LIKE :assistant_id || '.%'
  AND key IN (
      'ground_truth_text_features_by_doc_id_dict_str',
      'ground_truth_text_empirical_threshold_list_str',
      'ground_truth_text_features_model_b64_pkl'
  )
ORDER BY key;
