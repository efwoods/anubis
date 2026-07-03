-- Inspect the stylometric calibration artifacts for one avatar.
-- Replace :user_id and :assistant_id (both are provided so the row prefix can be
-- labelled, but the match keys on :assistant_id alone — see below).
--
-- These rows are written by calibrate_ground_truth() / _store_signature_key_phrases()
-- via store.aput. They are NOT LangChain Documents (there is no
-- value->'document'->...->'page_content'), so page_content_by_assistant_and_namespace
-- cannot see them — match on prefix/key instead. Each value is {"value": "<string>"}.
--
--   * key_phrase_profile                              -> JSON array of signature phrases
--   * style_profile                                   -> style-profile summary string
--   * ground_truth_text_features_by_doc_id_dict_str   -> {document_id: [33 floats]}
--       (n_documents = number of corpus rows currently stored)
--   * ground_truth_text_empirical_threshold_list_str  -> Tukey-fence threshold
--   * ground_truth_text_features_model_b64_pkl        -> base64 IsolationForest
--
-- WHY THE OLD `prefix LIKE :assistant_id || '.%'` RETURNED NOTHING
-- ---------------------------------------------------------------
-- The write side is inconsistent about the namespace tuple, so the stored prefix
-- has one of TWO shapes for the same key across avatars:
--     3-tuple  (user_id, assistant_id, key)  -> prefix = '<user_id>.<assistant_id>.<key>'
--     2-tuple  (assistant_id, key)           -> prefix = '<assistant_id>.<key>'
-- (In calibrate_ground_truth.py: key_phrase_profile / *_dict_str / style_profile
-- use the 3-tuple; *_threshold_list_str / *_model_b64_pkl use the 2-tuple.)
-- A `prefix LIKE :assistant_id || '.%'` filter misses every 3-tuple row because
-- that prefix starts with the user_id, not the assistant_id. The filter below
-- instead requires :assistant_id to appear as a whole dot-delimited component
-- immediately before the key, which matches BOTH shapes and needs no user_id.
--
-- The dict / model rows can be large (tens of MB); n_documents parses the dict
-- once, and value_preview truncates so nothing dumps the full blob.

SELECT
    key,
    prefix,
    CASE
        WHEN prefix LIKE :user_id || '.%' THEN 'user_id.assistant_id.key (3-tuple)'
        ELSE 'assistant_id.key (2-tuple)'
    END                                                   AS prefix_shape,
    length(value->>'value')                               AS value_len,
    CASE
        WHEN key = 'ground_truth_text_features_by_doc_id_dict_str'
        THEN (SELECT count(*) FROM jsonb_object_keys((value->>'value')::jsonb))
    END                                                   AS n_documents,
    CASE
        WHEN key = 'key_phrase_profile'
        THEN jsonb_array_length((value->>'value')::jsonb)
    END                                                   AS n_key_phrases,
    CASE
        WHEN key = 'ground_truth_text_empirical_threshold_list_str'
        THEN value->>'value'
    END                                                   AS threshold_value,
    left(value->>'value', 120)                            AS value_preview
FROM store
WHERE key IN (
        'key_phrase_profile',
        'style_profile',
        'ground_truth_text_features_by_doc_id_dict_str',
        'ground_truth_text_empirical_threshold_list_str',
        'ground_truth_text_features_model_b64_pkl'
      )
  AND ('.' || prefix) LIKE '%.' || :assistant_id || '.' || key
ORDER BY key;
