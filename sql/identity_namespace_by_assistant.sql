-- All store rows for the assistant "identity" namespace (user_id, assistant_id, 'identity').
-- Document shapes differ: some rows use metadata.fact / metadata.fact_context; others
-- (e.g. image chunks) use metadata.namespace = 'identity', type, filename, chunk_index, etc.
--
-- Replace :assistant_id with the assistant UUID.

SELECT
    value->'document'->'kwargs'->'metadata'->>'fact' AS fact,
    value->'document'->'kwargs'->'metadata'->>'fact_context' AS fact_context,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS metadata_namespace,
    value->'document'->'kwargs'->'metadata'->>'type' AS metadata_type,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'chunk_index' AS chunk_index,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    value->'document'->'kwargs'->'metadata' AS full_metadata,
    prefix,
    key,
    value->'document'->'kwargs'->'metadata'->>'user_id' AS owner_user_id
FROM store
WHERE split_part(prefix, '.', -1) = 'identity'
  AND value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
ORDER BY updated_at DESC NULLS LAST, prefix, key;
