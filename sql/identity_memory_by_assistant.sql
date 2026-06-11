-- LangGraph Postgres store: namespace (user_id, assistant_id, 'identity_memory')
-- is stored as prefix "user_id.assistant_id.identity_memory" (see langgraph
-- store postgres _namespace_to_text). Rows may omit metadata.namespace; use
-- prefix suffix + assistant_id in metadata for a stable filter.
--
-- Replace :assistant_id with the assistant UUID.

SELECT
    value->'document'->'kwargs'->'metadata'->>'fact' AS fact,
    value->'document'->'kwargs'->'metadata'->>'fact_context' AS fact_context,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    prefix,
    key,
    value->'document'->'kwargs'->'metadata'->>'user_id' AS owner_user_id,
    value->'document'->'kwargs'->>'metadata' AS metadata
FROM store
WHERE split_part(prefix, '.', -1) = 'identity_memory'
  AND value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
ORDER BY updated_at DESC NULLS LAST, prefix, key;
