-- Replace :assistant_id and :namespace (e.g. 'analysis', 'identity', 'memory', 'quote').
-- One row per store row (usually one LangChain Document chunk).

SELECT
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS namespace,
    value->'document'->'kwargs'->'metadata'->>'feature' AS feature,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    value->'document'->'kwargs'->'metadata'->>'chunk_index' AS chunk_index,
    value->'document'->'kwargs'->'metadata'->>'total_chunks' AS total_chunks,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    prefix,
    value->'document'->'kwargs'->>'metadata' AS metadata,
    key
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
  AND value->'document'->'kwargs'->'metadata'->>'namespace' = :namespace
ORDER BY
    value->'document'->'kwargs'->'metadata'->>'feature' NULLS LAST,
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ),
    (NULLIF(value->'document'->'kwargs'->'metadata'->>'chunk_index', ''))::int NULLS LAST,
    prefix,
    key;
