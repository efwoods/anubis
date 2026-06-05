-- Replace :assistant_id. One row per store row (usually one LangChain Document chunk).
-- Rows for the same upload share file_key; order puts chunks in chunk_index order when present.

SELECT
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key,
    value->'document'->'kwargs'->'metadata'->>'chunk_index' AS chunk_index,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS namespace,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    value->'document'->'kwargs'->'metadata'->>'total_chunks' AS total_chunks,
    value->'document'->'kwargs'->>'metadata' AS metadata,
    prefix,
    key
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
ORDER BY
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ),
    (NULLIF(value->'document'->'kwargs'->'metadata'->>'chunk_index', ''))::int NULLS LAST,
    prefix,
    key;
