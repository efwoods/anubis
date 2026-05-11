-- Replace :assistant_id and :filename. Filename matches metadata filename exactly.
-- One row per store row (usually one LangChain Document chunk).

SELECT
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS namespace,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    value->'document'->'kwargs'->'metadata'->>'chunk_index' AS chunk_index,
    value->'document'->'kwargs'->'metadata'->>'total_chunks' AS total_chunks,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    prefix,
    key
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
  AND value->'document'->'kwargs'->'metadata'->>'filename' = :filename
ORDER BY
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ),
    (NULLIF(value->'document'->'kwargs'->'metadata'->>'chunk_index', ''))::int NULLS LAST,
    prefix,
    key;
