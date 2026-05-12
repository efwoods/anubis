-- Replace :assistant_id with the target UUID (or use prepared statement placeholder).
-- One row per logical file (deduped on filename_uuid5, else document_id).

SELECT DISTINCT ON (
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    )
)
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key,
    prefix,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS namespace,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    value->'document'->'kwargs'->'metadata'->>'filename_uuid5' AS filename_uuid5
    
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
  AND COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) IS NOT NULL
ORDER BY
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ),
    filename;
