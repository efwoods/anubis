-- Replace :assistant_id with the target UUID (or use prepared statement placeholder).
-- Counts one row per logical file (chunks share filename_uuid5; falls back to document_id).

SELECT COUNT(*) AS unique_file_count
FROM (
    SELECT DISTINCT COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key
    FROM store
    WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
      AND COALESCE(
            value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
            value->'document'->'kwargs'->'metadata'->>'document_id'
        ) IS NOT NULL
) AS distinct_files;
