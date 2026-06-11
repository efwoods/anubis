-- Replace :assistant_id and :page_content. Finds the store row(s) whose Document
-- page_content matches the supplied text for that assistant.
--
-- :page_content is matched exactly. If you only have a fragment (or want to be
-- resilient to whitespace differences), swap the `=` predicate for the LIKE one
-- below — wrap the value in %...% when you call it.

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
    value->'document'->'kwargs'->>'metadata' AS metadata,
    prefix,
    key
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
  AND value->'document'->'kwargs'->>'page_content' = :page_content
  -- Fragment / whitespace-tolerant alternative (pass '%fragment%'):
  -- AND value->'document'->'kwargs'->>'page_content' LIKE :page_content
ORDER BY
    prefix,
    key;
