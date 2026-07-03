-- :user_id and :assistant_id are the avatar owner + avatar IDs
SELECT
    split_part(prefix, '.', 3)        AS dataset_type,
    split_part(prefix, '.', 4)        AS source_uuid5,
    key,                                          -- == source_uuid5
    value ->> 'source_filename'       AS source_filename,
    (value ->> 'row_count')::int      AS row_count,
    value ->> 'created_at'            AS created_at,
    value ->> 'value'                  AS dataset_json  -- the full dataset as ONE JSON-array document string
FROM store
WHERE prefix LIKE :user_id || '.' || :assistant_id || '.%'
  AND split_part(prefix, '.', 3) IN (
        'q_and_a_adapter',
        'langsmith_factual_q_and_a',
        'multi_turn_dataset_adapter'
  )
ORDER BY dataset_type, created_at;
