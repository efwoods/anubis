-- List every assistant_id that has a style_profile for a given user.
--
-- Written by calibrate_ground_truth() / /calibrate_ground_truth via store.aput at:
--     namespace = (user_id, assistant_id, "style_profile")
--     key      = "style_profile"
--     value    = {"value": "<style-profile summary string>"}
--
-- Stored prefix shape (3-tuple):
--     '<user_id>.<assistant_id>.style_profile'
--
-- Replace :user_id (e.g. '69e5e49980b783d7dff3012b').

SELECT
    split_part(prefix, '.', 2)        AS assistant_id,
    prefix,
    key,
    length(value->>'value')           AS value_len,
    left(value->>'value', 120)        AS value_preview,
    updated_at
FROM store
WHERE key = 'style_profile'
  AND prefix LIKE :user_id || '.%.style_profile'
  AND split_part(prefix, '.', 1) = :user_id
  AND split_part(prefix, '.', 3) = 'style_profile'
  AND coalesce(value->>'value', '') <> ''
ORDER BY updated_at DESC NULLS LAST, assistant_id;
