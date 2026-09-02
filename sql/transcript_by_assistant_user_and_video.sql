-- Replace :assistant_id, :user_id and :video. Prints the text that was
-- transcribed from one video (or audio) upload for one avatar.
--
-- :user_id is the account the API key belongs to. An API key itself can never be
-- a filter here: API keys are not stored in Postgres at all -- security.auth
-- hashes the key with SHA-256 and Auth0 holds only that digest under
-- app_metadata.api_key. The account behind a key is the bare Auth0 identifier
-- that lookup returns, which IS in the database: in the document metadata below,
-- in assistant.metadata->>'user_id', and as the first element of store.prefix.
--
-- :video names the source. Exact identifiers match first (filename_uuid5 /
-- namespace_filename -- the uuid5 over the filename or URL a video is keyed
-- under, or a playlist's own uuid5, which returns every video of that playlist),
-- otherwise :video is matched as a substring of the filename, the source URL, or
-- the audio_filename -- so a watch URL, a file name, or a fragment of either
-- works without computing the hash by hand.
--
-- Only transcript-bearing rows are returned: a video's audio track is extracted
-- and transcribed, and every Document produced that way carries audio_filename
-- (see process_media_graph.utils.nodes). speaker / segment_start / segment_end
-- are populated when the audio was diarized, and are NULL for a plain
-- whisper transcription. Delete the audio_filename/content_type/type predicate at
-- the bottom to see every document the source produced, transcript or not.
--
-- One row per store row (usually one LangChain Document chunk, or one diarized
-- segment); page_content holds the transcribed text.

SELECT
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'namespace_filename',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ) AS file_key,
    value->'document'->'kwargs'->'metadata'->>'filename' AS filename,
    value->'document'->'kwargs'->'metadata'->>'audio_filename' AS audio_filename,
    value->'document'->'kwargs'->'metadata'->>'namespace' AS namespace,
    value->'document'->'kwargs'->'metadata'->>'content_type' AS content_type,
    value->'document'->'kwargs'->'metadata'->>'speaker' AS speaker,
    value->'document'->'kwargs'->'metadata'->>'start' AS segment_start,
    value->'document'->'kwargs'->'metadata'->>'end' AS segment_end,
    value->'document'->'kwargs'->'metadata'->>'chunk_index' AS chunk_index,
    value->'document'->'kwargs'->'metadata'->>'total_chunks' AS total_chunks,
    value->'document'->'kwargs'->>'page_content' AS page_content,
    prefix,
    value->'document'->'kwargs'->'metadata'->>'document_id' AS document_id,
    key
FROM store
WHERE value->'document'->'kwargs'->'metadata'->>'assistant_id' = :assistant_id
  AND value->'document'->'kwargs'->'metadata'->>'user_id' = :user_id
  AND (
        -- Exact source identifiers: the uuid5 the source is keyed under, under
        -- either metadata name, or the playlist whose videos carry its uuid5.
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5' = :video
        OR value->'document'->'kwargs'->'metadata'->>'namespace_filename' = :video
        OR value->'document'->'kwargs'->'metadata'->>'playlist_namespace_filename' = :video
        -- Human-readable fallbacks: the file name, the source URL, the name of
        -- the audio the transcript came from, or a fragment of any of them.
        OR value->'document'->'kwargs'->'metadata'->>'filename' ILIKE '%' || :video || '%'
        OR value->'document'->'kwargs'->'metadata'->>'audio_filename' ILIKE '%' || :video || '%'
        OR value->'document'->'kwargs'->'metadata'->>'source' ILIKE '%' || :video || '%'
      )
  AND value->'document'->'kwargs'->>'page_content' IS NOT NULL
  -- Transcribed text only. audio_filename is the marker the transcription path
  -- stamps; the content_type / type tests keep a video whose documents predate
  -- that stamp from dropping out of the result.
  AND (
        value->'document'->'kwargs'->'metadata'->>'audio_filename' IS NOT NULL
        OR value->'document'->'kwargs'->'metadata'->>'content_type' LIKE 'video/%'
        OR value->'document'->'kwargs'->'metadata'->>'content_type' LIKE 'audio/%'
        OR value->'document'->'kwargs'->'metadata'->>'type' IN ('audio', 'video')
      )
ORDER BY
    COALESCE(
        value->'document'->'kwargs'->'metadata'->>'filename_uuid5',
        value->'document'->'kwargs'->'metadata'->>'namespace_filename',
        value->'document'->'kwargs'->'metadata'->>'document_id'
    ),
    value->'document'->'kwargs'->'metadata'->>'namespace' NULLS LAST,
    (NULLIF(value->'document'->'kwargs'->'metadata'->>'chunk_index', ''))::int NULLS LAST,
    (NULLIF(value->'document'->'kwargs'->'metadata'->>'start', ''))::float NULLS LAST,
    prefix,
    key;
