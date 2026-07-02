"""Unit tests for JSON / JSON-Lines / tabular-JSON upload ingestion.

These cover the three ingestion repairs for the Elon-Musk-tweet-dataset bug
reports:

* ``_parse_json_or_json_lines_upload`` (process_media_graph nodes) — ``.jsonl``
  files are parsed line-by-line into the ``{"statements": [...]}`` contract
  instead of failing whole-file ``json.loads`` with "Extra data";
* ``_normalize_tabular_json_to_rows`` / ``_is_json_upload`` (webapp edge) —
  pandas ``to_json`` tables (orient="columns" / orient="records") are detected
  and normalized to ``(headers, rows)`` for the shared CSV statements pipeline,
  while contract-shaped JSON passes through untouched;
* the graph JSON media handler — an unrecognized JSON shape yields a clear
  error Document instead of raising ``KeyError: 'messages'``.
"""

import json

import pytest

from src.subgraphs.process_media_graph.utils.nodes import (
    _parse_json_lines_to_statements_payload,
    _parse_json_or_json_lines_upload,
    process_media_item_task,
)


def _statement_line(content: str) -> str:
    return json.dumps(
        {
            "messages": [{"role": "assistant", "content": content}],
            "metadata": {"target": "Test Target", "source": "test.jsonl"},
        }
    )


""" ------------------------------------------------------------------ """
""" JSON-Lines parsing                                                 """
""" ------------------------------------------------------------------ """


def test_jsonl_suffix_parses_line_by_line_into_statements():
    raw_text = "\n".join(_statement_line(f"quote {i}") for i in range(3))
    payload = _parse_json_or_json_lines_upload(
        raw_text=raw_text, filename="quotes.jsonl", suffix=".jsonl"
    )
    assert set(payload) == {"statements"}
    assert len(payload["statements"]) == 3
    assert payload["statements"][0]["messages"][0]["content"] == "quote 0"


def test_jsonl_skips_corrupt_and_blank_lines():
    raw_text = "\n".join(
        [_statement_line("good one"), "", "{not valid json", _statement_line("good two")]
    )
    payload = _parse_json_lines_to_statements_payload(raw_text, "quotes.jsonl")
    assert len(payload["statements"]) == 2


def test_jsonl_with_no_parseable_object_raises_value_error():
    with pytest.raises(ValueError):
        _parse_json_lines_to_statements_payload("plain text\nnot json", "bad.jsonl")


def test_json_suffix_still_parses_whole_document():
    document = {"messages": [{"role": "assistant", "content": "hello"}]}
    parsed = _parse_json_or_json_lines_upload(
        raw_text=json.dumps(document), filename="conversation.json", suffix=".json"
    )
    assert parsed == document


def test_json_suffix_with_concatenated_objects_falls_back_to_jsonl():
    # Per-line objects saved under a ``.json`` name: whole-file json.loads fails
    # with "Extra data", so the JSON-Lines parse is retried.
    raw_text = "\n".join(_statement_line(f"quote {i}") for i in range(2))
    payload = _parse_json_or_json_lines_upload(
        raw_text=raw_text, filename="quotes.json", suffix=".json"
    )
    assert len(payload["statements"]) == 2


""" ------------------------------------------------------------------ """
""" Tabular JSON detection at the API edge                             """
""" ------------------------------------------------------------------ """


def _webapp():
    # webapp import is deferred so a broken FastAPI-side import cannot take the
    # graph-side tests above down with this module.
    from src.api import webapp

    return webapp


def test_pandas_columns_orient_is_normalized_to_headers_and_rows():
    parsed = {
        "id": {"0": 1, "1": 2},
        "text": {"0": "first tweet", "1": "second tweet"},
        "user_name": {"0": "Elon Musk", "1": "Elon Musk"},
    }
    headers, rows = _webapp()._normalize_tabular_json_to_rows(parsed)
    assert headers == ["id", "text", "user_name"]
    assert rows == [
        {"id": "1", "text": "first tweet", "user_name": "Elon Musk"},
        {"id": "2", "text": "second tweet", "user_name": "Elon Musk"},
    ]


def test_records_orient_is_normalized_to_headers_and_rows():
    parsed = [
        {"text": "first", "favorites": 10},
        {"text": "second", "favorites": None},
    ]
    headers, rows = _webapp()._normalize_tabular_json_to_rows(parsed)
    assert headers == ["text", "favorites"]
    assert rows == [
        {"text": "first", "favorites": "10"},
        {"text": "second", "favorites": ""},
    ]


def test_contract_shapes_are_not_treated_as_tables():
    normalize = _webapp()._normalize_tabular_json_to_rows
    assert normalize({"statements": [{"messages": []}]}) is None
    assert normalize({"messages": [{"role": "assistant", "content": "hi"}]}) is None
    # Statement-shaped records hold nested containers -> not tabular either.
    assert (
        normalize([{"messages": [{"role": "assistant", "content": "hi"}]}]) is None
    )


def test_non_tabular_json_is_not_treated_as_a_table():
    normalize = _webapp()._normalize_tabular_json_to_rows
    assert normalize("just a string") is None
    assert normalize(42) is None
    assert normalize({}) is None
    assert normalize([]) is None
    assert normalize({"config": {"nested": {"deep": True}}}) is None


def test_is_json_upload_detection():
    is_json_upload = _webapp()._is_json_upload
    assert is_json_upload("tweets.json", "application/json")
    # .jsonl commonly arrives as application/octet-stream (the reported bug).
    assert is_json_upload("tweets.jsonl", "application/octet-stream")
    assert is_json_upload("tweets.ndjson", "")
    assert not is_json_upload("notes.txt", "text/plain")
    assert not is_json_upload("table.csv", "text/csv")


""" ------------------------------------------------------------------ """
""" Graph JSON media handler: unrecognized shapes must not crash       """
""" ------------------------------------------------------------------ """


@pytest.mark.asyncio
async def test_unrecognized_json_shape_returns_error_document():
    # A dict with neither "statements" nor "messages" (e.g. arbitrary JSON that
    # bypassed edge preprocessing) used to raise KeyError: 'messages' and sink
    # the whole media item.
    media_item = {
        "type": "json",
        "content": {"unexpected": {"shape": True}},
        "metadata": {
            "filename": "weird.json",
            "user_id": "user-1",
            "assistant_id": "assistant-1",
            "namespace_filename": "weird_json",
        },
    }
    documents = await process_media_item_task(media_item, None, None, None)
    assert len(documents) == 1
    assert documents[0].metadata["status"] == "error"
    assert "unrecognized_json_shape" in documents[0].metadata["error"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
