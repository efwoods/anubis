"""The reference-audio store helper: write once, replace on request, read back.

Pinned down:

- The first write stores a listable Document (``filename`` /
  ``namespace_filename`` / ``namespace`` metadata) beside the clip.
- A second write without ``replace`` leaves the stored clip alone and returns
  ``None``; with ``replace=True`` the clip is swapped.
- ``read_reference_audio`` round-trips the stored row, and also understands
  the older plain-dictionary document shape.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.anubis.utils.voice import reference_audio

USER_ID = "auth0-user"
ASSISTANT_ID = "assistant-1"


class _Store:
    def __init__(self):
        self.rows = {}
        self.writes = 0

    async def aget(self, namespace, key):
        value = self.rows.get((namespace, key))
        return None if value is None else SimpleNamespace(value=value)

    async def aput(self, namespace, key, value):
        self.writes += 1
        await asyncio.sleep(0)  # let a concurrent writer interleave if unlocked
        self.rows[(namespace, key)] = value


async def _write(store, filename, *, replace=False):
    return await reference_audio.store_reference_audio(
        store,
        user_id=USER_ID,
        assistant_id=ASSISTANT_ID,
        audio_data_uri=f"data:audio/mp3;base64,{filename}",
        transcript_text=f"transcript of {filename}",
        filename=filename,
        namespace_filename=f"key-{filename}",
        duration_seconds=4.5,
        source="upload",
        replace=replace,
    )


@pytest.mark.asyncio
async def test_the_first_write_stores_a_listable_document():
    store = _Store()
    document = await _write(store, "Mom.m4a")
    assert document is not None
    assert document.metadata["filename"] == "Mom.m4a"
    assert document.metadata["namespace_filename"] == "key-Mom.m4a"
    assert document.metadata["namespace"] == "reference_audio"
    assert document.metadata["reference_audio"] is True
    stored = await reference_audio.read_reference_audio(store, USER_ID, ASSISTANT_ID)
    assert stored == {
        "audio_data_uri": "data:audio/mp3;base64,Mom.m4a",
        "transcript_text": "transcript of Mom.m4a",
        "filename": "Mom.m4a",
        "namespace_filename": "key-Mom.m4a",
        "duration_seconds": 4.5,
    }


@pytest.mark.asyncio
async def test_a_later_write_never_replaces_the_reference_unless_asked():
    store = _Store()
    await _write(store, "Mom.m4a")
    assert await _write(store, "talk.mp4") is None
    stored = await reference_audio.read_reference_audio(store, USER_ID, ASSISTANT_ID)
    assert stored["filename"] == "Mom.m4a"

    replaced = await _write(store, "talk.mp4", replace=True)
    assert replaced is not None
    stored = await reference_audio.read_reference_audio(store, USER_ID, ASSISTANT_ID)
    assert stored["filename"] == "talk.mp4"
    assert store.writes == 2


@pytest.mark.asyncio
async def test_concurrent_writers_store_exactly_one_reference():
    store = _Store()
    results = await asyncio.gather(
        _write(store, "first.m4a"), _write(store, "second.m4a"), _write(store, "third.m4a")
    )
    assert sum(1 for document in results if document is not None) == 1
    assert store.writes == 1


@pytest.mark.asyncio
async def test_reading_understands_the_older_plain_document_shape():
    store = _Store()
    namespace = reference_audio.reference_audio_namespace(USER_ID, ASSISTANT_ID)
    store.rows[(namespace, ASSISTANT_ID)] = {
        "reference_audio_data": "data:audio/mp3;base64,QUJD",
        "document": {
            "page_content": "hello",
            "metadata": {"reference_audio": True, "source": "recorder"},
        },
    }
    stored = await reference_audio.read_reference_audio(store, USER_ID, ASSISTANT_ID)
    assert stored["transcript_text"] == "hello"
    assert stored["filename"] is None
    assert stored["audio_data_uri"] == "data:audio/mp3;base64,QUJD"


@pytest.mark.asyncio
async def test_a_missing_or_broken_store_reads_as_no_reference():
    assert await reference_audio.read_reference_audio(None, USER_ID, ASSISTANT_ID) is None
