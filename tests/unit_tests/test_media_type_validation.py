"""Media-type validation and per-item batch isolation for media uploads.

Two behaviors are covered:

1. The declared ``Content-Type`` of an upload is a hint derived from the
   filename extension by the client (curl and browsers both do this), so a JPEG
   that was saved as ``screenshot.PNG`` arrives declared ``image/png``. The
   magic bytes decide what the file actually is.
2. A multi-item upload is a set of independent jobs. An item that cannot be
   turned into a job is skipped and reported; every other item still runs.
"""

import io
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

# Magic-byte prefixes long enough to satisfy the sniffer's fixed-length checks.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 512
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 512
PDF_BYTES = b"%PDF-1.7\n" + b"0" * 512


def _upload_file(filename: str, data: bytes, content_type: str) -> UploadFile:
    """Build an UploadFile the way multipart parsing does, with a declared type."""
    return UploadFile(
        filename=filename,
        file=io.BytesIO(data),
        headers=Headers({"content-type": content_type}),
    )


# --------------------------------------------------------------------------- #
# validate_upload_image_bytes: file contents win over the declared type
# --------------------------------------------------------------------------- #


def test_jpeg_declared_as_png_is_accepted_as_jpeg():
    """A JPEG named ``.PNG`` (curl declares image/png from the extension) is a
    perfectly ingestible still image; the sniffed type is returned, not a 400."""
    from src.api.webapp import validate_upload_image_bytes

    assert validate_upload_image_bytes("image/png", JPEG_BYTES) == "image/jpeg"


def test_png_declared_as_jpeg_is_accepted_as_png():
    """The mislabeling is symmetric: the magic bytes are always the authority."""
    from src.api.webapp import validate_upload_image_bytes

    assert validate_upload_image_bytes("image/jpeg", PNG_BYTES) == "image/png"


def test_octet_stream_upload_is_resolved_from_contents():
    from src.api.webapp import validate_upload_image_bytes

    assert (
        validate_upload_image_bytes("application/octet-stream", JPEG_BYTES)
        == "image/jpeg"
    )


def test_non_image_contents_declared_as_image_are_rejected_with_actual_type():
    """Contents that are not an image are still refused — and the message names
    what the file actually is rather than blaming the mismatch."""
    from src.api.webapp import validate_upload_image_bytes

    with pytest.raises(HTTPException) as excinfo:
        validate_upload_image_bytes("image/png", PDF_BYTES)
    assert excinfo.value.status_code == 400
    assert "application/pdf" in str(excinfo.value.detail)


def test_unrecognized_bytes_fall_back_to_the_declaration():
    """When the sniffer recognizes nothing, the declared type is all there is."""
    from src.api.webapp import validate_upload_image_bytes

    unknown = b"NOTAMAGICNUMBER" + b"\x00" * 512
    assert validate_upload_image_bytes("image/png", unknown) == "image/png"
    with pytest.raises(HTTPException):
        validate_upload_image_bytes("application/octet-stream", unknown)
    with pytest.raises(HTTPException):
        validate_upload_image_bytes("image/tiff", unknown)


# --------------------------------------------------------------------------- #
# _build_media_entries_for_file: contents decide the pipeline branch
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mislabeled_image_file_builds_an_image_entry():
    from src.api.webapp import _build_media_entries_for_file

    entries = await _build_media_entries_for_file(
        "evan_woods_writing_style_data.PNG",
        JPEG_BYTES,
        "image/png",
        reference_image=False,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
    )
    assert len(entries) == 1
    assert entries[0]["content_type"] == "image/jpeg"
    assert entries[0]["base64_encoded_str"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_pdf_declared_as_octet_stream_takes_the_pdf_branch():
    """An extension-less PDF (or a client that sends application/octet-stream)
    must not fall through to the plain-text branch and be ingested as binary."""
    from src.api.webapp import _build_media_entries_for_file

    entries = await _build_media_entries_for_file(
        "Application_may_1_2025",
        PDF_BYTES,
        "application/octet-stream",
        reference_image=False,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
    )
    assert len(entries) == 1
    assert entries[0]["content_type"] == "application/pdf"


# --------------------------------------------------------------------------- #
# Endpoint: one unusable item does not discard the rest of the batch
# --------------------------------------------------------------------------- #


@pytest.fixture
def upload_endpoint_environment(monkeypatch):
    """Stub the endpoint's collaborators (auth, assistant lookup, metering,
    store, background runner) so the request path can be exercised in-process."""
    import src.api.webapp as webapp

    monkeypatch.setattr(webapp, "enforce_tier_capability", lambda *a, **k: None)

    class _Assistants:
        async def get(self, assistant_id):
            return {
                "metadata": {"user_id": "u1"},
                "name": "Avatar",
                "description": "d",
            }

    monkeypatch.setattr(
        webapp, "get_client", lambda **k: SimpleNamespace(assistants=_Assistants())
    )

    async def _estimate(entries):
        for entry in entries:
            entry["estimated_tokens"] = 1
        return len(entries)

    monkeypatch.setattr(webapp, "_estimate_media_entries_tokens", _estimate)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(webapp, "enforce_remaining_allotment", _noop)
    monkeypatch.setattr(webapp, "enforce_token_rate_limit", _noop)
    monkeypatch.setattr(
        webapp,
        "resolve_metering_bypass",
        lambda user: SimpleNamespace(
            skips_metering_writes=True, usage_response_fields=lambda: {}
        ),
    )

    async def _usage_snapshot(*a, **k):
        return {}

    monkeypatch.setattr(webapp, "_build_meter_usage_snapshot", _usage_snapshot)
    monkeypatch.setattr(webapp, "run_batch_media_job", _noop)

    class _Store:
        async def asearch(self, namespace, limit=None):
            return []

    webapp.app.state.store = _Store()
    webapp.app.state.media_jobs = {}
    webapp.app.state.context = SimpleNamespace(media_processing_concurrency=1)
    webapp.app.state.stripe = None
    webapp.app.state.pool = None
    return webapp


@pytest.mark.asyncio
async def test_unprocessable_item_is_skipped_and_the_batch_continues(
    upload_endpoint_environment,
):
    """A batch carrying one item that fails validation still starts jobs for the
    others, and reports the skipped item instead of failing the whole request."""
    webapp = upload_endpoint_environment

    good_markdown = _upload_file(
        "Application.md", b"Some application prose.", "text/markdown"
    )
    good_image = _upload_file("photo.PNG", JPEG_BYTES, "image/png")
    # image/tiff is not an allowed still image and its bytes sniff to nothing,
    # so entry building raises for this item and this item only.
    bad_item = _upload_file("scan.tiff", b"II*\x00unknown" + b"\x00" * 64, "image/tiff")

    response = await webapp.update_avatar_identity_with_media(
        files=[good_markdown, bad_item, good_image],
        assistant_id="a1",
        current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
    )

    assert response.status_code == 202
    payload = json.loads(response.body)
    assert payload["items_accepted"] == 2
    assert sorted(payload["filenames"]) == ["Application.md", "photo.PNG"]
    assert payload["items_rejected"] == 1
    assert payload["rejected"][0]["filename"] == "scan.tiff"
    assert "image/tiff" in payload["rejected"][0]["reason"]
    assert "skipped 1 unprocessable item(s)" in payload["message"]


@pytest.mark.asyncio
async def test_request_fails_only_when_every_item_is_rejected(
    upload_endpoint_environment,
):
    """With nothing left to process the request is still a 400 — carrying the
    per-item reasons so the caller knows which upload to fix."""
    webapp = upload_endpoint_environment

    bad_one = _upload_file("scan.tiff", b"II*\x00unknown" + b"\x00" * 64, "image/tiff")
    bad_two = _upload_file("archive.zip", b"PK\x03\x04" + b"\x00" * 64, "application/zip")

    with pytest.raises(HTTPException) as excinfo:
        await webapp.update_avatar_identity_with_media(
            files=[bad_one, bad_two],
            assistant_id="a1",
            current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
        )

    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert isinstance(detail, dict)
    assert len(detail["rejected"]) == 2


@pytest.mark.asyncio
async def test_fandom_wiki_url_is_accepted_instead_of_400(
    upload_endpoint_environment, monkeypatch
):
    """A Fandom character page used to 400: the HTML skin is Cloudflare-
    challenged. The parse API on the same host is not, so the upload must
    accept the URL and start a job."""
    webapp = upload_endpoint_environment
    html = (
        b"<html><head><title>Jester Lavorre</title></head>"
        b"<body><p>bio</p></body></html>"
    )

    async def fake_mediawiki(url):
        return html, "text/html"

    monkeypatch.setattr(
        "src.anubis.utils.classes.URLDocumentLoaderClass.fetch_mediawiki_article_html",
        fake_mediawiki,
    )

    response = await webapp.update_avatar_identity_with_media(
        files=None,
        url=["https://criticalrole.fandom.com/wiki/Jester_Lavorre"],
        assistant_id="a1",
        current_user={"identities": [{"user_id": "u1"}], "API_KEY": "k"},
    )

    assert response.status_code == 202
    payload = json.loads(response.body)
    assert payload["items_accepted"] >= 1
    assert payload.get("items_rejected", 0) == 0
