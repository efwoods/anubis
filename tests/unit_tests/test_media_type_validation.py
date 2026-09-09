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
        # SVG is a drawing the models are not given, so it stays out of the
        # accepted still-image types even when the caller names it.
        validate_upload_image_bytes("image/svg+xml", unknown)


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


# --------------------------------------------------------------------------- #
# WebP: a client that cannot name the extension must not lose the picture
# --------------------------------------------------------------------------- #

# A still WebP: the RIFF container, the WEBP form type, and a lossy chunk tag.
WEBP_BYTES = b"RIFF" + (512).to_bytes(4, "little") + b"WEBPVP8 " + b"\x00" * 512


def test_webp_declared_as_octet_stream_resolves_to_webp():
    """curl and several mobile pickers have no ``.webp`` row in their extension
    table and send ``application/octet-stream``; the magic bytes name the type."""
    from src.api.webapp import effective_upload_mime_type

    assert effective_upload_mime_type("application/octet-stream", WEBP_BYTES) == (
        "image/webp"
    )
    assert effective_upload_mime_type("", WEBP_BYTES) == "image/webp"
    assert effective_upload_mime_type("text/plain", WEBP_BYTES) == "image/webp"


def test_a_declaration_agreeing_with_the_bytes_is_kept_as_sent():
    """The sniffer's category is coarser than a client's declaration, so a
    QuickTime recording keeps ``video/quicktime`` rather than becoming mp4, and
    a type the sniffer cannot recognize at all is left exactly as declared."""
    from src.api.webapp import effective_upload_mime_type

    quicktime = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 64
    assert effective_upload_mime_type("video/quicktime", quicktime) == "video/quicktime"
    assert effective_upload_mime_type("text/csv", b"a,b\n1,2\n") == "text/csv"


@pytest.mark.asyncio
async def test_webp_declared_as_octet_stream_builds_an_image_entry():
    from src.api.webapp import _build_media_entries_for_file

    entries = await _build_media_entries_for_file(
        "portrait.webp",
        WEBP_BYTES,
        "application/octet-stream",
        reference_image=False,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
    )
    assert len(entries) == 1
    assert entries[0]["content_type"] == "image/webp"
    assert entries[0]["base64_encoded_str"].startswith("data:image/webp;base64,")


@pytest.mark.asyncio
async def test_webp_reference_portrait_declared_as_octet_stream_is_accepted():
    """The reference portrait is a single still image; a WebP the client could
    not name is still that still image."""
    from src.api.webapp import _build_media_entries_for_file

    entries = await _build_media_entries_for_file(
        "portrait.webp",
        WEBP_BYTES,
        "application/octet-stream",
        reference_image=True,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
    )
    assert len(entries) == 1
    assert entries[0]["reference_image"] is True
    assert entries[0]["content_type"] == "image/webp"


@pytest.mark.asyncio
async def test_message_attachment_webp_declared_as_octet_stream_is_an_image_block():
    """On ``POST /message/{assistant_id}`` the same WebP must reach the model as
    an image block; routing it by the declaration alone left the avatar
    answering from the filename."""
    from src.api.webapp import process_files_for_message

    _text, multimodal_content, image_filenames = await process_files_for_message(
        files=[
            _upload_file("portrait.webp", WEBP_BYTES, "application/octet-stream")
        ],
        message="What is in this picture?",
    )
    assert image_filenames == ["portrait.webp"]
    assert multimodal_content is not None
    image_blocks = [
        block for block in multimodal_content if block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/webp;base64,")


@pytest.mark.asyncio
async def test_message_attachment_text_file_stays_text():
    """Resolving the type from the bytes must not turn a document into an image:
    the sniffer recognizes no text format, so the declaration still decides."""
    from src.api.webapp import process_files_for_message

    text_content, multimodal_content, image_filenames = (
        await process_files_for_message(
            files=[_upload_file("notes.txt", b"a line of notes", "text/plain")],
            message="",
        )
    )
    assert multimodal_content is None
    assert image_filenames == []
    assert "a line of notes" in text_content


# --------------------------------------------------------------------------- #
# Phone and design-tool formats: accepted, and transcoded for the model vendors
# --------------------------------------------------------------------------- #


def _still_image_bytes(image_format: str, *, transparent: bool = False) -> bytes:
    """Encode a small picture in one of the formats a person actually uploads."""
    from PIL import Image

    if image_format in ("HEIF", "AVIF"):
        pytest.importorskip("pillow_heif") if image_format == "HEIF" else None
        if image_format == "HEIF":
            import pillow_heif

            pillow_heif.register_heif_opener()
    picture = Image.new(
        "RGBA" if transparent else "RGB",
        (48, 48),
        (12, 180, 96, 128) if transparent else (12, 180, 96),
    )
    encoded = io.BytesIO()
    picture.save(encoded, format=image_format)
    return encoded.getvalue()


def test_iso_base_media_brands_separate_photos_from_movies():
    """HEIC, AVIF and MP4 share one container; the ``ftyp`` brand is what says
    whether the bytes are a photograph or a movie."""
    from src.api.webapp import _sniff_media_category_from_bytes

    padding = b"\x00" * 32
    assert (
        _sniff_media_category_from_bytes(b"\x00\x00\x00\x1cftypheic" + padding)
        == "image/heic"
    )
    assert (
        _sniff_media_category_from_bytes(b"\x00\x00\x00\x1cftypmif1" + padding)
        == "image/heif"
    )
    assert (
        _sniff_media_category_from_bytes(b"\x00\x00\x00\x20ftypavif" + padding)
        == "image/avif"
    )
    assert (
        _sniff_media_category_from_bytes(b"\x00\x00\x00\x18ftypmp42" + padding)
        == "video/mp4"
    )
    assert (
        _sniff_media_category_from_bytes(b"\x00\x00\x00\x14ftypqt  " + padding)
        == "video/mp4"
    )


def test_bmp_and_tiff_are_recognized_from_their_magic_bytes():
    from src.api.webapp import _sniff_media_category_from_bytes

    assert _sniff_media_category_from_bytes(b"BM" + b"\x00" * 64) == "image/bmp"
    assert _sniff_media_category_from_bytes(b"II\x2a\x00" + b"\x00" * 64) == "image/tiff"
    assert _sniff_media_category_from_bytes(b"MM\x00\x2a" + b"\x00" * 64) == "image/tiff"


@pytest.mark.parametrize(
    "image_format",
    ["AVIF", "BMP", "TIFF", pytest.param("HEIF", id="HEIC")],
)
def test_phone_and_scanner_formats_are_transcoded_to_jpeg(image_format):
    """A HEIC photo, an AVIF export, a BMP screenshot and a TIFF scan are all
    accepted, and what comes back is a format every vision model reads."""
    if image_format == "HEIF":
        pytest.importorskip("pillow_heif")
    from src.api.webapp import MODEL_READABLE_IMAGE_MIMES, prepare_still_image_upload

    prepared_mime, prepared_bytes = prepare_still_image_upload(
        "application/octet-stream", _still_image_bytes(image_format)
    )
    assert prepared_mime == "image/jpeg"
    assert prepared_mime in MODEL_READABLE_IMAGE_MIMES
    assert prepared_bytes.startswith(b"\xff\xd8\xff")


def test_a_transparent_image_becomes_png_so_it_does_not_go_black():
    """JPEG has no alpha channel, so an image that carries transparency is
    re-encoded as PNG instead."""
    from src.api.webapp import prepare_still_image_upload

    prepared_mime, prepared_bytes = prepare_still_image_upload(
        "image/tiff", _still_image_bytes("TIFF", transparent=True)
    )
    assert prepared_mime == "image/png"
    assert prepared_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_an_already_readable_image_is_passed_through_untouched():
    """Transcoding is only for the formats the vendors cannot read: a JPEG is
    handed on as the very bytes the person uploaded."""
    from src.api.webapp import prepare_still_image_upload

    prepared_mime, prepared_bytes = prepare_still_image_upload("image/jpeg", JPEG_BYTES)
    assert prepared_mime == "image/jpeg"
    assert prepared_bytes is JPEG_BYTES


def test_declared_image_aliases_are_folded_to_one_spelling():
    """Clients and web servers spell these formats several ways; the bytes are
    the same picture, so the names are folded before anything is decided."""
    from src.api.webapp import normalize_declared_image_mime

    assert normalize_declared_image_mime("image/x-ms-bmp") == "image/bmp"
    assert normalize_declared_image_mime("IMAGE/TIF") == "image/tiff"
    assert normalize_declared_image_mime("image/heic-sequence") == "image/heic"
    assert normalize_declared_image_mime("image/jpg") == "image/jpeg"


@pytest.mark.asyncio
async def test_heic_photo_builds_an_image_entry_the_pipeline_can_read():
    """An iPhone photo declared ``application/octet-stream`` becomes an entry
    whose stored type and stored bytes are both JPEG."""
    pytest.importorskip("pillow_heif")
    from src.api.webapp import _build_media_entries_for_file

    entries = await _build_media_entries_for_file(
        "IMG_4821.HEIC",
        _still_image_bytes("HEIF"),
        "application/octet-stream",
        reference_image=True,
        reference_audio=False,
        user_id="u1",
        assistant_id="a1",
    )
    assert len(entries) == 1
    assert entries[0]["content_type"] == "image/jpeg"
    assert entries[0]["base64_encoded_str"].startswith("data:image/jpeg;base64,")
    assert entries[0]["content"].startswith(b"\xff\xd8\xff")


@pytest.mark.asyncio
async def test_message_attachment_heic_photo_becomes_a_jpeg_image_block():
    pytest.importorskip("pillow_heif")
    from src.api.webapp import process_files_for_message

    _text, multimodal_content, image_filenames = await process_files_for_message(
        files=[
            _upload_file(
                "IMG_4821.HEIC",
                _still_image_bytes("HEIF"),
                "application/octet-stream",
            )
        ],
        message="What is in this photo?",
    )
    assert image_filenames == ["IMG_4821.HEIC"]
    image_blocks = [
        block for block in multimodal_content if block.get("type") == "image_url"
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_an_unreadable_attached_image_is_described_not_fatal():
    """A picture that cannot be prepared must not take the whole message down:
    the turn still reaches the avatar, carrying a line saying what happened."""
    from src.api.webapp import process_files_for_message

    truncated_heic = b"\x00\x00\x00\x1cftypheic" + b"\x00" * 32
    text_content, multimodal_content, image_filenames = (
        await process_files_for_message(
            files=[
                _upload_file("broken.heic", truncated_heic, "application/octet-stream")
            ],
            message="What is in this photo?",
        )
    )
    assert multimodal_content is None
    assert image_filenames == []
    assert "could not be read" in text_content
