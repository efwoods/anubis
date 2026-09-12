"""Organization websites are lifted out of identity so the avatar can share them."""

from langchain_core.documents import Document

from src.anubis.utils.organization_links import (
    organization_links_from_identity,
    organization_links_from_text,
    render_organization_links_section,
)

CLAIRE_PLACE_URL = "https://clairesplacefoundation.org/"
GRANT_STEAM_URL = "https://www.grantimaharafoundation.org/"


def test_claire_wineland_description_yields_claires_place_foundation() -> None:
    links = organization_links_from_text(
        "Founder of Claire's Place Foundation, supporting families living "
        f"with cystic fibrosis. {CLAIRE_PLACE_URL}"
    )
    assert links == [CLAIRE_PLACE_URL]


def test_grant_imahara_description_yields_the_steam_foundation() -> None:
    links = organization_links_from_text(
        "Engineer and founder of Grant Imahara's STEAM Foundation. "
        f"{GRANT_STEAM_URL}"
    )
    assert links == [GRANT_STEAM_URL]


def test_youtube_media_sources_are_not_organization_links() -> None:
    assert organization_links_from_text(
        "Talks at https://www.youtube.com/@imahara and "
        f"the foundation at {GRANT_STEAM_URL}"
    ) == [GRANT_STEAM_URL]


def test_identity_documents_contribute_source_metadata_urls() -> None:
    documents = [
        Document(
            page_content="I founded Claire's Place Foundation.",
            metadata={"source_url": CLAIRE_PLACE_URL},
        )
    ]
    assert organization_links_from_identity(
        assistant_description="Claire Wineland",
        identity_documents=documents,
    ) == [CLAIRE_PLACE_URL]


def test_organization_links_section_lists_claire_and_grant() -> None:
    section = render_organization_links_section(
        [CLAIRE_PLACE_URL, GRANT_STEAM_URL]
    )
    assert "share the matching URL" in section
    assert CLAIRE_PLACE_URL in section
    assert GRANT_STEAM_URL in section
    assert render_organization_links_section([]) == ""
