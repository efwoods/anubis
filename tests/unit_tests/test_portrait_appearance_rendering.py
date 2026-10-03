"""The portrait document renders as the avatar's own appearance in === YOUR IDENTITY ===."""

from langchain_core.documents import Document

from src.anubis.utils.classes.DynamicPromptBuilder import (
    PORTRAIT_APPEARANCE_LABEL,
    render_identity_document,
)

# Stored portrait text of avatar 47cfdaa2 (2026-07-26): a refusal preamble
# followed by a third-person description.
REFUSAL_PORTRAIT_TEXT = (
    "I can’t describe this person in the first-person voice as if I were them, "
    "but here is a neutral description of their appearance:\n\n"
    "He appears to be a young man with short, dark hair and rectangular glasses."
)


def test_refusal_preamble_is_dropped_and_portrait_is_labelled_as_own_appearance() -> None:
    portrait_document = Document(
        page_content=REFUSAL_PORTRAIT_TEXT, metadata={"reference_image": True}
    )

    rendered_portrait = render_identity_document(portrait_document)

    assert rendered_portrait.startswith(PORTRAIT_APPEARANCE_LABEL)
    assert "can’t describe" not in rendered_portrait
    assert rendered_portrait.endswith(
        "He appears to be a young man with short, dark hair and rectangular glasses."
    )


def test_sorry_refusal_preamble_is_dropped() -> None:
    portrait_document = Document(
        page_content=(
            "Sorry, I can’t describe someone from a first-person perspective as if I "
            "were that person. I can offer a neutral description of what’s visible: "
            "Short brown hair."
        ),
        metadata={"reference_image": True},
    )

    assert render_identity_document(portrait_document) == (
        f"{PORTRAIT_APPEARANCE_LABEL}\nShort brown hair."
    )


def test_first_person_portrait_keeps_text_with_colons() -> None:
    portrait_text = "I am a woman with dark chestnut hair. My style: casual and warm."
    portrait_document = Document(
        page_content=portrait_text, metadata={"reference_image": True}
    )

    assert render_identity_document(portrait_document) == (
        f"{PORTRAIT_APPEARANCE_LABEL}\n{portrait_text}"
    )


def test_other_identity_documents_render_unchanged() -> None:
    identity_document = Document(page_content="I was born in Ohio.", metadata={})

    assert render_identity_document(identity_document) == "I was born in Ohio."


def test_retrieval_query_keeps_only_the_person_words() -> None:
    """A picture description lengthened every store embedding (715 ms vs 2,910 ms)."""
    from src.anubis.utils.nodes import retrieval_query_without_image_descriptions

    resolved_message = (
        "What do I look like?\n\n---\nImage descriptions:\n"
        "[minecraft_view.jpg]\nA blocky figure with a tan head and blue legs."
    )
    assert retrieval_query_without_image_descriptions(resolved_message) == "What do I look like?"
    only_image = "---\nImage descriptions:\n[photo.jpg]\nA red barn."
    assert retrieval_query_without_image_descriptions(only_image) == only_image
    assert retrieval_query_without_image_descriptions("hello") == "hello"
