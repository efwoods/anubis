"""created_at and DEV text-model metadata on a transcript row."""

from langchain_core.messages import AIMessage, HumanMessage


def test_additional_kwargs_with_created_at_stamps_a_missing_instant():
    from src.anubis.utils.message_record import additional_kwargs_with_created_at

    stamped = additional_kwargs_with_created_at({"kind": "spoken"})
    assert stamped["kind"] == "spoken"
    assert stamped["created_at"]
    already = additional_kwargs_with_created_at(
        {"created_at": "2026-09-13T15:08:00+00:00"}
    )
    assert already["created_at"] == "2026-09-13T15:08:00+00:00"


def test_attach_visible_reply_metadata_writes_created_at_and_dev_text_model(
    monkeypatch,
):
    from src.anubis.utils import message_record
    from src.anubis.utils.model import record_text_inference

    monkeypatch.setenv("DEV", "TRUE")
    monkeypatch.setenv("MODEL_PROVIDER", "OPEN_AI")
    monkeypatch.setenv("MODEL", "gpt-5.6-luna")
    from src.anubis.utils.context import GlobalContext

    record_text_inference(
        model_provider="OPEN_AI",
        model="gpt-5.6-luna",
    )
    reply = AIMessage(content="Hello.")
    message_record.attach_visible_reply_metadata(reply, context=GlobalContext())
    assert reply.response_metadata["created_at"]
    assert reply.additional_kwargs["created_at"] == reply.response_metadata["created_at"]
    assert reply.response_metadata["text_model"] == "gpt-5.6-luna"
    assert reply.response_metadata["text_model_provider"] == "OPEN_AI"


def test_human_message_keeps_an_existing_created_at():
    from src.anubis.utils.message_record import additional_kwargs_with_created_at

    kwargs = additional_kwargs_with_created_at(
        {"created_at": "2026-09-13T12:00:00+00:00", "hidden": False}
    )
    human = HumanMessage(content="hey", additional_kwargs=kwargs)
    assert human.additional_kwargs["created_at"] == "2026-09-13T12:00:00+00:00"


def test_attach_visible_reply_metadata_omits_text_model_outside_dev(monkeypatch):
    from src.anubis.utils import message_record

    monkeypatch.setenv("DEV", "FALSE")
    monkeypatch.setenv("MODEL_PROVIDER", "OPEN_AI")
    monkeypatch.setenv("MODEL", "gpt-5.6-luna")
    from src.anubis.utils.context import GlobalContext

    reply = AIMessage(content="Hello.")
    message_record.attach_visible_reply_metadata(reply, context=GlobalContext())
    assert reply.response_metadata["created_at"]
    assert "text_model" not in reply.response_metadata
