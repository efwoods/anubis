"""Relearn a fact about the user: replace the stored fact, any number of times.

``learn_information_about_the_user`` only ever ADDED facts. A user who moved
from Boston to Denver ended up with both "I live in Boston." and "I live in
Denver." stored, and the avatar had no way to tell which one is current. A fact
about the user is mutable without limit: the newest statement replaces the
earlier statement, and the earlier statement is deleted from the store.

Two pieces make a replacement real:

- ``replace_user_facts`` deletes the stored fact the new fact replaces. The
  stored fact is matched by exact text (case and trailing punctuation aside),
  never by similarity, so a replacement can never delete an unrelated fact.
  When the text matches nothing, the nearest stored facts are returned so the
  avatar can name the stored fact exactly.
- ``reconcile_documents_with_store`` drops a deleted fact from the documents a
  conversation thread carries in graph state, and refreshes a fact that was
  rewritten. Thread state keeps every user fact the thread ever retrieved, so
  without the reconciliation a fact deleted from the store would stay in the
  system prompt of every conversation that had already seen the fact.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.anubis.utils.learning.feedback import _item_document, list_namespace_items

logger = logging.getLogger(__name__)

_NEAREST_STORED_FACT_COUNT = 5


_MAXIMUM_STORED_FACTS_COMPARED = 5


class OutdatedUserFacts(BaseModel):
    """The stored facts about the user that a new fact about the user makes outdated."""

    outdated_stored_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Every stored fact that the NEW_FACT makes outdated, VERBATIM as given in "
            "STORED_FACTS. Empty when the NEW_FACT makes no stored fact outdated."
        ),
    )
    reason: str = Field(description="One short sentence explaining the decision.")


OUTDATED_USER_FACTS_SYSTEM_PROMPT = """You compare one NEW_FACT a user just stated about \
the user against STORED_FACTS the same user stated earlier. You decide which stored facts \
the NEW_FACT makes outdated.

A stored fact is outdated when the stored fact and the NEW_FACT state a current value of \
the same attribute of the user and the two values cannot both be true now: where the user \
lives ("I live in Boston." against "I live in Denver now."), the user's name, the user's \
employer or job, the user's relationship status, the user's age, a current preference or \
opinion the user reversed ("I love sushi." against "I do not like sushi anymore."), or a \
stored absence the NEW_FACT fills ("I do not have a pet." against "I have a dog.").

A stored fact is NOT outdated when:
- the stored fact and the NEW_FACT can both be true now ("I have a dog." and "I have a cat.");
- the stored fact is about a different attribute, a different person, or a different thing;
- the stored fact is a past event, and the NEW_FACT does not say the past event never \
happened ("I ran a marathon in 2019." stays when the user says "I stopped running.");
- the stored fact only shares a topic with the NEW_FACT.

List each outdated stored fact VERBATIM from STORED_FACTS. When no stored fact is outdated, \
return an empty list. When unsure whether a stored fact is outdated, do not list the stored \
fact: keeping a fact is safer than deleting a fact.
"""


async def find_outdated_user_facts(new_fact: str, stored_facts: Sequence[str]) -> list[str]:
    """Return the ``stored_facts`` that ``new_fact`` makes outdated, verbatim.

    One structured model call, made only when there are stored facts to compare.
    A returned text that is not one of ``stored_facts`` is discarded, so the
    model can never name a fact that is not stored. A model failure returns an
    empty list: the new fact is then stored beside the earlier facts, which is
    the behaviour before this check existed.
    """
    candidates = [
        fact
        for fact in dict.fromkeys(stored_facts or [])
        if (fact or "").strip()
        and normalize_fact_text(fact) != normalize_fact_text(new_fact)
    ][:_MAXIMUM_STORED_FACTS_COMPARED]
    if not candidates or not (new_fact or "").strip():
        return []
    numbered = "\n".join(f"{index + 1}. {fact}" for index, fact in enumerate(candidates))
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.anubis.utils.model import init_model

        model = init_model(response_format=OutdatedUserFacts)
        response = await model.ainvoke(
            [
                SystemMessage(content=OUTDATED_USER_FACTS_SYSTEM_PROMPT),
                HumanMessage(content=f"NEW_FACT: {new_fact}\n\nSTORED_FACTS:\n{numbered}"),
            ]
        )
    except Exception:
        logger.exception("The outdated user fact check failed; no stored fact is replaced")
        return []
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, dict):
        response = OutdatedUserFacts.model_validate(response)
    by_normalized = {normalize_fact_text(fact): fact for fact in candidates}
    outdated: list[str] = []
    for text in getattr(response, "outdated_stored_facts", None) or []:
        stored = by_normalized.get(normalize_fact_text(text))
        if stored and stored not in outdated:
            outdated.append(stored)
    return outdated


def normalize_fact_text(fact: str) -> str:
    """Return ``fact`` lowercased, with collapsed spaces and no trailing punctuation."""
    return " ".join((fact or "").lower().split()).rstrip(".!?;: ")


async def replace_user_facts(
    store: Any, namespace: tuple, replaced_fact_texts: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """Delete each stored fact named in ``replaced_fact_texts`` from ``namespace``.

    Returns ``(deleted_facts, unmatched_texts, nearest_stored_facts)``:
    the stored facts that were deleted, the requested texts that matched no
    stored fact, and, when any text was unmatched, the stored facts nearest to
    the unmatched texts so the caller can name the stored fact exactly.
    """
    wanted = [text for text in (replaced_fact_texts or []) if (text or "").strip()]
    if store is None or not wanted:
        return [], [], []
    items = await list_namespace_items(store, namespace)
    deleted_facts: list[str] = []
    unmatched_texts: list[str] = []
    for text in wanted:
        normalized = normalize_fact_text(text)
        matched = False
        for item in items:
            document = _item_document(item)
            if document is None:
                continue
            stored_fact = str(document.metadata.get("fact") or "")
            if normalize_fact_text(stored_fact) != normalized:
                continue
            key = getattr(item, "key", None) or document.metadata.get("document_id")
            await store.adelete(namespace, key)
            deleted_facts.append(stored_fact)
            matched = True
        if not matched:
            unmatched_texts.append(text)
    nearest_stored_facts: list[str] = []
    for text in unmatched_texts:
        try:
            nearest_items = await store.asearch(
                namespace, query=text, limit=_NEAREST_STORED_FACT_COUNT
            )
        except Exception as search_error:  # noqa: BLE001 - the listing is advisory
            logger.debug("Nearest stored fact search failed: %s", search_error)
            nearest_items = []
        for item in nearest_items or []:
            document = _item_document(item)
            stored_fact = str((document.metadata if document else {}).get("fact") or "")
            if stored_fact and stored_fact not in nearest_stored_facts:
                nearest_stored_facts.append(stored_fact)
    return deleted_facts, unmatched_texts, nearest_stored_facts


async def reconcile_documents_with_store(
    store: Any, namespace: tuple, documents: Sequence[Document] | None
) -> list[Document]:
    """Return ``documents`` as the store holds the documents now.

    A document whose store record was deleted is dropped; a document whose
    store record changed is replaced by the stored copy. A document with no
    ``document_id`` cannot be checked and is kept. A store failure keeps every
    document, so an outage never empties the prompt.
    """
    prior_documents = list(documents or [])
    if store is None or not prior_documents:
        return prior_documents

    async def _current_copy(document: Document) -> Document | None:
        key = (getattr(document, "metadata", None) or {}).get("document_id")
        if not key:
            return document
        item = await store.aget(namespace, str(key))
        if item is None:
            return None
        return _item_document(item) or document

    try:
        current_copies = await asyncio.gather(
            *(_current_copy(document) for document in prior_documents)
        )
    except Exception as reconcile_error:  # noqa: BLE001 - never fail a turn on a read
        logger.warning("User fact reconciliation failed: %s", reconcile_error)
        return prior_documents
    return [document for document in current_copies if document is not None]
