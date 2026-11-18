"""Remove a name the user rejected from every record the system prompt reads.

A user who says "my name is Evan, do not call me David" expects the name David
to disappear. Before this module, each correction only ADDED records beside the
stale record ("Address me as David in responses"), and every added record
repeated the rejected name ("The user wants to be addressed as Evan and never
as David"). The stale record and the corrections then competed in a similarity
search, and the stale record won on a message about a light switch (avatar
47cfdaa2, 2026-11-18).

``purge_rejected_address_name`` makes a correction subtractive:

- every address preference that mentions the rejected name is deleted;
- every other record whose atomic fact says how the user is named or addressed
  and mentions the rejected name is deleted ("My name isn't David.");
- a record whose fact is unrelated to naming keeps the fact, and loses the
  rejected name from the quoted context ("David, I don't want you to be gone."
  becomes "I don't want you to be gone.");
- a record that mentions the rejected name as a third person ("My brother David
  lives in Ohio.") is left untouched;
- exactly one positive record per kind is kept or written: the address
  preference "Address the user as Evan." and the user fact "My name is Evan."

The rejected name is never written back in any form: a sentence that says
"never David" teaches the name David.

Nothing here is permanent. No name is ever blocked: a name the user rejected
can be stated again later and is learned again, and the user can change the
name any number of times. ``relearn_user_address_name`` makes a newly stated
name ("my name is Sam", "call me Sam") REPLACE the earlier names instead of
piling up beside the earlier names, so the store always holds the one current
answer to "what does the user want to be called".
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from src.anubis.utils.learning.feedback import (
    ADDRESS_PREFERENCE_CATEGORY,
    _item_document,
    list_namespace_items,
    store_user_preference,
)
from src.anubis.utils.learning.namespaces import (
    preference_namespace,
    what_feels_real_namespace,
)

logger = logging.getLogger(__name__)

# Words that make a sentence a statement about how the user is named or addressed.
_NAMING_STATEMENT_PATTERN = re.compile(
    r"\b(name|names|named|naming|call|calls|called|calling|address|addresses|"
    r"addressed|addressing|refer|refers|referred|referring|nickname|"
    r"use|uses|used|using|say|says|said|saying)\b",
    re.IGNORECASE,
)

# A candidate name: one capitalized word, optionally hyphenated or apostrophized.
_NAME_TOKEN = r"[A-Z][A-Za-z'’\-]{1,40}"

# Phrasings in which the user rejects a name for the user ("don't call me David",
# "my name isn't David", "stop calling me David", "I'm not David").
_REJECTED_NAME_PATTERNS = (
    re.compile(
        r"(?i:\b(?:do\s+not|don't|don’t|dont|never|stop|quit|not\s+to)\b)[^.?!\n]{0,40}?"
        r"(?i:\bcall(?:ing|ed)?\s+me\s+(?:by\s+the\s+name\s+)?)[\"'“‘]?(" + _NAME_TOKEN + ")",
    ),
    re.compile(
        r"(?i:\bmy\s+name\s+(?:is\s+not|isn't|isn’t|isnt|ain't|was\s+never)\s+)[\"'“‘]?("
        + _NAME_TOKEN
        + ")",
    ),
    re.compile(
        r"(?i:\bI(?:'m|’m|\s+am)\s+not\s+(?:named\s+|called\s+))[\"'“‘]?(" + _NAME_TOKEN + ")"
    ),
)

# Phrasings in which the user states the name the user wants ("my name is Evan",
# "call me Evan"). The name must be capitalized, so "my name is on the list" and
# "call me later" are not read as names.
_PREFERRED_NAME_PATTERNS = (
    re.compile(r"(?i:\bmy\s+name\s+is\s+)[\"'“‘]?(" + _NAME_TOKEN + ")"),
    re.compile(r"(?i:\bI(?:'m|’m|\s+am)\s+(?:called|named)\s+)[\"'“‘]?(" + _NAME_TOKEN + ")"),
    re.compile(
        r"(?i:(?:^|[.;,!?]\s*|\band\s+|\bplease\s+|\bto\s+)call\s+me\s+)[\"'“‘]?("
        + _NAME_TOKEN
        + ")"
    ),
)

# Capitalized words that follow "call me" without being a name.
_NOT_A_NAME = frozenset(
    {
        "A", "An", "The", "That", "This", "Again", "Anything", "Something", "Names",
        "By", "When", "If", "Back", "Later", "Tomorrow", "Today", "Maybe", "Crazy",
        "Old", "What", "Whatever", "Not", "Never", "Please", "Sir", "Ma'am", "Now",
    }
)


def _clean_name(candidate: str) -> str:
    name = (candidate or "").strip().strip("\"'“”‘’.,;:!?").strip()
    # A possessive is not part of the name ("David's").
    name = re.sub(r"['’]s$", "", name)
    return name


def _is_usable_name(name: str) -> bool:
    return bool(name) and name not in _NOT_A_NAME and len(name) >= 2


def extract_rejected_address_names(message_text: str) -> list[str]:
    """Names the user rejects for the user in ``message_text``, in order, unique."""
    rejected_names: list[str] = []
    for pattern in _REJECTED_NAME_PATTERNS:
        for match in pattern.finditer(message_text or ""):
            name = _clean_name(match.group(1))
            if _is_usable_name(name) and name.lower() not in {
                existing.lower() for existing in rejected_names
            }:
                rejected_names.append(name)
    return rejected_names


def extract_preferred_address_name(message_text: str) -> str | None:
    """Return the name the user states as the user's own name in ``message_text``."""
    rejected_lowercase = {
        name.lower() for name in extract_rejected_address_names(message_text)
    }
    for pattern in _PREFERRED_NAME_PATTERNS:
        for match in pattern.finditer(message_text or ""):
            name = _clean_name(match.group(1))
            if _is_usable_name(name) and name.lower() not in rejected_lowercase:
                return name
    return None


def extract_stated_user_name(message_text: str) -> str | None:
    """Return the name the user states with "my name is <name>" in ``message_text``.

    Only the "my name is" phrasing is read here, because "my name is" states a
    fact about the user in every context. "Call me <name>" is left to the
    avatar's judgement (``learn_user_preference``), because "call me" also
    opens quotations and titles.
    """
    rejected_lowercase = {
        name.lower() for name in extract_rejected_address_names(message_text)
    }
    for match in _PREFERRED_NAME_PATTERNS[0].finditer(message_text or ""):
        name = _clean_name(match.group(1))
        if _is_usable_name(name) and name.lower() not in rejected_lowercase:
            return name
    return None


def _name_pattern(name: str) -> re.Pattern[str]:
    # A following capitalized word makes the match another person's full name
    # ("David Bowie"), which is not the rejected first name.
    return re.compile(
        r"(?<![\w'’\-])(?i:" + re.escape(name) + r")(?![\w\-])(?!\s+[A-Z][a-z])"
    )


def text_mentions_name(text: str, name: str) -> bool:
    """Report whether ``text`` contains ``name`` as a whole word, ignoring case."""
    return bool(name) and bool(_name_pattern(name).search(text or ""))


def fact_names_user_with_rejected_name(fact: str, rejected_name: str) -> bool:
    """Report whether ``fact`` says how the user is named and mentions ``rejected_name``."""
    return text_mentions_name(fact, rejected_name) and bool(
        _NAMING_STATEMENT_PATTERN.search(fact or "")
    )


def strip_vocative_name(text: str, name: str) -> str:
    """Remove ``name`` where ``name`` is spoken in direct address.

    "David, I don't want you to be gone." becomes "I don't want you to be gone.",
    and "Thanks, David." becomes "Thanks." A name used as a subject or an object
    ("David lives in Ohio") is not a direct address and is kept.
    """
    escaped = re.escape(name)
    # Leading direct address: start of the text, of a sentence, or of a quotation.
    leading = re.compile(
        r"(^|[.!?;:]\s+|[\"“‘'(\[]\s*|\n\s*)" + escaped + r"\s*[,!:—–-]\s*",
        re.IGNORECASE,
    )
    stripped = leading.sub(
        lambda match: match.group(1), text or ""
    )
    # Trailing direct address: ", David" before the end of a sentence or quotation.
    trailing = re.compile(
        r"\s*,\s*" + escaped + r"(?=\s*(?:[.!?;\"”’')\]]|$))", re.IGNORECASE
    )
    stripped = trailing.sub("", stripped)
    return _capitalize_sentence_starts(stripped)


def _capitalize_sentence_starts(text: str) -> str:
    def _upper(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2).upper()

    return re.sub(r"(^|[.!?]\s+|[\"“‘]\s*)([a-z])", _upper, text)


@dataclass
class RejectedNamePurgeReport:
    """What ``purge_rejected_address_name`` deleted, rewrote, and wrote."""

    rejected_name: str
    correct_name: str | None
    deleted: list[dict[str, str]] = field(default_factory=list)
    rewritten: list[dict[str, str]] = field(default_factory=list)
    written: list[dict[str, str]] = field(default_factory=list)
    untouched_mentions: list[dict[str, str]] = field(default_factory=list)

    @property
    def changed_record_count(self) -> int:
        """Count the records the purge deleted, rewrote, and wrote."""
        return len(self.deleted) + len(self.rewritten) + len(self.written)


def user_identity_namespace(user_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the namespace ``learn_information_about_the_user`` writes user facts to."""
    return (assistant_id, user_id, "identity")


def _namespace_label(namespace: tuple) -> str:
    return ".".join(str(part) for part in namespace)


def _rebuild_page_content(page_content: str, old_context: str, new_context: str) -> str:
    """``page_content`` with the stored context replaced by ``new_context``."""
    if old_context and old_context in page_content:
        rebuilt = page_content.replace(old_context, new_context)
    else:
        rebuilt = page_content
    if not new_context:
        # A learning record ends with "\nContext: <context>"; an empty context
        # line carries nothing.
        rebuilt = re.sub(r"\n?Context:\s*$", "", rebuilt).rstrip()
    return rebuilt


async def _purge_namespace(
    store: Any,
    namespace: tuple,
    rejected_name: str,
    report: RejectedNamePurgeReport,
    *,
    dry_run: bool,
) -> None:
    for item in await list_namespace_items(store, namespace):
        document = _item_document(item)
        if document is None:
            continue
        key = getattr(item, "key", None) or document.metadata.get("document_id")
        fact = str(document.metadata.get("fact") or "")
        context = str(document.metadata.get("fact_context") or "")
        page_content = document.page_content or ""
        fact_mentions = text_mentions_name(fact, rejected_name)
        context_mentions = text_mentions_name(context, rejected_name)
        page_mentions = text_mentions_name(page_content, rejected_name)
        if not (fact_mentions or context_mentions or page_mentions):
            continue
        entry = {
            "namespace": _namespace_label(namespace),
            "key": str(key),
            "fact": fact,
            "fact_context": context,
        }
        is_address_record = (
            document.metadata.get("category") == ADDRESS_PREFERENCE_CATEGORY
        )
        fact_is_naming_statement = fact_mentions and bool(
            _NAMING_STATEMENT_PATTERN.search(fact)
        )
        if is_address_record or fact_is_naming_statement:
            report.deleted.append(entry)
            if not dry_run:
                await store.adelete(namespace, key)
            continue
        if fact_mentions:
            # The fact names a person with the rejected name without saying how the
            # user is addressed ("My brother David lives in Ohio."): not the user.
            report.untouched_mentions.append(entry)
            continue
        new_context = strip_vocative_name(context, rejected_name)
        if text_mentions_name(new_context, rejected_name):
            if _NAMING_STATEMENT_PATTERN.search(new_context):
                # The context is the correction itself ("asking not to be called
                # David"); the fact stands without the context.
                new_context = ""
            else:
                report.untouched_mentions.append(entry)
                continue
        new_page_content = _rebuild_page_content(page_content, context, new_context)
        if text_mentions_name(new_page_content, rejected_name):
            new_page_content = strip_vocative_name(new_page_content, rejected_name)
        if text_mentions_name(new_page_content, rejected_name):
            report.untouched_mentions.append(entry)
            continue
        report.rewritten.append({**entry, "new_fact_context": new_context})
        if not dry_run:
            new_metadata = dict(document.metadata)
            new_metadata["fact_context"] = new_context
            rewritten_document = Document(
                page_content=new_page_content, metadata=new_metadata
            )
            await store.aput(
                namespace, key=key, value={"document": rewritten_document.to_json()}
            )


async def _ensure_user_name_fact(
    store: Any,
    user_id: str,
    assistant_id: str,
    correct_name: str,
    report: RejectedNamePurgeReport,
    *,
    dry_run: bool,
) -> None:
    """Keep exactly one "My name is <correct_name>." fact about the user."""
    from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context

    namespace = user_identity_namespace(user_id, assistant_id)
    name_fact = f"My name is {correct_name}."
    normalized_name_fact = name_fact.strip().rstrip(".").lower()
    for item in await list_namespace_items(store, namespace):
        document = _item_document(item)
        if document is None:
            continue
        stored_fact = str(document.metadata.get("fact") or "")
        if stored_fact.strip().rstrip(".").lower() == normalized_name_fact:
            return
    fact_context = "The user stated the user's own name."
    document_id = str(uuid.uuid4())
    report.written.append(
        {
            "namespace": _namespace_label(namespace),
            "key": document_id,
            "fact": name_fact,
            "fact_context": fact_context,
        }
    )
    if dry_run:
        return
    document = Document(
        page_content=wrap_fact_with_context(name_fact, fact_context),
        metadata={
            "user_id": user_id,
            "assistant_id": assistant_id,
            "document_id": document_id,
            "fact_context": fact_context,
            "fact": name_fact,
        },
    )
    await store.aput(namespace, key=document_id, value={"document": document.to_json()})


async def purge_rejected_address_name(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    rejected_name: str,
    correct_name: str | None = None,
    dry_run: bool = False,
) -> RejectedNamePurgeReport:
    """Remove ``rejected_name`` as a name for the user from every prompt-read record.

    With ``dry_run`` the report lists every record that would change and the
    store is not written.
    """
    rejected_name = _clean_name(rejected_name)
    correct_name = _clean_name(correct_name or "") or None
    report = RejectedNamePurgeReport(rejected_name=rejected_name, correct_name=correct_name)
    if store is None or not user_id or not assistant_id or not rejected_name:
        return report
    if correct_name and correct_name.lower() == rejected_name.lower():
        # The same name cannot be both rejected and wanted; change nothing.
        return report
    for namespace in (
        preference_namespace(user_id, assistant_id),
        what_feels_real_namespace(user_id, assistant_id),
        user_identity_namespace(user_id, assistant_id),
    ):
        await _purge_namespace(store, namespace, rejected_name, report, dry_run=dry_run)
    if correct_name:
        address_preference = f"Address the user as {correct_name}."
        if dry_run:
            report.written.append(
                {
                    "namespace": _namespace_label(preference_namespace(user_id, assistant_id)),
                    "key": "(new, unless an equal preference is already stored)",
                    "fact": address_preference,
                    "fact_context": "The user stated the name the user wants to be called.",
                }
            )
        else:
            stored_preference = await store_user_preference(
                store,
                user_id,
                assistant_id,
                preference=address_preference,
                preference_context="The user stated the name the user wants to be called.",
                category=ADDRESS_PREFERENCE_CATEGORY,
                source="dictated",
            )
            if stored_preference is not None:
                report.written.append(
                    {
                        "namespace": _namespace_label(
                            preference_namespace(user_id, assistant_id)
                        ),
                        "key": str(stored_preference.metadata.get("document_id")),
                        "fact": address_preference,
                        "fact_context": stored_preference.metadata.get("fact_context", ""),
                    }
                )
        await _ensure_user_name_fact(
            store, user_id, assistant_id, correct_name, report, dry_run=dry_run
        )
    logger.info(
        "Rejected address name purge for %s/%s: %d deleted, %d rewritten, %d written%s",
        user_id,
        assistant_id,
        len(report.deleted),
        len(report.rewritten),
        len(report.written),
        " (dry run)" if dry_run else "",
    )
    return report


async def purge_names_rejected_in_message(
    store: Any, user_id: str, assistant_id: str, message_text: str
) -> list[RejectedNamePurgeReport]:
    """Purge every name the user rejects in ``message_text``; no model call.

    The purge must not depend on the avatar choosing to call a tool, so the
    user's own words are read directly on every turn.
    """
    rejected_names = extract_rejected_address_names(message_text)
    if not rejected_names:
        return []
    correct_name = extract_preferred_address_name(message_text)
    reports: list[RejectedNamePurgeReport] = []
    for rejected_name in rejected_names:
        reports.append(
            await purge_rejected_address_name(
                store,
                user_id,
                assistant_id,
                rejected_name=rejected_name,
                correct_name=correct_name,
            )
        )
    return reports


async def known_user_address_names(
    store: Any, user_id: str, assistant_id: str
) -> set[str]:
    """Names the user stated for the user: dictated address preferences and name facts."""
    names: set[str] = set()
    for item in await list_namespace_items(
        store, preference_namespace(user_id, assistant_id)
    ):
        document = _item_document(item)
        if (
            document is None
            or document.metadata.get("category") != ADDRESS_PREFERENCE_CATEGORY
            or document.metadata.get("source") != "dictated"
        ):
            continue
        fact = str(document.metadata.get("fact") or "")
        match = re.search(
            r"\b(?:as|called|call\s+(?:me|the\s+user))\s+[\"'“‘]?(" + _NAME_TOKEN + ")", fact
        )
        if match and _is_usable_name(_clean_name(match.group(1))):
            names.add(_clean_name(match.group(1)))
    for item in await list_namespace_items(
        store, user_identity_namespace(user_id, assistant_id)
    ):
        document = _item_document(item)
        if document is None:
            continue
        fact = str(document.metadata.get("fact") or "")
        match = re.search(r"(?i:\bmy\s+name\s+is\s+)[\"'“‘]?(" + _NAME_TOKEN + ")", fact)
        if match and _is_usable_name(_clean_name(match.group(1))):
            names.add(_clean_name(match.group(1)))
    return names


# The name an address preference asks for. A comma or a colon may stand between
# the keyword and the name ("addressed by your first name, Evan").
_ADDRESS_NAME_IN_PREFERENCE_PATTERN = re.compile(
    r"(?i:\b(?:as|called|call\s+(?:me|the\s+user)|named?|name\s+is)\s*[,:]?\s+)[\"'“‘]?("
    + _NAME_TOKEN
    + ")"
)
_USER_NAME_FACT_PATTERN = re.compile(
    r"(?i:\bmy\s+name\s+is\s+)[\"'“‘]?(" + _NAME_TOKEN + ")"
)


def _names_in_entries(entries: list[dict[str, str]], excluded_name: str) -> list[str]:
    """Names of the user found in the facts of ``entries``, without ``excluded_name``."""
    names: list[str] = []
    for entry in entries:
        fact = entry.get("fact") or ""
        for pattern in (_USER_NAME_FACT_PATTERN, _ADDRESS_NAME_IN_PREFERENCE_PATTERN):
            for match in pattern.finditer(fact):
                name = _clean_name(match.group(1))
                if (
                    _is_usable_name(name)
                    and name.lower() != excluded_name.lower()
                    and name.lower() not in {existing.lower() for existing in names}
                ):
                    names.append(name)
    return names


async def relearn_user_address_name(
    store: Any,
    user_id: str,
    assistant_id: str,
    *,
    new_name: str,
    replaces_name_fact: bool,
    dry_run: bool = False,
) -> RejectedNamePurgeReport:
    """Make ``new_name`` the one current name the user is addressed by.

    Every address preference that names a different name is deleted, and one
    address preference for ``new_name`` is written with the current time, so
    the newest record is always the current record. With ``replaces_name_fact``
    (the user said "my name is <new_name>") every stored "My name is <other>."
    fact about the user is deleted and "My name is <new_name>." is kept or
    written. Without ``replaces_name_fact`` (the user said "call me
    <new_name>", which can be a nickname) the stored name fact is left alone.

    The function can be called any number of times with any sequence of names,
    including a name that was replaced or rejected earlier.
    """
    new_name = _clean_name(new_name)
    report = RejectedNamePurgeReport(rejected_name="", correct_name=new_name)
    if store is None or not user_id or not assistant_id or not _is_usable_name(new_name):
        return report
    preferences = preference_namespace(user_id, assistant_id)
    current_address_fact = f"Address the user as {new_name}."
    stale_copies_of_current: list[dict[str, str]] = []
    for item in await list_namespace_items(store, preferences):
        document = _item_document(item)
        if (
            document is None
            or document.metadata.get("category") != ADDRESS_PREFERENCE_CATEGORY
        ):
            continue
        fact = str(document.metadata.get("fact") or "")
        named = [
            _clean_name(match.group(1))
            for match in _ADDRESS_NAME_IN_PREFERENCE_PATTERN.finditer(fact)
        ]
        names_another_name = any(
            _is_usable_name(name) and name.lower() != new_name.lower() for name in named
        )
        is_stale_copy_of_current = (
            fact.strip().rstrip(".").lower()
            == current_address_fact.rstrip(".").lower()
        )
        if not (names_another_name or is_stale_copy_of_current):
            continue
        key = getattr(item, "key", None) or document.metadata.get("document_id")
        entry = {
            "namespace": _namespace_label(preferences),
            "key": str(key),
            "fact": fact,
            "fact_context": str(document.metadata.get("fact_context") or ""),
        }
        if is_stale_copy_of_current and not names_another_name:
            stale_copies_of_current.append(entry)
            continue
        report.deleted.append(entry)
        if not dry_run:
            await store.adelete(preferences, key)
    # The record for ``new_name`` is re-dated only when another name was just
    # replaced, so repeating an unchanged name writes nothing.
    if report.deleted:
        for entry in stale_copies_of_current:
            report.deleted.append(entry)
            if not dry_run:
                await store.adelete(preferences, entry["key"])
    if replaces_name_fact:
        identity = user_identity_namespace(user_id, assistant_id)
        for item in await list_namespace_items(store, identity):
            document = _item_document(item)
            if document is None:
                continue
            fact = str(document.metadata.get("fact") or "")
            match = _USER_NAME_FACT_PATTERN.search(fact)
            if not match or _clean_name(match.group(1)).lower() == new_name.lower():
                continue
            key = getattr(item, "key", None) or document.metadata.get("document_id")
            report.deleted.append(
                {
                    "namespace": _namespace_label(identity),
                    "key": str(key),
                    "fact": fact,
                    "fact_context": str(document.metadata.get("fact_context") or ""),
                }
            )
            if not dry_run:
                await store.adelete(identity, key)
    # An address preference phrased in a way the name pattern does not read can
    # still carry a replaced name. Every address preference that mentions a
    # replaced name is deleted, so the store never holds two names.
    replaced_names = _names_in_entries(report.deleted, new_name)
    if replaced_names:
        deleted_preference_keys = {
            entry["key"]
            for entry in report.deleted
            if entry["namespace"] == _namespace_label(preferences)
        }
        for item in await list_namespace_items(store, preferences):
            document = _item_document(item)
            if (
                document is None
                or document.metadata.get("category") != ADDRESS_PREFERENCE_CATEGORY
            ):
                continue
            key = getattr(item, "key", None) or document.metadata.get("document_id")
            fact = str(document.metadata.get("fact") or "")
            if str(key) in deleted_preference_keys or not any(
                text_mentions_name(fact, replaced_name) for replaced_name in replaced_names
            ):
                continue
            report.deleted.append(
                {
                    "namespace": _namespace_label(preferences),
                    "key": str(key),
                    "fact": fact,
                    "fact_context": str(document.metadata.get("fact_context") or ""),
                }
            )
            if not dry_run:
                await store.adelete(preferences, key)
    if dry_run:
        return report
    stored_preference = await store_user_preference(
        store,
        user_id,
        assistant_id,
        preference=current_address_fact,
        preference_context="The user stated the name the user wants to be called.",
        category=ADDRESS_PREFERENCE_CATEGORY,
        source="dictated",
    )
    if stored_preference is not None:
        report.written.append(
            {
                "namespace": _namespace_label(preferences),
                "key": str(stored_preference.metadata.get("document_id")),
                "fact": current_address_fact,
                "fact_context": stored_preference.metadata.get("fact_context", ""),
            }
        )
    if replaces_name_fact:
        await _ensure_user_name_fact(
            store, user_id, assistant_id, new_name, report, dry_run=False
        )
    return report


async def apply_names_stated_in_message_with_reports(
    store: Any, user_id: str, assistant_id: str, message_text: str
) -> tuple[list[str], str | None, list[RejectedNamePurgeReport]]:
    """Apply every name change the user states in ``message_text``; no model call.

    Returns the rejected names, the newly stated name, and one report per store
    change. A rejected name is purged; a name stated with "my name is" replaces
    the earlier names.
    """
    rejected_names = extract_rejected_address_names(message_text)
    stated_name = extract_stated_user_name(message_text)
    reports: list[RejectedNamePurgeReport] = []
    if rejected_names:
        reports.extend(
            await purge_names_rejected_in_message(store, user_id, assistant_id, message_text)
        )
    if stated_name:
        reports.append(
            await relearn_user_address_name(
                store, user_id, assistant_id, new_name=stated_name, replaces_name_fact=True
            )
        )
    return rejected_names, stated_name, reports


async def apply_names_stated_in_message(
    store: Any, user_id: str, assistant_id: str, message_text: str
) -> tuple[list[str], str | None]:
    """Apply every name change the user states in ``message_text``; no model call.

    Returns the rejected names and the newly stated name.
    """
    rejected_names, stated_name, _reports = await apply_names_stated_in_message_with_reports(
        store, user_id, assistant_id, message_text
    )
    return rejected_names, stated_name


REJECTED_NAME_LEARNED_FACT = "The user rejected a name."


def learned_fact_entries_for_name_reports(
    rejected_names: list[str],
    stated_name: str | None,
    reports: list[RejectedNamePurgeReport],
) -> list[dict[str, str]]:
    """Describe the name changes in ``reports`` as learned-fact badge entries.

    Each entry carries ``fact``, ``kind``, ``status`` and, for a replaced name,
    ``previous_fact``. A rejected name is never written into an entry.
    """
    entries: list[dict[str, str]] = []
    rejected_lowercase = {name.lower() for name in rejected_names}
    if any(
        report.rejected_name and (report.deleted or report.rewritten) for report in reports
    ):
        entries.append(
            {"fact": REJECTED_NAME_LEARNED_FACT, "kind": "preference", "status": "removed"}
        )
    if not stated_name:
        return entries
    address_fact = f"Address the user as {stated_name}."
    name_fact = f"My name is {stated_name}."
    wrote_stated_name = any(
        written.get("fact") in (address_fact, name_fact)
        for report in reports
        for written in report.written
    )
    replaced_names = [
        name
        for report in reports
        if not report.rejected_name
        for name in _names_in_entries(report.deleted, stated_name)
        if name.lower() not in rejected_lowercase
    ]
    entry = {"fact": address_fact, "kind": "preference", "status": "known"}
    if replaced_names:
        entry["status"] = "updated"
        entry["previous_fact"] = f"Address the user as {replaced_names[0]}."
    elif wrote_stated_name:
        entry["status"] = "learned"
    entries.append(entry)
    return entries
