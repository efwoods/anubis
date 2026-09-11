"""Reading browsing into facts about the person, and traits of the person.

One structured-output call per pass produces both, because both come off the
same evidence and a second call would double the cost of a feature that runs
while the person browses.

Where the findings land, and why each place:

- **Facts** go to the identity namespace, ``(assistant_id, user_id,
  "identity")``, in exactly the shape ``learn_information_about_the_user``
  writes — that namespace is loaded in full into every reply, so a fact
  learned from browsing is knowledge the avatar simply has.
- **Traits** fold into the accumulated psychological profile as the graded
  dimension ``browsing_behaviour``, so a second pass reinforces or moves a
  score instead of stacking another restatement of the same observation. The
  profile is rendered into its own section of the system prompt every turn.
- **Findings** are also written as analysis-namespace Documents, so they are
  retrievable by similarity alongside every other analyzed trait.
- **A report** is saved per pass when reports are switched on, which is the
  thing the person reads rather than the thing the avatar carries.

Nothing here decides WHEN to run; ``sweeper`` and the avatar's own tool do
that. This module is given visits and returns what was learned.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.browsing.digest import hosts_of, render_digest, summarize_visits
from src.anubis.utils.prompts.browsing_analysis_prompt import (
    BROWSING_ANALYSIS_INPUT_TEMPLATE,
    BROWSING_ANALYSIS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# The dimension name the browsing traits accumulate under in the psychological
# profile. Spelled once here; the profile keys on the string.
BROWSING_DIMENSION = "browsing_behaviour"

# The kind a saved browsing report is filed under in the reports list.
BROWSING_REPORT_KIND = "browsing_insights"

# A fact the model is less sure of than this is not written to the identity
# namespace: a wrong fact about the person is worse than a missing one,
# because the avatar will state a wrong fact as its own knowledge.
MINIMUM_FACT_CONFIDENCE = 0.55


class BrowsingFact(BaseModel):
    """One durable fact about the person, read off the browsing record."""

    fact: str = Field(
        description=(
            "A durable statement about the person, in the third person, that "
            "would still be true next month."
        )
    )
    fact_context: str = Field(
        default="",
        description="The circumstance the fact belongs to, in one sentence.",
    )
    evidence: str = Field(
        default="",
        description="The searches, page titles, or websites that support the fact.",
    )
    confidence: float = Field(
        default=0.0, description="Zero to one: how strongly the record supports the fact."
    )


class BrowsingTrait(BaseModel):
    """One scored trait of how the person thinks and works."""

    trait: str = Field(description="The trait name, from the list in the instructions.")
    score: float = Field(default=0.0, description="Zero to one.")
    confidence: float = Field(default=0.0, description="Zero to one.")
    first_person_statement: str = Field(
        default="", description="The trait as the person would say the trait of themselves."
    )
    supporting_evidence: str = Field(
        default="", description="What in the record supports the score."
    )


class BrowsingInsights(BaseModel):
    """Everything one pass over a browsing record produced."""

    facts: list[BrowsingFact] = Field(default_factory=list)
    traits: list[BrowsingTrait] = Field(default_factory=list)
    summary_markdown: str = Field(
        default="", description="Three to six sentences addressed to the person."
    )


def _machine_description(visits: Iterable[Mapping[str, Any]]) -> str:
    """Which machines and browsers a batch of visits came from."""
    machines = sorted(
        {
            str(visit.get("device_label") or "").strip()
            for visit in visits
            if str(visit.get("device_label") or "").strip()
        }
    )
    browsers = sorted(
        {
            str(visit.get("browser_name") or "").strip()
            for visit in visits
            if str(visit.get("browser_name") or "").strip()
        }
    )
    machine_text = ", ".join(machines) if machines else "the person's computer"
    browser_text = ", ".join(browsers) if browsers else "their web browser"
    return f"{machine_text} ({browser_text})"


async def analyze_visits(
    visits: list[dict[str, Any]],
    *,
    target_name: str,
    known_hosts: Iterable[str] = (),
    max_digest_characters: int = 24000,
) -> BrowsingInsights | None:
    """Read one batch of visits into facts and traits, or ``None`` when it says nothing."""
    from src.anubis.utils.model import init_model

    digest = render_digest(
        visits, known_hosts=known_hosts, max_characters=max_digest_characters
    )
    if not digest:
        return None
    summary = summarize_visits(visits)
    try:
        system_prompt = BROWSING_ANALYSIS_SYSTEM_PROMPT.format(
            target_name=target_name or "the person"
        )
    except (KeyError, IndexError):
        system_prompt = BROWSING_ANALYSIS_SYSTEM_PROMPT
    human_text = BROWSING_ANALYSIS_INPUT_TEMPLATE.format(
        period_start=summary.get("period_start") or "the start of the period",
        period_end=summary.get("period_end") or "now",
        machine_description=_machine_description(visits),
        digest=digest,
    )
    model = init_model(response_format=BrowsingInsights)
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    )
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, BrowsingInsights):
        return response
    # A provider that answered with a mapping rather than the model instance.
    if isinstance(response, Mapping):
        try:
            return BrowsingInsights.model_validate(dict(response))
        except Exception as validation_error:  # noqa: BLE001 - a pass must not crash
            logger.warning("Browsing insights could not be read: %s", validation_error)
    return None


# ---------------------------------------------------------------------------
# Where the findings land
# ---------------------------------------------------------------------------


def trait_finding(insights: BrowsingInsights) -> dict[str, Any]:
    """Return the browsing traits in the shape the psychological profile merges."""
    traits: dict[str, Any] = {}
    for trait in insights.traits:
        name = (trait.trait or "").strip()
        if not name:
            continue
        try:
            score = float(trait.score or 0.0)
            confidence = float(trait.confidence or 0.0)
        except (TypeError, ValueError):
            continue
        traits[name] = {
            "score": max(0.0, min(1.0, score)),
            "confidence": max(0.0, min(1.0, confidence)),
            "statement": (trait.first_person_statement or "").strip(),
            "evidence": (trait.supporting_evidence or "").strip(),
        }
    return {"dimension": BROWSING_DIMENSION, "kind": "graded", "traits": traits}


def insight_documents(
    insights: BrowsingInsights,
    *,
    user_id: str,
    assistant_id: str,
    device_label: str,
    period_end: str,
) -> list[Document]:
    """Return the findings as analysis-namespace Documents, retrievable by similarity."""
    stamp = datetime.now(tz=UTC).isoformat()
    # One filename per machine per pass, so a re-run of the same pass replaces
    # its own rows instead of appending a second copy of the same findings.
    namespace_filename = f"browsing-insights-{device_label or 'machine'}-{period_end}"
    documents: list[Document] = []
    for trait in insights.traits:
        statement = (trait.first_person_statement or "").strip()
        if not statement or float(trait.score or 0.0) <= 0.0:
            continue
        documents.append(
            Document(
                page_content=statement,
                metadata={
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "feature": BROWSING_DIMENSION,
                    "trait": (trait.trait or "").strip(),
                    "score": float(trait.score or 0.0),
                    "confidence": float(trait.confidence or 0.0),
                    "evidence": (trait.supporting_evidence or "").strip(),
                    "source": "browsing_history",
                    "device_label": device_label,
                    "namespace": "analysis",
                    "namespace_filename": namespace_filename,
                    "vectorstore_acceptable": True,
                    "adapter_acceptable": False,
                    "analysis_acceptable": False,
                    "synthetic": True,
                    "created_at": stamp,
                    "document_id": str(uuid.uuid4()),
                    "model_inference_type": "structured_output",
                },
            )
        )
    return documents


def _fact_is_worth_keeping(fact: BrowsingFact) -> bool:
    """Whether one produced fact is confident enough to become knowledge."""
    text = (fact.fact or "").strip()
    if len(text) < 8:
        return False
    try:
        return float(fact.confidence or 0.0) >= MINIMUM_FACT_CONFIDENCE
    except (TypeError, ValueError):
        return False


async def store_facts(
    store: Any,
    insights: BrowsingInsights,
    *,
    user_id: str,
    assistant_id: str,
    device_label: str,
) -> list[str]:
    """Write the confident facts to the identity namespace; return what was written.

    A fact already known is skipped rather than written twice: the identity
    namespace is loaded whole into every reply, so a duplicate costs prompt
    space on every turn for the life of the avatar.
    """
    from src.anubis.utils.tools.identity.identity_tools import (
        _put_fact_document_and_confirm,
        _store_items_contain_fact,
        wrap_fact_with_context,
    )

    if store is None:
        return []
    namespace = (assistant_id, user_id, "identity")
    written: list[str] = []
    for fact in insights.facts:
        if not _fact_is_worth_keeping(fact):
            continue
        text = (fact.fact or "").strip()
        context_text = (fact.fact_context or "").strip() or (
            f"Read from {device_label or 'the owner'}'s web browsing."
        )
        try:
            existing = await store.asearch(namespace, query=text)
        except Exception as search_error:  # noqa: BLE001 - a search outage must not lose the pass
            logger.warning("Could not check for an existing fact: %s", search_error)
            existing = []
        if _store_items_contain_fact(existing, text):
            continue
        document_id = str(uuid.uuid4())
        document = Document(
            page_content=wrap_fact_with_context(text, context_text),
            metadata={
                "user_id": user_id,
                "assistant_id": assistant_id,
                "document_id": document_id,
                "fact_context": context_text,
                "fact": text,
                "source": "browsing_history",
                "evidence": (fact.evidence or "").strip(),
                "confidence": float(fact.confidence or 0.0),
                "device_label": device_label,
            },
        )
        if await _put_fact_document_and_confirm(
            store, namespace, document_id, {"document": document.to_json()}
        ):
            written.append(text)
    return written


async def store_trait_findings(
    store: Any,
    insights: BrowsingInsights,
    *,
    user_id: str,
    assistant_id: str,
    device_label: str,
    period_end: str,
) -> int:
    """Fold the traits into the profile and index them; return how many were kept."""
    from src.anubis.utils.psycho.profile import (
        merge_findings_into_profile,
        read_profile_record,
        write_profile_record,
    )

    finding = trait_finding(insights)
    if not finding["traits"]:
        return 0
    profile = await read_profile_record(store, user_id, assistant_id)
    merged = merge_findings_into_profile(profile, [finding])
    await write_profile_record(store, user_id, assistant_id, merged)
    documents = insight_documents(
        insights,
        user_id=user_id,
        assistant_id=assistant_id,
        device_label=device_label,
        period_end=period_end,
    )
    if documents:
        try:
            from src.subgraphs.vector_store_graph.utils.helper_functions import (
                batch_index_documents_vectorstore,
            )

            await batch_index_documents_vectorstore(
                store, user_id, assistant_id, documents
            )
        except Exception as index_error:  # noqa: BLE001 - the profile already landed
            logger.warning("Browsing findings could not be indexed: %s", index_error)
    return len(finding["traits"])


async def save_browsing_report(
    insights: BrowsingInsights,
    *,
    user_id: str,
    assistant_id: str,
    visits: list[dict[str, Any]],
    facts_written: list[str],
    device_label: str,
) -> str | None:
    """Save the pass as a report the person can read; return the report's id."""
    from src.anubis.utils.analytics.reports import get_report_repository

    repository = get_report_repository()
    if repository is None:
        return None
    summary = summarize_visits(visits)
    lines = [(insights.summary_markdown or "").strip()]
    if facts_written:
        lines.append("")
        lines.append("**What your avatar learned about you**")
        lines.extend(f"- {fact}" for fact in facts_written)
    if insights.traits:
        lines.append("")
        lines.append("**How you worked**")
        lines.extend(
            f"- {(trait.first_person_statement or trait.trait).strip()} "
            f"({trait.trait}: {float(trait.score or 0.0):.2f})"
            for trait in insights.traits
            if float(trait.score or 0.0) > 0.0
        )
    top_sites = ", ".join(host for host, _count in summary["top_hosts"][:8] if host)
    if top_sites:
        lines.append("")
        lines.append(f"**Where you spent the period** {top_sites}")
    try:
        row = await repository.create(
            {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "kind": BROWSING_REPORT_KIND,
                "title": (
                    f"Browsing insights — {summary['visit_count']} visits across "
                    f"{summary['distinct_hosts']} sites"
                ),
                "summary_markdown": "\n".join(line for line in lines if line is not None),
                "charts": [],
                "sources": [f"browsing history ({device_label})" if device_label else "browsing history"],
                "period_start": summary.get("period_start") or None,
                "period_end": summary.get("period_end") or None,
                "thread_id": None,
            }
        )
    except Exception as save_error:  # noqa: BLE001 - the findings already landed
        logger.warning("The browsing report could not be saved: %s", save_error)
        return None
    return str((row or {}).get("report_id") or (row or {}).get("id") or "") or None


async def apply_insights(
    store: Any,
    insights: BrowsingInsights,
    *,
    user_id: str,
    assistant_id: str,
    visits: list[dict[str, Any]],
    device_label: str,
    write_report: bool = True,
) -> dict[str, Any]:
    """Put one pass's findings everywhere they belong; return what happened."""
    summary = summarize_visits(visits)
    period_end = str(summary.get("period_end") or datetime.now(tz=UTC).isoformat())
    facts_written = await store_facts(
        store,
        insights,
        user_id=user_id,
        assistant_id=assistant_id,
        device_label=device_label,
    )
    traits_kept = await store_trait_findings(
        store,
        insights,
        user_id=user_id,
        assistant_id=assistant_id,
        device_label=device_label,
        period_end=period_end,
    )
    report_id = None
    if write_report:
        report_id = await save_browsing_report(
            insights,
            user_id=user_id,
            assistant_id=assistant_id,
            visits=visits,
            facts_written=facts_written,
            device_label=device_label,
        )
    return {
        "facts_written": facts_written,
        "fact_count": len(facts_written),
        "trait_count": traits_kept,
        "report_id": report_id,
        "visit_count": summary["visit_count"],
        "distinct_hosts": summary["distinct_hosts"],
        "hosts": hosts_of(visits),
        "summary_markdown": (insights.summary_markdown or "").strip(),
    }
