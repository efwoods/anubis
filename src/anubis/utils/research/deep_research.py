"""Deep research with web-based fact verification for one avatar.

The pipeline follows the LangChain "Deep Research with LangGraph" course
structure — **scope**, **research**, **write** — with the written output being
verified first-person facts rather than a prose report:

1. **Scope.** ``build_research_brief`` reads the avatar's name, description, an
   optional hint from the creator, and every fact the avatar already holds
   about the avatar's own identity, then writes a research brief: a subject
   summary, the open questions worth answering, and a handful of research
   topics to delegate.
2. **Research.** Every topic is handed to a researcher
   (``research_one_topic``) that writes the topic's search queries, searches
   every configured provider, reads each source page, reflects on what the
   sources left unanswered, runs one more round of follow-up queries when the
   topic is still thin, and compresses the sources into atomic first-person
   facts, each carrying the URL of the source that stated the fact. Topics run
   concurrently, the way the course's supervisor delegates to parallel
   researchers.
3. **Verify.** Facts that state the same claim are clustered by embedding
   similarity — and the avatar's existing identity facts are clustered in
   alongside them — so every cluster is judged across sources: ``consistent``
   when two or more sources agree, ``inconsistent`` when sources (or a source
   and a fact the avatar already holds) contradict each other, ``unverified``
   when only one source states the claim.
4. **Apply.** Facts the sources do not contradict are written into the
   avatar's identity immediately and appear in what the avatar has learned.
   Only the inconsistent clusters are held back as proposals for the creator
   to accept, edit, or ignore (``list_proposals`` / ``resolve_proposals``),
   because those are the ones where a person has to decide which version is
   true.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.research.web_search import (
    SearchResult,
    read_page_text,
    search_web,
)

logger = logging.getLogger(__name__)

RESEARCH_PROPOSAL_KIND = "research_proposal"
PROPOSAL_STATUS_PENDING = "pending"
VERIFICATION_STATUSES = ("consistent", "inconsistent", "unverified")

# The identity metadata mark that says a fact came from deep research. The
# ``/avatar_identity_facts`` endpoint reads this mark to group researched facts
# separately from the facts extracted out of uploaded media.
RESEARCH_FACT_SOURCE = "deep_research"

# Facts whose embeddings score at least this similar are the same claim.
_CLUSTER_SIMILARITY_THRESHOLD = 0.72
_SOURCE_TEXT_CHARACTER_LIMIT = 12_000
# How many facts the avatar already holds are carried into the brief and into
# the clustering. Enough to catch a contradiction, bounded so a well-fed avatar
# does not push the brief past the model's context window.
_EXISTING_FACT_LIMIT = 200

# A fact the avatar already holds is a "source" for clustering purposes, so a
# single web page that contradicts a held fact is still a two-source cluster
# and reaches the verifier. This prefix names that pseudo-source.
IDENTITY_SOURCE_PREFIX = "identity://"

FACT_ORIGIN_WEB = "web"
FACT_ORIGIN_IDENTITY = "identity"


def research_proposal_namespace(
    creator_id: str, assistant_id: str
) -> tuple[str, str, str]:
    """Return the store namespace holding one avatar's pending research proposals."""
    return (creator_id, assistant_id, RESEARCH_PROPOSAL_KIND)


def identity_namespace(creator_id: str, assistant_id: str) -> tuple[str, str, str]:
    """Return the namespace ``load_consciousness`` reads the avatar's own facts from."""
    return (creator_id, assistant_id, "identity")


# ── structured models ───────────────────────────────────────────────────────


class ResearchTopic(BaseModel):
    """One delegated research assignment."""

    topic: str = Field(
        description="A short name for the topic, for example 'early life and education'."
    )
    assignment: str = Field(
        description="What the researcher must find out about the subject under this topic, in one or two sentences."
    )


class ResearchBrief(BaseModel):
    """The scoping step's output: what the subject is and what to research."""

    subject_summary: str = Field(
        description="One or two sentences on who or what the subject is, as far as the inputs say."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions the existing facts leave unanswered, contradict each other on, or state without a source.",
    )
    topics: list[ResearchTopic] = Field(
        description="Two to six research topics that together cover the subject's identity, history, notable facts, relationships, and public statements."
    )


class TopicQueries(BaseModel):
    """The web queries one researcher runs for one topic."""

    queries: list[str] = Field(
        description="Two to four web search queries for this topic. Include the subject's exact name in every query, with disambiguating words when the name is common."
    )


class ResearchReflection(BaseModel):
    """The researcher's judgement on whether the topic is answered yet."""

    topic_is_answered: bool = Field(
        description="Whether the sources read so far answer the topic's assignment well enough to stop searching."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Up to three further web search queries that would close the remaining gaps. Empty when the topic is answered.",
    )
    gap_summary: str = Field(
        default="", description="One sentence naming what is still missing."
    )


class ExtractedFact(BaseModel):
    """One atomic claim about the subject, as the subject would state the claim."""

    fact: str = Field(
        description="One atomic fact about the subject, written in the first person as the subject would state the fact, preserving names, dates, places, and numbers exactly."
    )
    fact_context: str = Field(
        description="A concise summary of the surrounding passage the fact came from."
    )


class ExtractedFacts(BaseModel):
    """Every fact one source states about the subject."""

    subject_present: bool = Field(
        description="Whether the source is actually about the subject (not a namesake or an unrelated page)."
    )
    facts: list[ExtractedFact] = Field(default_factory=list)


class FactVerification(BaseModel):
    """The verifier's judgement on one cluster of statements of the same claim."""

    status: Literal["consistent", "inconsistent", "unverified"] = Field(
        description="consistent when two or more sources agree; inconsistent when sources contradict each other; unverified when only one source states the fact."
    )
    proposed_fact: str = Field(
        description="The single best first-person statement of the fact, resolving wording differences."
    )
    conflicting_statements: list[str] = Field(
        default_factory=list,
        description="The contradictory versions, verbatim, when the status is inconsistent.",
    )
    reasoning: str = Field(
        default="", description="One sentence on why the status was chosen."
    )


SCOPING_SYSTEM_PROMPT = """
<ROLE>
You plan web research about one subject — a person, a place, a monument, a historical site, an object — so an avatar of that subject holds verified public facts.
</ROLE>

<INSTRUCTIONS>
From the SUBJECT inputs, write a one or two sentence subject summary, the open questions, and two to six research topics.
The KNOWN_FACTS are what the avatar already holds about the subject. Read the known facts for claims that disagree with each other, claims that carry no source, and claims that a public source could confirm or correct; every such claim becomes an open question, and the topics must cover the open questions first.
Then cover the rest of the subject: who or what the subject is; history and dates; notable facts and achievements; relationships and context; public statements.
Each topic names what the researcher must find out, in one or two sentences. Never invent facts about the subject here; only plan the research.
</INSTRUCTIONS>
"""

QUERY_SYSTEM_PROMPT = """
<ROLE>
You write the web search queries for one research topic about one subject.
</ROLE>

<INSTRUCTIONS>
Write two to four distinct queries that together answer the topic's assignment. Include the subject's exact name in every query, and add disambiguating words (a place, a year, a role) when the name is common. Do not repeat a query that has already been run; the queries already run are listed as QUERIES_ALREADY_RUN.
</INSTRUCTIONS>
"""

REFLECTION_SYSTEM_PROMPT = """
<ROLE>
You judge whether the sources read so far answer one research topic about one subject.
</ROLE>

<INSTRUCTIONS>
Read the topic's assignment and the titles and excerpts of the sources gathered. Decide whether the assignment is answered well enough to stop searching.
When the assignment is not answered, write up to three follow-up web search queries that would close the gap, and name the gap in one sentence. Never repeat a query listed under QUERIES_ALREADY_RUN.
</INSTRUCTIONS>
"""

EXTRACTION_SYSTEM_PROMPT = """
<ROLE>
You extract facts about one subject from a web page for an avatar of that subject.
</ROLE>

<INSTRUCTIONS>
The SUBJECT names who or what the facts must be about. Read the SOURCE and decide first whether the source is actually about the subject; a namesake or an unrelated page is not.
Extract every distinct atomic fact about the subject. Write each fact in the FIRST PERSON as the subject would state the fact ("I was born in ...", "I was unveiled in 1921 ..."), preserving names, dates, places, and numbers exactly as the source states them. Give each fact a concise context summary of the passage the fact came from.
Never invent a fact. Never include facts about other people or things except as they relate to the subject.
</INSTRUCTIONS>

<SUBJECT>
{subject}
</SUBJECT>
"""

VERIFICATION_SYSTEM_PROMPT = """
<ROLE>
You verify one fact about a subject against several sources.
</ROLE>

<INSTRUCTIONS>
The CANDIDATE_STATEMENTS are versions of the same claim, each with the source that stated the version. A statement whose source begins with "identity://" is a fact the avatar already holds, not a web source. Decide:
- consistent: two or more distinct sources state the same fact (wording may differ).
- inconsistent: the sources contradict each other on the substance (a different date, place, number, name), including a web source contradicting a fact the avatar already holds.
- unverified: only one source states the fact.
Write the single best first-person statement of the fact as proposed_fact. When inconsistent, list the contradictory versions verbatim.
</INSTRUCTIONS>
"""


async def invoke_structured(
    response_format: type[BaseModel], system_prompt: str, human_text: str
):
    """One structured-output model call. Isolated so tests can replace this."""
    from src.anubis.utils.model import init_model

    model = init_model(response_format=response_format)
    response = await model.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
    )
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, response_format):
        return response
    return response_format.model_validate(response)


async def score_similarity(query: str, texts: list[str]) -> list[float]:
    """Cosine similarity through the store's embedding model. Isolated for tests."""
    from src.anubis.utils.runtime_handles import async_score_query_against_texts

    return await async_score_query_against_texts(query, texts)


# ── scoping ─────────────────────────────────────────────────────────────────

EventSink = Callable[[dict[str, Any]], None]


def _subject_text(
    name: str,
    description: str | None,
    research_hint: str | None,
) -> str:
    """Return the subject block every prompt in the pipeline is given."""
    parts = [f"Name: {name}"]
    if description:
        parts.append(f"Description: {description}")
    if research_hint:
        parts.append(f"Hint from the creator: {research_hint}")
    return "\n".join(parts)


async def load_existing_identity_facts(
    store: Any, creator_id: str, assistant_id: str, *, limit: int = _EXISTING_FACT_LIMIT
) -> list[dict[str, Any]]:
    """Read the facts the avatar already holds about the avatar's own identity.

    Read from the same ``(creator_id, assistant_id, "identity")`` prefix
    ``load_consciousness`` reads every turn, so the research compares against
    exactly what the avatar believes today. Each entry carries a pseudo source
    URL (``identity://<key>``) so the verifier can weigh a held fact against a
    web page in the same cluster.
    """
    from src.anubis.utils.tools.identity.identity_tools import _extract_clean_fact

    try:
        items = await store.asearch(
            identity_namespace(creator_id, assistant_id), limit=limit
        )
    except Exception:  # noqa: BLE001 - research still runs without the held facts
        logger.debug(
            "Could not read the avatar's existing identity facts", exc_info=True
        )
        return []
    facts: list[dict[str, Any]] = []
    for item in items or []:
        try:
            fact_text = _extract_clean_fact(item)
        except Exception:  # noqa: BLE001 - one unreadable row is not fatal
            continue
        if not fact_text or not fact_text.strip():
            continue
        key = getattr(item, "key", None) or ""
        facts.append(
            {
                "fact": fact_text.strip(),
                "fact_context": "",
                "source_url": f"{IDENTITY_SOURCE_PREFIX}{key}",
                "source_title": "A fact the avatar already holds",
                "origin": FACT_ORIGIN_IDENTITY,
                "identity_key": key,
            }
        )
    return facts


async def build_research_brief(
    subject: str, existing_facts: list[dict[str, Any]], *, max_topics: int
) -> ResearchBrief:
    """Scope the research: the subject summary, the open questions, the topics."""
    known_facts_block = "\n".join(
        f"- {fact['fact']}" for fact in existing_facts[:_EXISTING_FACT_LIMIT]
    )
    human_text = (
        f"<SUBJECT>\n{subject}\n</SUBJECT>\n"
        f"<KNOWN_FACTS>\n{known_facts_block or '(the avatar holds no facts yet)'}\n</KNOWN_FACTS>"
    )
    brief = await invoke_structured(ResearchBrief, SCOPING_SYSTEM_PROMPT, human_text)
    brief.topics = [topic for topic in brief.topics if (topic.topic or "").strip()][
        : max(1, max_topics)
    ]
    if not brief.topics:
        brief.topics = [
            ResearchTopic(
                topic="identity and history",
                assignment="Find out who or what the subject is, the subject's history, and the notable public facts about the subject.",
            )
        ]
    return brief


# ── research: one researcher per topic ──────────────────────────────────────


async def write_topic_queries(
    subject: str,
    topic: ResearchTopic,
    queries_already_run: list[str],
    *,
    max_queries: int,
) -> list[str]:
    """Write the queries one researcher runs for one topic."""
    human_text = (
        f"<SUBJECT>\n{subject}\n</SUBJECT>\n"
        f"<TOPIC>\n{topic.topic}: {topic.assignment}\n</TOPIC>\n"
        f"<QUERIES_ALREADY_RUN>\n{chr(10).join(queries_already_run) or '(none)'}\n</QUERIES_ALREADY_RUN>"
    )
    try:
        written = await invoke_structured(TopicQueries, QUERY_SYSTEM_PROMPT, human_text)
        queries = [query.strip() for query in written.queries if query.strip()]
    except Exception as query_error:  # noqa: BLE001 - the topic name is a usable query
        logger.warning("Writing queries for %r failed: %s", topic.topic, query_error)
        queries = []
    if not queries:
        queries = [f"{subject.splitlines()[0].removeprefix('Name: ')} {topic.topic}"]
    return queries[: max(1, max_queries)]


async def gather_sources(
    queries: list[str],
    *,
    per_query: int,
    max_sources: int,
    context: Any,
    seen_urls: set[str] | None = None,
) -> list[SearchResult]:
    """Search every query, merge the results, and rank the most-hit sources first."""
    outcomes = await asyncio.gather(
        *(search_web(query, limit=per_query, context=context) for query in queries),
        return_exceptions=True,
    )
    merged: dict[str, SearchResult] = {}
    for query, outcome in zip(queries, outcomes):
        if isinstance(outcome, Exception):
            logger.warning("Search failed for %r: %s", query, outcome)
            continue
        for result in outcome:
            if seen_urls is not None and result.url in seen_urls:
                continue
            existing = merged.get(result.url)
            if existing is None:
                merged[result.url] = result
            else:
                existing.queries.extend(result.queries)
                if not existing.content and result.content:
                    existing.content = result.content
    # Sources hit by more queries first: those are the most likely to be about
    # the subject rather than a namesake.
    ranked = sorted(merged.values(), key=lambda result: -len(set(result.queries)))
    return ranked[:max_sources]


async def read_sources(
    sources: list[SearchResult], *, concurrency: int
) -> list[SearchResult]:
    """Fill in the page text of every source that arrived without content."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _read(source: SearchResult) -> SearchResult:
        if source.content:
            return source
        async with semaphore:
            try:
                source.content = await read_page_text(source.url)
            except Exception as read_error:  # noqa: BLE001 - one unreadable page is not fatal
                logger.info("Could not read %s: %s", source.url, read_error)
                source.content = source.snippet
        return source

    return list(await asyncio.gather(*(_read(source) for source in sources)))


async def reflect_on_topic(
    subject: str,
    topic: ResearchTopic,
    sources: list[SearchResult],
    queries_already_run: list[str],
) -> ResearchReflection:
    """Decide whether the topic still needs another round of searching."""
    source_block = "\n".join(
        f"- [{source.url}] {source.title}: {(source.content or source.snippet or '')[:600]}"
        for source in sources
    )
    human_text = (
        f"<SUBJECT>\n{subject}\n</SUBJECT>\n"
        f"<TOPIC>\n{topic.topic}: {topic.assignment}\n</TOPIC>\n"
        f"<SOURCES>\n{source_block or '(no sources were found)'}\n</SOURCES>\n"
        f"<QUERIES_ALREADY_RUN>\n{chr(10).join(queries_already_run)}\n</QUERIES_ALREADY_RUN>"
    )
    try:
        return await invoke_structured(
            ResearchReflection, REFLECTION_SYSTEM_PROMPT, human_text
        )
    except Exception as reflection_error:  # noqa: BLE001 - stop searching on failure
        logger.warning("Reflection on %r failed: %s", topic.topic, reflection_error)
        return ResearchReflection(topic_is_answered=True, follow_up_queries=[])


async def extract_facts(
    sources: list[SearchResult], *, subject: str, concurrency: int, topic: str = ""
) -> list[dict[str, Any]]:
    """Every extracted fact with the URL of the source that stated the fact."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _extract(source: SearchResult) -> list[dict[str, Any]]:
        text = (source.content or source.snippet or "").strip()
        if not text:
            return []
        async with semaphore:
            try:
                extracted = await invoke_structured(
                    ExtractedFacts,
                    EXTRACTION_SYSTEM_PROMPT.format(subject=subject),
                    f'<SOURCE url="{source.url}" title="{source.title}">\n'
                    f"{text[:_SOURCE_TEXT_CHARACTER_LIMIT]}\n</SOURCE>",
                )
            except Exception as extraction_error:  # noqa: BLE001
                logger.warning(
                    "Fact extraction failed for %s: %s", source.url, extraction_error
                )
                return []
        if not extracted.subject_present:
            return []
        return [
            {
                "fact": item.fact.strip(),
                "fact_context": item.fact_context.strip(),
                "source_url": source.url,
                "source_title": source.title,
                "origin": FACT_ORIGIN_WEB,
                "topic": topic,
            }
            for item in extracted.facts
            if item.fact.strip()
        ]

    nested = await asyncio.gather(*(_extract(source) for source in sources))
    return [fact for facts in nested for fact in facts]


async def research_one_topic(
    subject: str,
    topic: ResearchTopic,
    *,
    context: Any,
    max_queries: int,
    max_sources: int,
    concurrency: int,
    follow_up_rounds: int,
    emit: EventSink,
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Search, read, reflect, and compress one topic into facts.

    The researcher runs one search round, judges whether the topic's assignment
    is answered, and runs up to ``follow_up_rounds`` further rounds on the gaps
    the reflection names — the search / reflect / search loop the course's
    researcher sub-agent runs, bounded so one topic cannot spend the whole job.
    """
    queries_run: list[str] = []
    seen_urls: set[str] = set()
    topic_sources: list[SearchResult] = []

    queries = await write_topic_queries(
        subject, topic, queries_run, max_queries=max_queries
    )
    for round_index in range(max(1, 1 + follow_up_rounds)):
        if is_cancelled() or not queries:
            break
        queries_run.extend(queries)
        emit(
            {
                "type": "research_progress",
                "stage": "searching",
                "topic": topic.topic,
                "queries": queries,
            }
        )
        found = await gather_sources(
            queries,
            per_query=max(3, max_sources // 2),
            max_sources=max_sources,
            context=context,
            seen_urls=seen_urls,
        )
        found = await read_sources(found, concurrency=concurrency)
        for source in found:
            seen_urls.add(source.url)
        topic_sources.extend(found)
        emit(
            {
                "type": "research_progress",
                "stage": "searched",
                "topic": topic.topic,
                "sources": [
                    {"url": source.url, "title": source.title} for source in found
                ],
            }
        )
        if round_index >= follow_up_rounds or is_cancelled():
            break
        reflection = await reflect_on_topic(subject, topic, topic_sources, queries_run)
        if reflection.topic_is_answered or not reflection.follow_up_queries:
            break
        emit(
            {
                "type": "research_progress",
                "stage": "reflecting",
                "topic": topic.topic,
                "gap_summary": reflection.gap_summary,
            }
        )
        queries = [
            query.strip()
            for query in reflection.follow_up_queries
            if query.strip() and query.strip() not in queries_run
        ][: max(1, max_queries)]

    if is_cancelled():
        return {
            "topic": topic.topic,
            "sources": topic_sources,
            "facts": [],
            "queries": queries_run,
        }

    emit({"type": "research_progress", "stage": "extracting", "topic": topic.topic})
    facts = await extract_facts(
        topic_sources, subject=subject, concurrency=concurrency, topic=topic.topic
    )
    emit(
        {
            "type": "research_progress",
            "stage": "extracted",
            "topic": topic.topic,
            "facts": len(facts),
        }
    )
    return {
        "topic": topic.topic,
        "sources": topic_sources,
        "facts": facts,
        "queries": queries_run,
    }


# ── verification ────────────────────────────────────────────────────────────


async def cluster_facts(
    facts: list[dict[str, Any]], *, threshold: float = _CLUSTER_SIMILARITY_THRESHOLD
) -> list[list[dict[str, Any]]]:
    """Greedy clustering: each fact joins the first cluster whose lead statement is similar enough."""
    clusters: list[list[dict[str, Any]]] = []
    for fact in facts:
        if not clusters:
            clusters.append([fact])
            continue
        leads = [cluster[0]["fact"] for cluster in clusters]
        try:
            scores = await score_similarity(fact["fact"], leads)
        except Exception as scoring_error:  # noqa: BLE001 - without scores every fact is its own cluster
            logger.warning("Fact clustering scoring failed: %s", scoring_error)
            scores = []
        best_index = (
            max(range(len(scores)), key=lambda index: scores[index]) if scores else None
        )
        if best_index is not None and scores[best_index] >= threshold:
            clusters[best_index].append(fact)
        else:
            clusters.append([fact])
    return clusters


def _existing_fact_in(cluster: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the fact the avatar already holds in this cluster, if the cluster has one."""
    for fact in cluster:
        if fact.get("origin") == FACT_ORIGIN_IDENTITY:
            return fact
    return None


async def verify_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge one cluster of statements of the same claim across the sources.

    A cluster holding only the fact the avatar already holds is reported as
    ``already_held``: nothing on the web spoke to the claim, so there is
    nothing to add and nothing to review.
    """
    source_urls = sorted({fact["source_url"] for fact in cluster})
    existing_fact = _existing_fact_in(cluster)
    web_facts = [fact for fact in cluster if fact.get("origin") != FACT_ORIGIN_IDENTITY]
    if existing_fact is not None and not web_facts:
        return {
            "status": "already_held",
            "proposed_fact": existing_fact["fact"],
            "fact_context": existing_fact.get("fact_context") or "",
            "supporting_source_urls": [],
            "conflicting_statements": [],
            "reasoning": "The avatar already holds this fact and no source spoke to the fact.",
            "statements": [existing_fact["fact"]],
            "existing_fact": existing_fact["fact"],
            "topics": [],
        }
    topics = sorted({fact.get("topic") or "" for fact in web_facts} - {""})
    if len(source_urls) == 1:
        lead = web_facts[0]
        return {
            "status": "unverified",
            "proposed_fact": lead["fact"],
            "fact_context": lead.get("fact_context") or "",
            "supporting_source_urls": source_urls,
            "conflicting_statements": [],
            "reasoning": "Stated by a single source.",
            "statements": [fact["fact"] for fact in cluster],
            "existing_fact": None,
            "topics": topics,
        }
    human_text = (
        "<CANDIDATE_STATEMENTS>\n"
        + "\n".join(f"- [{fact['source_url']}] {fact['fact']}" for fact in cluster)
        + "\n</CANDIDATE_STATEMENTS>"
    )
    try:
        verification = await invoke_structured(
            FactVerification, VERIFICATION_SYSTEM_PROMPT, human_text
        )
    except Exception as verification_error:  # noqa: BLE001
        logger.warning("Fact verification failed: %s", verification_error)
        verification = FactVerification(
            status="unverified",
            proposed_fact=web_facts[0]["fact"],
            reasoning=f"Verification failed: {verification_error}",
        )
    return {
        "status": verification.status,
        "proposed_fact": verification.proposed_fact.strip() or web_facts[0]["fact"],
        "fact_context": (web_facts[0].get("fact_context") or ""),
        "supporting_source_urls": [
            url for url in source_urls if not url.startswith(IDENTITY_SOURCE_PREFIX)
        ],
        "conflicting_statements": list(verification.conflicting_statements),
        "reasoning": verification.reasoning,
        "statements": [fact["fact"] for fact in cluster],
        "existing_fact": existing_fact["fact"] if existing_fact else None,
        "topics": topics,
    }


def partition_verified_facts(
    verified: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the verified clusters into what is applied now and what needs review.

    Only a contradiction needs a person: an ``inconsistent`` cluster is held
    back as a proposal. Everything else the web stated is written into the
    avatar's identity immediately, minus the clusters that only restate a fact
    the avatar already holds — those are already learned.
    """
    to_apply: list[dict[str, Any]] = []
    to_review: list[dict[str, Any]] = []
    for entry in verified:
        if entry["status"] == "inconsistent":
            to_review.append(entry)
        elif entry["status"] == "already_held":
            continue
        elif entry.get("existing_fact"):
            # The web agrees with what the avatar already holds; adding the
            # same claim a second time would only duplicate the identity.
            continue
        else:
            to_apply.append(entry)
    return to_apply, to_review


def verified_source_urls(
    to_apply: list[dict[str, Any]], *, limit: int
) -> list[str]:
    """Rank the source URLs behind facts the research verified, best-supported first.

    Only sources that actually corroborated a fact are handed to the media
    pipeline. A page the search returned and nothing was verified from is not
    worth transcribing, and transcription is the expensive step — limit
    is the ceiling on how many sources one research run may send through, so a
    single run can never start an unbounded batch of transcriptions.

    Sources are ordered by how many verified facts each one supported, so when
    the cap bites it keeps the sources the research leaned on most.
    """
    if limit <= 0:
        return []
    support_count: dict[str, int] = {}
    for entry in to_apply:
        for url in entry.get("supporting_source_urls") or []:
            cleaned = str(url or "").strip()
            if not cleaned:
                continue
            support_count[cleaned] = support_count.get(cleaned, 0) + 1
    ranked = sorted(
        support_count.items(), key=lambda pair: (-pair[1], pair[0])
    )
    return [url for url, _ in ranked[:limit]]


# ── writing facts and proposals into the store ──────────────────────────────


MEDIA_FACT_SOURCE = "media_verification"


def build_identity_document(
    verified: dict[str, Any],
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    source: str = RESEARCH_FACT_SOURCE,
) -> Document:
    """One verified fact, shaped like every other fact the avatar has learned.

    ``source`` records where the fact came from — the web, or the media the
    avatar was taught from — so a later reader can tell a researched claim from
    one drawn out of the owner's own uploads.
    """
    from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context

    document_id = str(uuid.uuid4())
    return Document(
        page_content=wrap_fact_with_context(
            verified["proposed_fact"], verified.get("fact_context") or ""
        ),
        metadata={
            "user_id": creator_id,
            "assistant_id": assistant_id,
            "document_id": document_id,
            "fact": verified["proposed_fact"],
            "fact_context": verified.get("fact_context") or "",
            "source": source,
            "source_urls": list(verified.get("supporting_source_urls") or []),
            "verification_status": verified["status"],
            "subject_name": subject_name,
            "created_at": datetime.now(tz=UTC).isoformat(),
        },
    )


def build_proposal_document(
    verified: dict[str, Any], *, creator_id: str, assistant_id: str, subject_name: str
) -> Document:
    """One contradiction, held for the creator to accept, edit, or ignore."""
    from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context

    fact_id = str(uuid.uuid4())
    return Document(
        page_content=wrap_fact_with_context(
            verified["proposed_fact"], verified.get("fact_context") or ""
        ),
        metadata={
            "document_id": fact_id,
            "fact_id": fact_id,
            "fact": verified["proposed_fact"],
            "fact_context": verified.get("fact_context") or "",
            "kind": RESEARCH_PROPOSAL_KIND,
            "status": PROPOSAL_STATUS_PENDING,
            "verification_status": verified["status"],
            "supporting_source_urls": list(
                verified.get("supporting_source_urls") or []
            ),
            "conflicting_statements": list(
                verified.get("conflicting_statements") or []
            ),
            "statements": list(verified.get("statements") or []),
            "existing_fact": verified.get("existing_fact"),
            "reasoning": verified.get("reasoning") or "",
            "subject_name": subject_name,
            "user_id": creator_id,
            "assistant_id": assistant_id,
            "proposed_at": datetime.now(tz=UTC).isoformat(),
        },
    )


async def _store_document(
    store: Any, namespace: tuple[str, ...], key: str, document: Document
) -> None:
    await store.aput(namespace, key=key, value={"document": document.to_json()})


# ── the whole pipeline ──────────────────────────────────────────────────────


async def run_deep_research(
    store: Any,
    context: Any,
    *,
    creator_id: str,
    assistant_id: str,
    subject_name: str,
    subject_description: str | None,
    research_hint: str | None,
    emit: EventSink,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Scope, research, verify, and apply. Emits progress; returns the job summary."""
    max_queries = int(getattr(context, "deep_research_max_queries", 4) or 4)
    max_sources = int(getattr(context, "deep_research_max_sources", 12) or 12)
    max_topics = int(getattr(context, "deep_research_max_topics", 4) or 4)
    concurrency = int(getattr(context, "deep_research_concurrency", 4) or 4)
    follow_up_rounds = int(getattr(context, "deep_research_follow_up_rounds", 1) or 0)
    subject = _subject_text(subject_name, subject_description, research_hint)

    emit({"type": "research_progress", "stage": "scoping"})
    existing_facts = await load_existing_identity_facts(store, creator_id, assistant_id)
    brief = await build_research_brief(subject, existing_facts, max_topics=max_topics)
    emit(
        {
            "type": "research_progress",
            "stage": "scoped",
            "subject_summary": brief.subject_summary,
            "open_questions": brief.open_questions,
            "topics": [topic.topic for topic in brief.topics],
            "known_facts": len(existing_facts),
        }
    )
    if is_cancelled():
        return {"cancelled": True}

    emit({"type": "research_progress", "stage": "researching"})
    topic_results = await asyncio.gather(
        *(
            research_one_topic(
                subject,
                topic,
                context=context,
                max_queries=max_queries,
                max_sources=max_sources,
                concurrency=concurrency,
                follow_up_rounds=follow_up_rounds,
                emit=emit,
                is_cancelled=is_cancelled,
            )
            for topic in brief.topics
        )
    )
    if is_cancelled():
        return {"cancelled": True}

    sources_by_url: dict[str, SearchResult] = {}
    web_facts: list[dict[str, Any]] = []
    queries_run: list[str] = []
    for result in topic_results:
        web_facts.extend(result["facts"])
        queries_run.extend(result["queries"])
        for source in result["sources"]:
            sources_by_url.setdefault(source.url, source)

    from src.anubis.utils.tokenizer import count_tokens

    tokens_read = sum(
        count_tokens(source.content or "") for source in sources_by_url.values()
    )
    emit(
        {
            "type": "research_progress",
            "stage": "read",
            "sources": len(sources_by_url),
            "facts": len(web_facts),
            "tokens_read": tokens_read,
        }
    )

    emit({"type": "research_progress", "stage": "verifying"})
    clusters = await cluster_facts(web_facts + existing_facts)
    verified = list(
        await asyncio.gather(*(verify_cluster(cluster) for cluster in clusters))
    )
    counts = {status: 0 for status in VERIFICATION_STATUSES}
    for entry in verified:
        if entry["status"] in counts:
            counts[entry["status"]] += 1
    emit({"type": "research_progress", "stage": "verified", **counts})
    if is_cancelled():
        return {"cancelled": True}

    to_apply, to_review = partition_verified_facts(verified)

    emit({"type": "research_progress", "stage": "applying", "facts": len(to_apply)})
    applied_fact_ids: list[str] = []
    for entry in to_apply:
        document = build_identity_document(
            entry,
            creator_id=creator_id,
            assistant_id=assistant_id,
            subject_name=subject_name,
        )
        await _store_document(
            store,
            identity_namespace(creator_id, assistant_id),
            document.metadata["document_id"],
            document,
        )
        applied_fact_ids.append(document.metadata["document_id"])

    proposal_ids: list[str] = []
    for entry in to_review:
        document = build_proposal_document(
            entry,
            creator_id=creator_id,
            assistant_id=assistant_id,
            subject_name=subject_name,
        )
        await _store_document(
            store,
            research_proposal_namespace(creator_id, assistant_id),
            document.metadata["fact_id"],
            document,
        )
        proposal_ids.append(document.metadata["fact_id"])
    emit(
        {
            "type": "research_progress",
            "stage": "applied",
            "applied": len(applied_fact_ids),
            "proposals": len(proposal_ids),
        }
    )
    # The media behind what the research actually verified. The caller feeds
    # these through the same pipeline an uploaded link takes, so a video the
    # research leaned on is transcribed and learned from rather than reduced to
    # the few sentences the page happened to expose.
    media_source_urls = verified_source_urls(
        to_apply,
        limit=int(getattr(context, "deep_research_max_media_items", 0) or 0),
    )
    emit(
        {
            "type": "research_progress",
            "stage": "verified_media",
            "media_sources": len(media_source_urls),
        }
    )
    return {
        "subject_summary": brief.subject_summary,
        "open_questions": brief.open_questions,
        "media_source_urls": media_source_urls,
        "topics": [topic.topic for topic in brief.topics],
        "queries": queries_run,
        "sources": [
            {"url": source.url, "title": source.title, "provider": source.provider}
            for source in sources_by_url.values()
        ],
        "known_facts": len(existing_facts),
        "facts_extracted": len(web_facts),
        "applied": len(applied_fact_ids),
        "proposals": len(proposal_ids),
        "verification_counts": counts,
        "tokens_read": tokens_read,
    }


# ── proposals: list and resolve ─────────────────────────────────────────────


def _item_document(item: Any) -> Document | None:
    value = getattr(item, "value", None) or {}
    document_json = value.get("document") if isinstance(value, dict) else None
    kwargs = (document_json or {}).get("kwargs") or {}
    if not isinstance(kwargs, dict) or not kwargs.get("page_content"):
        return None
    return Document(
        page_content=kwargs["page_content"], metadata=dict(kwargs.get("metadata") or {})
    )


async def list_proposals(
    store: Any, creator_id: str, assistant_id: str
) -> list[dict[str, Any]]:
    """Every inconsistency still waiting for the creator's decision."""
    try:
        items = await store.asearch(
            research_proposal_namespace(creator_id, assistant_id), limit=1000
        )
    except Exception:  # noqa: BLE001
        return []
    proposals = []
    for item in items or []:
        document = _item_document(item)
        if (
            document is None
            or document.metadata.get("status") != PROPOSAL_STATUS_PENDING
        ):
            continue
        metadata = document.metadata
        proposals.append(
            {
                "fact_id": metadata.get("fact_id"),
                "fact": metadata.get("fact"),
                "fact_context": metadata.get("fact_context"),
                "verification_status": metadata.get("verification_status"),
                "supporting_source_urls": metadata.get("supporting_source_urls") or [],
                "conflicting_statements": metadata.get("conflicting_statements") or [],
                "statements": metadata.get("statements") or [],
                "existing_fact": metadata.get("existing_fact"),
                "reasoning": metadata.get("reasoning"),
                "proposed_at": metadata.get("proposed_at"),
            }
        )
    order = {"inconsistent": 0, "unverified": 1, "consistent": 2}
    proposals.sort(
        key=lambda proposal: (
            order.get(proposal["verification_status"], 3),
            proposal["proposed_at"] or "",
        )
    )
    return proposals


class ProposalResolution(BaseModel):
    """One decision the creator made about one proposed fact."""

    fact_id: str
    action: Literal["accept", "edit", "ignore"]
    corrected_text: str | None = None


async def resolve_proposals(
    store: Any,
    creator_id: str,
    assistant_id: str,
    resolutions: list[ProposalResolution],
) -> dict[str, Any]:
    """Apply the creator's decisions; returns counts and the accepted source URLs."""
    proposal_namespace = research_proposal_namespace(creator_id, assistant_id)
    accepted = edited = ignored = missing = 0
    accepted_source_urls: set[str] = set()
    for resolution in resolutions:
        item = await store.aget(proposal_namespace, key=resolution.fact_id)
        document = _item_document(item) if item is not None else None
        if document is None:
            missing += 1
            continue
        if resolution.action == "ignore":
            await store.adelete(proposal_namespace, resolution.fact_id)
            ignored += 1
            continue
        fact_text = (
            (resolution.corrected_text or "").strip()
            if resolution.action == "edit"
            else document.metadata.get("fact", "")
        )
        if not fact_text:
            missing += 1
            continue
        identity_document = build_identity_document(
            {
                "proposed_fact": fact_text,
                "fact_context": document.metadata.get("fact_context") or "",
                "supporting_source_urls": document.metadata.get(
                    "supporting_source_urls"
                )
                or [],
                "status": document.metadata.get("verification_status")
                or "inconsistent",
            },
            creator_id=creator_id,
            assistant_id=assistant_id,
            subject_name=document.metadata.get("subject_name") or "",
        )
        await _store_document(
            store,
            identity_namespace(creator_id, assistant_id),
            identity_document.metadata["document_id"],
            identity_document,
        )
        await store.adelete(proposal_namespace, resolution.fact_id)
        accepted_source_urls.update(
            document.metadata.get("supporting_source_urls") or []
        )
        if resolution.action == "edit":
            edited += 1
        else:
            accepted += 1
    return {
        "accepted": accepted,
        "edited": edited,
        "ignored": ignored,
        "missing": missing,
        "accepted_source_urls": sorted(accepted_source_urls),
    }


__all__ = [
    "PROPOSAL_STATUS_PENDING",
    "RESEARCH_FACT_SOURCE",
    "RESEARCH_PROPOSAL_KIND",
    "ProposalResolution",
    "ResearchBrief",
    "ResearchTopic",
    "build_identity_document",
    "build_proposal_document",
    "build_research_brief",
    "cluster_facts",
    "extract_facts",
    "list_proposals",
    "load_existing_identity_facts",
    "partition_verified_facts",
    "research_one_topic",
    "research_proposal_namespace",
    "resolve_proposals",
    "run_deep_research",
    "verify_cluster",
]
