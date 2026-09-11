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
    read_page,
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

# How much of one page the researcher agent is shown per search result. The full
# page is kept on the SearchResult for fact extraction; this is only what goes
# into the model's context while it decides whether to search again, and a whole
# page per result would fill that context after two searches.
_SOURCE_EXCERPT_FOR_MODEL = 2_000

# Ceiling on the compression step's input, so a topic that gathered a great many
# pages cannot overflow the model that cleans them up.
_COMPRESSION_INPUT_LIMIT = 60_000

# Hard ceiling on model calls in one topic's loop, counting reflections as well
# as searches. The search budget in the prompt is the real limit; this is the
# backstop for a model that reflects forever without ever finishing.
_MAX_AGENT_TURNS = 14

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

RESEARCH_AGENT_SYSTEM_PROMPT = """
<Role>
You are a research assistant gathering information about one topic concerning one named subject. Your research runs as a tool-calling loop: you choose what to search for, you read what comes back, and you decide when you have enough.
</Role>

<Available_Tools>
You have two tools:
1. search_the_web: run one web search and read the pages it returns.
2. record_reflection: write down what you have found, what is still missing, and what you intend to do next.

Use record_reflection after every single search_the_web call. The reflection is not shown to anyone and is not part of the research; it is the deliberate pause that makes the next decision a considered one rather than a reflex.
</Available_Tools>

<Instructions>
Work the way a researcher with a limited budget works.
1. Read the assignment carefully. Decide what would actually answer it about this specific subject.
2. Start broad. The first search should be a wide query naming the subject, not a narrow one.
3. After each search, call record_reflection: what did this search establish, what is still missing, and is that gap worth another search?
4. Narrow as you go. Later searches should aim at the specific gaps the reflections named, never repeat a query already run.
5. Stop as soon as the assignment is answered. Do not keep searching for completeness that nobody asked for.
</Instructions>

<Hard_Limits>
Search budget, which exists to stop one topic spending the whole research run:
- A straightforward assignment: two or three search_the_web calls at most.
- A difficult assignment: up to five search_the_web calls, and no more.
- Always stop after five search_the_web calls, whether or not the assignment is answered. Report what you did find.

Stop immediately when any of these is true:
- The assignment is answered.
- Three or more sources cover the assignment.
- The last two searches returned much the same information as each other.
</Hard_Limits>

<Guarding_Against_The_Wrong_Person>
Many people share a name. A source that names the subject but describes a different person — a different occupation, a different country, a different century — is not about this subject, and a fact taken from it would be written into the wrong person's identity. Say so in a reflection when you see it, and search in a way that separates the two.
</Guarding_Against_The_Wrong_Person>

<Finishing>
When you are done, reply with a plain message and no tool call. That message ends the research on this topic.
</Finishing>
"""


COMPRESSION_SYSTEM_PROMPT = """
<Role>
You are cleaning up the findings a researcher gathered about one topic concerning one named subject, by web search. The researcher has finished; your job is to make what they found legible without losing any of it.
</Role>

<Task>
Rewrite the gathered information in a cleaner form. Repeat the relevant statements verbatim rather than summarizing them. The purpose of this step is only to remove duplication and material that has nothing to do with the assignment. When three sources state the same thing, say that three sources state it and give the statement once.

Losing information here is the failure to avoid. A later step reads only what you write, so anything you leave out is gone.
</Task>

<What_To_Exclude>
Exclude the researcher's own reflections, which were recorded with record_reflection. Those are the researcher's internal reasoning about what to do next; they contain no information about the subject and must not reach the findings.

Include everything that came back from search_the_web.
</What_To_Exclude>

<Citations>
Every statement keeps the address of the page it came from, written inline after the statement. End with a Sources section listing every page the researcher read, so no source is lost.
</Citations>

<Output_Format>
**Queries run**
**Findings**
**Sources**
</Output_Format>
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
    sources: list[SearchResult], *, concurrency: int, collect_images: bool = False
) -> list[SearchResult]:
    """Fill in the page text of every source that arrived without content.

    With ``collect_images`` the same fetch also harvests the pictures each page
    declares as representing itself, so the asset acquisition gets portrait
    candidates from the pages the research was already reading. It is off unless
    the run may actually acquire a portrait: a source that arrived with its
    content already filled in (Tavily returns page text inline) would otherwise
    be fetched a second time purely for pictures nothing is going to look at.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _read(source: SearchResult) -> SearchResult:
        wants_images = collect_images and not source.images
        if source.content and not wants_images:
            return source
        async with semaphore:
            try:
                page = await read_page(source.url, collect_images=wants_images)
                if not source.content:
                    source.content = page.text
                if wants_images:
                    source.images = page.images
            except Exception as read_error:  # noqa: BLE001 - one unreadable page is not fatal
                logger.info("Could not read %s: %s", source.url, read_error)
                if not source.content:
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


# ── the researcher agent: a tool-calling loop with a reflection tool ─────────
#
# This is the shape the LangChain deep-research course's researcher sub-agent
# uses, and the reason it is a loop rather than a fixed sequence: the model,
# not the pipeline, decides whether what it has read answers the assignment.
# A fixed "search, reflect, search once more" spends a second search it does not
# need on an easy topic and stops one short on a hard one. The two tools are the
# search and a reflection the model records before deciding what to do next.
#
# The budget is enforced twice over — stated in the prompt so the model plans
# against it, and capped in code below so a model that ignores it still cannot
# spend the whole run on one topic.


def _reflection_tool_result(reflection: str) -> str:
    """Record one reflection and hand it straight back to the model.

    The tool does nothing but acknowledge. Its whole value is that calling it
    forces the model to state its findings, its gaps, and its intent before it
    chooses the next action, instead of firing off another search by reflex.
    The reflections are deliberately kept out of the compressed findings: they
    are reasoning about the research, not information about the subject.
    """
    return f"Reflection recorded: {reflection}"


def _format_sources_for_model(sources: list[SearchResult], *, limit: int) -> str:
    """Render what one search returned as the tool result the model reads."""
    if not sources:
        return "No results."
    blocks = []
    for source in sources:
        body = (source.content or source.snippet or "").strip()
        blocks.append(
            f'<source url="{source.url}" title="{source.title}">\n'
            f"{body[:limit]}\n"
            f"</source>"
        )
    return "\n\n".join(blocks)


async def run_topic_research_agent(
    subject: str,
    topic: ResearchTopic,
    *,
    context: Any,
    max_sources: int,
    max_searches: int,
    concurrency: int,
    collect_images: bool,
    emit: EventSink,
    is_cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Research one topic in a tool-calling loop; return the messages and sources.

    Returns ``{"messages", "sources", "queries"}``. The sources are kept as
    ``SearchResult`` objects rather than only as text, because everything
    downstream needs more than the prose: verification needs each fact's source
    URL, the media hand-off needs the page addresses, and the portrait
    acquisition needs the pictures those pages declared.
    """
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool

    from src.anubis.utils.model import init_model

    collected: dict[str, SearchResult] = {}
    queries_run: list[str] = []
    searches_used = 0

    @tool
    async def search_the_web(query: str) -> str:
        """Search the web for one query and read the pages that come back.

        Args:
            query: What to search for. Name the subject explicitly; a bare topic
                word will return pages about someone else with the same name.
        """
        nonlocal searches_used
        searches_used += 1
        queries_run.append(query)
        emit(
            {
                "type": "research_progress",
                "stage": "searching",
                "topic": topic.topic,
                "queries": [query],
            }
        )
        found = await gather_sources(
            [query],
            per_query=max(3, max_sources // 2),
            max_sources=max_sources,
            context=context,
            seen_urls=set(collected),
        )
        found = await read_sources(
            found, concurrency=concurrency, collect_images=collect_images
        )
        for source in found:
            collected.setdefault(source.url, source)
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
        return _format_sources_for_model(found, limit=_SOURCE_EXCERPT_FOR_MODEL)

    @tool
    async def record_reflection(reflection: str) -> str:
        """Record what you have found, what is missing, and what you will do next.

        Args:
            reflection: Your assessment of the research so far and your intent.
        """
        emit(
            {
                "type": "research_progress",
                "stage": "reflecting",
                "topic": topic.topic,
                "gap_summary": reflection,
            }
        )
        return _reflection_tool_result(reflection)

    tools_by_name = {
        "search_the_web": search_the_web,
        "record_reflection": record_reflection,
    }
    try:
        model = init_model(tools=list(tools_by_name.values()), tool_choice="auto")
    except Exception as model_error:  # noqa: BLE001 - fall back to fixed queries
        # A provider this deployment cannot reach must not cost the topic. The
        # fallback below still searches, using queries the pipeline writes.
        logger.warning(
            "The researcher for %r could not start: %s", topic.topic, model_error
        )
        model = None

    messages: list[Any] = [
        SystemMessage(content=RESEARCH_AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"<SUBJECT>\n{subject}\n</SUBJECT>\n"
                f"<ASSIGNMENT>\n{topic.topic}: {topic.assignment}\n</ASSIGNMENT>"
            )
        ),
    ]

    # One turn per model call. The cap counts turns, not searches, so a model
    # that only reflects still terminates.
    for _turn in range(_MAX_AGENT_TURNS if model is not None else 0):
        if is_cancelled():
            break
        try:
            response = await model.ainvoke(messages)
        except Exception as agent_error:  # noqa: BLE001 - keep whatever was gathered
            logger.warning(
                "The researcher for %r stopped early: %s", topic.topic, agent_error
            )
            break
        if isinstance(response, tuple):
            response = response[0]
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # A reply with no tool call is the model saying it is finished —
            # the same routing decision should_continue makes in the course.
            break
        for tool_call in tool_calls:
            name = tool_call.get("name")
            chosen = tools_by_name.get(name)
            if chosen is None:
                observation = f"There is no tool called {name!r}."
            elif name == "search_the_web" and searches_used >= max_searches:
                # The budget the prompt states, enforced where a model cannot
                # talk its way past it.
                observation = (
                    "The search budget for this topic is spent. "
                    "Report what you have found."
                )
            else:
                try:
                    observation = await chosen.ainvoke(tool_call.get("args") or {})
                except Exception as tool_error:  # noqa: BLE001 - one failed tool is not fatal
                    logger.info("Tool %s failed: %s", name, tool_error)
                    observation = f"That tool call failed: {tool_error}"
            messages.append(
                ToolMessage(
                    content=str(observation),
                    name=str(name),
                    tool_call_id=tool_call.get("id") or str(uuid.uuid4()),
                )
            )
    else:
        if model is not None:
            logger.info(
                "The researcher for %r hit the turn cap; compressing what it has.",
                topic.topic,
            )

    if not collected and not is_cancelled():
        # The model never searched, or every search failed. Fall back to the
        # queries the pipeline would have written itself, so a topic is never
        # silently dropped.
        fallback_queries = await write_topic_queries(
            subject, topic, queries_run, max_queries=2
        )
        found = await gather_sources(
            fallback_queries,
            per_query=max(3, max_sources // 2),
            max_sources=max_sources,
            context=context,
        )
        found = await read_sources(
            found, concurrency=concurrency, collect_images=collect_images
        )
        for source in found:
            collected.setdefault(source.url, source)
        queries_run.extend(fallback_queries)

    return {
        "messages": messages,
        "sources": list(collected.values()),
        "queries": queries_run,
    }


async def compress_topic_research(
    subject: str, topic: ResearchTopic, messages: list[Any]
) -> str:
    """Rewrite what the researcher gathered, verbatim but cleaned and cited.

    The course's compression step, and it earns its place for the same reason
    there: the loop leaves behind a transcript in which the same claim appears
    in three tool results and the researcher's own reasoning is mixed in with
    the sources. What comes out is the findings alone, deduplicated, each
    statement still carrying the page it came from.
    """
    from langchain_core.messages import ToolMessage

    from src.anubis.utils.model import init_model

    # Only what the search tool returned. The reflections are the researcher
    # reasoning about its own progress and say nothing about the subject.
    gathered = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "search_the_web"
    ]
    if not gathered:
        return ""
    transcript = "\n\n".join(str(message.content) for message in gathered)
    human_text = (
        f"<SUBJECT>\n{subject}\n</SUBJECT>\n"
        f"<ASSIGNMENT>\n{topic.topic}: {topic.assignment}\n</ASSIGNMENT>\n"
        f"<GATHERED>\n{transcript[:_COMPRESSION_INPUT_LIMIT]}\n</GATHERED>"
    )
    try:
        model = init_model()
        response = await model.ainvoke(
            [
                SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                HumanMessage(content=human_text),
            ]
        )
        if isinstance(response, tuple):
            response = response[0]
        return str(getattr(response, "content", "") or "")
    except Exception as compression_error:  # noqa: BLE001 - the sources still stand
        logger.warning(
            "Compressing the findings for %r failed: %s", topic.topic, compression_error
        )
        return ""


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
    collect_images: bool = False,
) -> dict[str, Any]:
    """Research one topic and return the facts it establishes.

    Three steps, the researcher sub-agent of the LangChain deep-research course
    applied to one topic about one person:

    1. **The agent loop** — the model searches, records a reflection, and decides
       for itself whether to search again or stop. ``max_searches`` is its
       budget.
    2. **Compression** — the loop's transcript is rewritten into clean, cited
       findings with the researcher's own reflections stripped out.
    3. **Extraction** — facts are pulled per source, so every fact keeps the
       address of the page that supports it. That attribution is what the
       verifier clusters on and what the media hand-off follows, so extraction
       reads the pages themselves rather than the compressed prose.

    ``max_queries`` is the agent's search budget; ``follow_up_rounds`` is no
    longer a round count, and is added to the budget so a deployment that raised
    it still gets a longer leash.
    """
    max_searches = max(1, max_queries + max(0, follow_up_rounds))

    researched = await run_topic_research_agent(
        subject,
        topic,
        context=context,
        max_sources=max_sources,
        max_searches=max_searches,
        concurrency=concurrency,
        collect_images=collect_images,
        emit=emit,
        is_cancelled=is_cancelled,
    )
    topic_sources: list[SearchResult] = researched["sources"]
    queries_run: list[str] = researched["queries"]

    if is_cancelled():
        return {
            "topic": topic.topic,
            "sources": topic_sources,
            "facts": [],
            "queries": queries_run,
            "compressed_research": "",
        }

    emit({"type": "research_progress", "stage": "compressing", "topic": topic.topic})
    compressed = await compress_topic_research(subject, topic, researched["messages"])

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
        "compressed_research": compressed,
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


def verified_source_urls(to_apply: list[dict[str, Any]], *, limit: int) -> list[str]:
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
    ranked = sorted(support_count.items(), key=lambda pair: (-pair[1], pair[0]))
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
    bootstrap: Any | None = None,
) -> dict[str, Any]:
    """Scope, research, verify, and apply. Emits progress; returns the job summary.

    ``bootstrap`` is an ``asset_bootstrap.BootstrapGateway`` when this run may
    also acquire the reference image and reference audio the avatar is missing.
    It is passed in rather than imported because the operations it wraps live in
    the API layer, which imports this module. Passing ``None`` runs exactly the
    fact research this function has always run.
    """
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
                collect_images=bootstrap is not None,
            )
            for topic in brief.topics
        )
    )
    if is_cancelled():
        return {"cancelled": True}

    sources_by_url: dict[str, SearchResult] = {}
    web_facts: list[dict[str, Any]] = []
    queries_run: list[str] = []
    # The cleaned, cited findings per topic. Facts are what the avatar learns;
    # this is the readable account of what the research actually read, kept on
    # the job summary so the run can be inspected after the fact.
    compressed_by_topic: dict[str, str] = {}
    for result in topic_results:
        web_facts.extend(result["facts"])
        queries_run.extend(result["queries"])
        if result.get("compressed_research"):
            compressed_by_topic[result["topic"]] = result["compressed_research"]
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

    # Acquiring the portrait and the voice needs the pages this research just
    # read, and nothing that comes after. Start it here, alongside verification,
    # so the video transcription — much the slowest thing either half does —
    # begins minutes earlier than it would if acquisition waited its turn.
    bootstrap_task = None
    if bootstrap is not None:
        from src.anubis.utils.research.asset_bootstrap import run_asset_bootstrap

        bootstrap_task = asyncio.create_task(
            run_asset_bootstrap(
                store,
                context,
                creator_id=creator_id,
                assistant_id=assistant_id,
                subject_name=subject_name,
                subject_summary=brief.subject_summary,
                sources=list(sources_by_url.values()),
                gateway=bootstrap,
                emit=emit,
                is_cancelled=is_cancelled,
            )
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
        if bootstrap_task is not None:
            bootstrap_task.cancel()
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

    # Wait for the acquisition that was started alongside verification. It is
    # best-effort by construction: a failure or a timeout costs the portrait and
    # the voice, never the facts this run already applied.
    bootstrap_summary: dict[str, Any] = {}
    if bootstrap_task is not None:
        try:
            bootstrap_summary = await bootstrap_task
        except asyncio.CancelledError:
            bootstrap_summary = {"cancelled": True}
        except Exception as bootstrap_error:  # noqa: BLE001 - the research still succeeded
            logger.exception("Asset acquisition failed for %s", assistant_id)
            bootstrap_summary = {"error": str(bootstrap_error)}
    # A source the acquisition already ingested must not be handed to the media
    # pipeline a second time: the chosen video would be downloaded, diarized and
    # transcribed twice, which is the single most expensive thing this pipeline
    # does.
    bootstrap_media_urls = list(bootstrap_summary.get("media_urls") or [])
    if bootstrap_media_urls:
        already_ingested = {
            url.strip().rstrip("/").lower() for url in bootstrap_media_urls
        }
        media_source_urls = [
            url
            for url in media_source_urls
            if url.strip().rstrip("/").lower() not in already_ingested
        ]

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
        "bootstrap": bootstrap_summary,
        "bootstrap_media_urls": bootstrap_media_urls,
        "topics": [topic.topic for topic in brief.topics],
        "compressed_research": compressed_by_topic,
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
