"""Unit tests for deep research with web-based fact verification.

Pinned down:

- **Scoping reads what the avatar already holds**, so the research brief can
  aim at the claims the uploads left uncertain.
- **A researcher searches, reflects, and searches again** only while the topic
  is unanswered, and never repeats a query already run.
- **Verification is what decides who reviews what**: a claim two sources agree
  on, or a claim only one source states, is applied to the avatar's identity
  immediately; only a contradiction — between two sources, or between a source
  and a fact the avatar already holds — waits for the creator.
- **Resolving a proposal writes the creator's wording**, not the researched
  wording, and drops what the creator ignored.
- **Emotion media generation is gated by a configurable minimum tier**, in the
  API and in the media graph alike.
"""

import pytest
from langgraph.store.memory import InMemoryStore

from src.anubis.utils.billing.tiers import (
    SubscriptionTier,
    minimum_tier_from_value,
    tier_meets_minimum,
)
from src.anubis.utils.research import deep_research
from src.anubis.utils.research.deep_research import (
    FACT_ORIGIN_IDENTITY,
    FACT_ORIGIN_WEB,
    IDENTITY_SOURCE_PREFIX,
    ProposalResolution,
    ResearchBrief,
    ResearchTopic,
    build_identity_document,
    cluster_facts,
    identity_namespace,
    list_proposals,
    load_existing_identity_facts,
    partition_verified_facts,
    research_proposal_namespace,
    resolve_proposals,
    run_deep_research,
    verify_cluster,
)
from src.anubis.utils.research.web_search import SearchResult
from src.anubis.utils.tools.identity.identity_tools import wrap_fact_with_context
from src.subgraphs.process_media_graph.utils.nodes import (
    emotion_media_tier_allows_generation,
)

CREATOR_ID = "auth0|creator"
ASSISTANT_ID = "assistant-1"


def _context(**overrides):
    from types import SimpleNamespace

    values = dict(
        deep_research_enabled="true",
        deep_research_max_topics=2,
        deep_research_max_queries=2,
        deep_research_max_sources=4,
        deep_research_follow_up_rounds=1,
        deep_research_concurrency=2,
        tavily_api_key=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _web_fact(fact, url, context_summary="context", topic="history"):
    return {
        "fact": fact,
        "fact_context": context_summary,
        "source_url": url,
        "source_title": url,
        "origin": FACT_ORIGIN_WEB,
        "topic": topic,
    }


def _identity_fact(fact, key="stored-1"):
    return {
        "fact": fact,
        "fact_context": "",
        "source_url": f"{IDENTITY_SOURCE_PREFIX}{key}",
        "source_title": "A fact the avatar already holds",
        "origin": FACT_ORIGIN_IDENTITY,
        "identity_key": key,
    }


# ── what the avatar already holds ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_existing_identity_facts_are_read_for_the_brief():
    """The facts under the identity namespace arrive as clusterable statements."""
    store = InMemoryStore()
    await store.aput(
        identity_namespace(CREATOR_ID, ASSISTANT_ID),
        key="fact-1",
        value={
            "document": {
                "kwargs": {
                    "page_content": wrap_fact_with_context(
                        "I was born in 1978.", "From an uploaded interview."
                    ),
                    "metadata": {"fact": "I was born in 1978."},
                }
            }
        },
    )

    facts = await load_existing_identity_facts(store, CREATOR_ID, ASSISTANT_ID)

    assert [fact["fact"] for fact in facts] == ["I was born in 1978."]
    assert facts[0]["origin"] == FACT_ORIGIN_IDENTITY
    assert facts[0]["source_url"].startswith(IDENTITY_SOURCE_PREFIX)


# ── verification ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_single_source_claim_is_unverified_without_a_model_call(monkeypatch):
    """One source cannot corroborate itself, so no verification call is spent."""

    async def _fail(*args, **kwargs):
        raise AssertionError("the verifier must not be called for one source")

    monkeypatch.setattr(deep_research, "invoke_structured", _fail)

    verified = await verify_cluster(
        [_web_fact("I sculpted the memorial.", "https://a")]
    )

    assert verified["status"] == "unverified"
    assert verified["supporting_source_urls"] == ["https://a"]
    assert verified["existing_fact"] is None


@pytest.mark.asyncio
async def test_a_held_fact_alone_is_reported_as_already_held(monkeypatch):
    """Nothing on the web spoke to the claim, so there is nothing to add or review."""

    async def _fail(*args, **kwargs):
        raise AssertionError("the verifier must not be called without a web source")

    monkeypatch.setattr(deep_research, "invoke_structured", _fail)

    verified = await verify_cluster([_identity_fact("I was born in 1978.")])

    assert verified["status"] == "already_held"
    assert verified["supporting_source_urls"] == []


@pytest.mark.asyncio
async def test_a_source_contradicting_a_held_fact_reaches_the_verifier(monkeypatch):
    """A held fact counts as a source, so one page disagreeing is a contradiction."""
    seen = {}

    async def _verify(response_format, system_prompt, human_text):
        seen["human_text"] = human_text
        return deep_research.FactVerification(
            status="inconsistent",
            proposed_fact="I was born in 1978.",
            conflicting_statements=["I was born in 1979."],
            reasoning="The dates differ.",
        )

    monkeypatch.setattr(deep_research, "invoke_structured", _verify)

    verified = await verify_cluster(
        [
            _web_fact("I was born in 1979.", "https://a"),
            _identity_fact("I was born in 1978."),
        ]
    )

    assert verified["status"] == "inconsistent"
    assert verified["existing_fact"] == "I was born in 1978."
    # The pseudo source names the held fact for the model but is never offered
    # to the creator as a citation.
    assert IDENTITY_SOURCE_PREFIX in seen["human_text"]
    assert verified["supporting_source_urls"] == ["https://a"]


def test_only_contradictions_wait_for_the_creator():
    """Consistent and unverified facts are applied; already-known claims are dropped."""
    consistent = {"status": "consistent", "existing_fact": None}
    unverified = {"status": "unverified", "existing_fact": None}
    corroborated_held = {"status": "consistent", "existing_fact": "I was born in 1978."}
    already_held = {"status": "already_held", "existing_fact": "I ran the mill."}
    contradiction = {"status": "inconsistent", "existing_fact": None}

    to_apply, to_review = partition_verified_facts(
        [consistent, unverified, corroborated_held, already_held, contradiction]
    )

    assert to_apply == [consistent, unverified]
    assert to_review == [contradiction]


@pytest.mark.asyncio
async def test_similar_statements_cluster_together(monkeypatch):
    """Two wordings of one claim are verified as one cluster, not two facts."""

    async def _score(query, texts):
        return [0.9 if "1978" in text and "1978" in query else 0.1 for text in texts]

    monkeypatch.setattr(deep_research, "score_similarity", _score)

    clusters = await cluster_facts(
        [
            _web_fact("I was born in 1978.", "https://a"),
            _web_fact("I was born in the year 1978.", "https://b"),
            _web_fact("I trained as a stonemason.", "https://c"),
        ]
    )

    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


# ── the whole pipeline ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_applies_agreed_facts_and_holds_the_contradiction(monkeypatch):
    """One run: the corroborated fact is learned, the contradicted one waits."""
    store = InMemoryStore()
    await store.aput(
        identity_namespace(CREATOR_ID, ASSISTANT_ID),
        key="fact-held",
        value={
            "document": {
                "kwargs": {
                    "page_content": wrap_fact_with_context("I was born in 1978.", ""),
                    "metadata": {"fact": "I was born in 1978."},
                }
            }
        },
    )

    async def _search(query, *, limit, context=None):
        return [
            SearchResult(
                url="https://a",
                title="A",
                content="page a",
                provider="test",
                queries=[query],
            ),
            SearchResult(
                url="https://b",
                title="B",
                content="page b",
                provider="test",
                queries=[query],
            ),
        ]

    monkeypatch.setattr(deep_research, "search_web", _search)

    async def _brief(subject, existing_facts, *, max_topics):
        assert "I was born in 1978." in str(existing_facts)
        return ResearchBrief(
            subject_summary="A stonemason.",
            open_questions=["When was the subject born?"],
            topics=[ResearchTopic(topic="history", assignment="Find the dates.")],
        )

    monkeypatch.setattr(deep_research, "build_research_brief", _brief)

    async def _queries(subject, topic, queries_already_run, *, max_queries):
        return ["stonemason dates"]

    monkeypatch.setattr(deep_research, "write_topic_queries", _queries)

    facts_by_source = {
        "https://a": [
            _web_fact("I carved the courthouse frieze.", "https://a"),
            _web_fact("I was born in 1979.", "https://a"),
        ],
        "https://b": [_web_fact("I carved the courthouse frieze.", "https://b")],
    }

    async def _extract(sources, *, subject, concurrency, topic=""):
        return [fact for source in sources for fact in facts_by_source[source.url]]

    monkeypatch.setattr(deep_research, "extract_facts", _extract)

    async def _score(query, texts):
        # Statements sharing their first four words are the same claim.
        head = " ".join(query.split()[:4])
        return [1.0 if text.startswith(head) else 0.0 for text in texts]

    monkeypatch.setattr(deep_research, "score_similarity", _score)

    async def _verify(response_format, system_prompt, human_text):
        if response_format is deep_research.ResearchReflection:
            # The topic is answered, so the researcher stops after one round.
            return deep_research.ResearchReflection(
                topic_is_answered=True, follow_up_queries=[]
            )
        if "1979" in human_text:
            return deep_research.FactVerification(
                status="inconsistent",
                proposed_fact="I was born in 1979.",
                conflicting_statements=["I was born in 1978."],
                reasoning="The dates differ.",
            )
        return deep_research.FactVerification(
            status="consistent",
            proposed_fact="I carved the courthouse frieze.",
            reasoning="Both sources agree.",
        )

    monkeypatch.setattr(deep_research, "invoke_structured", _verify)

    events = []
    summary = await run_deep_research(
        store,
        _context(),
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        subject_name="A stonemason",
        subject_description=None,
        research_hint=None,
        emit=events.append,
        is_cancelled=lambda: False,
    )

    assert summary["applied"] == 1
    assert summary["proposals"] == 1
    assert summary["verification_counts"]["inconsistent"] == 1

    learned = await store.asearch(identity_namespace(CREATOR_ID, ASSISTANT_ID))
    learned_facts = [
        (item.value["document"]["kwargs"]["metadata"] or {}).get("fact")
        for item in learned
    ]
    assert "I carved the courthouse frieze." in learned_facts
    # The contradicted date never reaches the avatar unreviewed.
    assert "I was born in 1979." not in learned_facts

    proposals = await list_proposals(store, CREATOR_ID, ASSISTANT_ID)
    assert [proposal["fact"] for proposal in proposals] == ["I was born in 1979."]
    assert proposals[0]["existing_fact"] == "I was born in 1978."
    assert [event["stage"] for event in events][0] == "scoping"


@pytest.mark.asyncio
async def _install_stonemason_research(monkeypatch, store):
    """Stub one whole research run: two sources, one agreed fact, one contradiction.

    The same scenario as the end-to-end test above, packaged so a second test can
    run it under a different configuration and compare the two outcomes.
    """
    await store.aput(
        identity_namespace(CREATOR_ID, ASSISTANT_ID),
        key="fact-held",
        value={
            "document": {
                "kwargs": {
                    "page_content": wrap_fact_with_context("I was born in 1978.", ""),
                    "metadata": {"fact": "I was born in 1978."},
                }
            }
        },
    )

    async def _search(query, *, limit, context=None):
        return [
            SearchResult(
                url="https://a",
                title="A",
                content="page a",
                provider="test",
                queries=[query],
            ),
            SearchResult(
                url="https://b",
                title="B",
                content="page b",
                provider="test",
                queries=[query],
            ),
        ]

    async def _brief(subject, existing_facts, *, max_topics):
        return ResearchBrief(
            subject_summary="A stonemason.",
            open_questions=["When was the subject born?"],
            topics=[ResearchTopic(topic="history", assignment="Find the dates.")],
        )

    async def _queries(subject, topic, queries_already_run, *, max_queries):
        return ["stonemason dates"]

    facts_by_source = {
        "https://a": [
            _web_fact("I carved the courthouse frieze.", "https://a"),
            _web_fact("I was born in 1979.", "https://a"),
        ],
        "https://b": [_web_fact("I carved the courthouse frieze.", "https://b")],
    }

    async def _extract(sources, *, subject, concurrency, topic=""):
        return [fact for source in sources for fact in facts_by_source[source.url]]

    async def _score(query, texts):
        head = " ".join(query.split()[:4])
        return [1.0 if text.startswith(head) else 0.0 for text in texts]

    async def _verify(response_format, system_prompt, human_text):
        if response_format is deep_research.ResearchReflection:
            return deep_research.ResearchReflection(
                topic_is_answered=True, follow_up_queries=[]
            )
        if "1979" in human_text:
            return deep_research.FactVerification(
                status="inconsistent",
                proposed_fact="I was born in 1979.",
                conflicting_statements=["I was born in 1978."],
                reasoning="The dates differ.",
            )
        return deep_research.FactVerification(
            status="consistent",
            proposed_fact="I carved the courthouse frieze.",
            reasoning="Both sources agree.",
        )

    monkeypatch.setattr(deep_research, "search_web", _search)
    monkeypatch.setattr(deep_research, "build_research_brief", _brief)
    monkeypatch.setattr(deep_research, "write_topic_queries", _queries)
    monkeypatch.setattr(deep_research, "extract_facts", _extract)
    monkeypatch.setattr(deep_research, "score_similarity", _score)
    monkeypatch.setattr(deep_research, "invoke_structured", _verify)


@pytest.mark.asyncio
async def test_facts_are_still_applied_while_the_asset_acquisition_runs(monkeypatch):
    """Attaching the portrait/voice acquisition changes nothing about the facts.

    Acquisition runs as a task alongside verification, so a failure or a hang in
    it must not cost the research its facts — that is the half the avatar's
    identity is actually built from. The same scenario as the end-to-end test
    above is run twice, once with a gateway attached and once without, and the
    facts applied and the contradictions queued must be identical.
    """

    async def _run(with_gateway):
        store = InMemoryStore()
        await _install_stonemason_research(monkeypatch, store)
        bootstrap_ran = []

        async def _acquire(*args, **kwargs):
            bootstrap_ran.append(True)
            # Acquisition failing is the case that must not touch the facts.
            raise RuntimeError("no portrait could be downloaded")

        monkeypatch.setattr(
            "src.anubis.utils.research.asset_bootstrap.run_asset_bootstrap", _acquire
        )
        summary = await run_deep_research(
            store,
            _context(),
            creator_id=CREATOR_ID,
            assistant_id=ASSISTANT_ID,
            subject_name="A stonemason",
            subject_description=None,
            research_hint=None,
            emit=lambda event: None,
            is_cancelled=lambda: False,
            bootstrap=object() if with_gateway else None,
        )
        learned = await store.asearch(identity_namespace(CREATOR_ID, ASSISTANT_ID))
        facts = sorted(
            (item.value["document"]["kwargs"]["metadata"] or {}).get("fact")
            for item in learned
        )
        return summary, facts, bool(bootstrap_ran)

    with_gateway, facts_with, ran = await _run(True)
    without_gateway, facts_without, did_not_run = await _run(False)

    assert ran is True and did_not_run is False
    # The store also holds the fact seeded before the run, so compare the two
    # runs against each other rather than against a bare list.
    assert facts_with == facts_without
    assert "I carved the courthouse frieze." in facts_with
    # The contradicted date is still held back for review in both runs.
    assert "I was born in 1979." not in facts_with
    assert with_gateway["applied"] == without_gateway["applied"] == 1
    assert with_gateway["proposals"] == without_gateway["proposals"] == 1
    # The acquisition blew up; the run still reports its facts and says so.
    assert "error" in with_gateway["bootstrap"]


class _ScriptedModel:
    """A chat model that replays a fixed list of turns, recording what it saw.

    Each entry is either a list of tool calls to emit or a plain string, which
    ends the loop the way a reply with no tool call does.
    """

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0
        self.seen_messages = []

    def __call__(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.seen_messages = list(messages)
        self.calls += 1
        turn = self._turns.pop(0) if self._turns else "Finished."
        if isinstance(turn, str):
            return AIMessage(content=turn)
        return AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": f"call-{index}"}
                for index, (name, args) in enumerate(turn)
            ],
        )


def _install_agent_model(monkeypatch, turns):
    """Point the researcher's model at a scripted transcript; return the model."""
    model = _ScriptedModel(turns)

    def _init_model(*args, **kwargs):
        return model

    monkeypatch.setattr("src.anubis.utils.model.init_model", _init_model)
    return model


@pytest.mark.asyncio
async def test_the_researcher_searches_reflects_and_decides_to_stop(monkeypatch):
    """The model, not the pipeline, ends the research on a topic.

    This is the whole point of the tool-calling loop: a reply with no tool call
    is the model saying the assignment is answered, and the loop stops there
    rather than running a further round it did not ask for.
    """
    searched = []

    async def _search(query, *, limit, context=None):
        searched.append(query)
        return [SearchResult(url=f"https://{len(searched)}", title="t", content="c")]

    monkeypatch.setattr(deep_research, "search_web", _search)

    async def _extract(sources, *, subject, concurrency, topic=""):
        return []

    monkeypatch.setattr(deep_research, "extract_facts", _extract)
    monkeypatch.setattr(deep_research, "compress_topic_research", _no_compression)

    model = _install_agent_model(
        monkeypatch,
        [
            [("search_the_web", {"query": "stonemason dates"})],
            [("record_reflection", {"reflection": "Dates found; nothing missing."})],
            "The assignment is answered.",
        ],
    )

    result = await deep_research.research_one_topic(
        "Name: A stonemason",
        ResearchTopic(topic="history", assignment="Find the dates."),
        context=_context(),
        max_queries=2,
        max_sources=4,
        concurrency=2,
        follow_up_rounds=1,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )

    assert searched == ["stonemason dates"]
    assert result["queries"] == ["stonemason dates"]
    # Three turns: the search, the reflection, and the closing reply.
    assert model.calls == 3


@pytest.mark.asyncio
async def test_the_researcher_searches_again_on_the_gap_it_named(monkeypatch):
    """An unanswered topic searches again, and the second query is the model's."""
    searched = []

    async def _search(query, *, limit, context=None):
        searched.append(query)
        return [SearchResult(url=f"https://{len(searched)}", title="t", content="c")]

    monkeypatch.setattr(deep_research, "search_web", _search)

    async def _extract(sources, *, subject, concurrency, topic=""):
        return []

    monkeypatch.setattr(deep_research, "extract_facts", _extract)
    monkeypatch.setattr(deep_research, "compress_topic_research", _no_compression)

    _install_agent_model(
        monkeypatch,
        [
            [("search_the_web", {"query": "first query"})],
            [("record_reflection", {"reflection": "No dates yet; try the archive."})],
            [("search_the_web", {"query": "second query"})],
            "Done.",
        ],
    )

    result = await deep_research.research_one_topic(
        "Name: A stonemason",
        ResearchTopic(topic="history", assignment="Find the dates."),
        context=_context(),
        max_queries=2,
        max_sources=4,
        concurrency=2,
        follow_up_rounds=1,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )

    assert searched == ["first query", "second query"]
    assert result["queries"] == ["first query", "second query"]


@pytest.mark.asyncio
async def test_the_search_budget_is_enforced_in_code_not_only_in_the_prompt(
    monkeypatch,
):
    """A model that ignores its stated budget still cannot keep searching.

    The prompt states the budget so the model can plan against it; this is the
    backstop, because one topic burning the whole run's search allowance is a
    real cost, not a style problem.
    """
    searched = []

    async def _search(query, *, limit, context=None):
        searched.append(query)
        return [SearchResult(url=f"https://{len(searched)}", title="t", content="c")]

    monkeypatch.setattr(deep_research, "search_web", _search)

    async def _extract(sources, *, subject, concurrency, topic=""):
        return []

    monkeypatch.setattr(deep_research, "extract_facts", _extract)
    monkeypatch.setattr(deep_research, "compress_topic_research", _no_compression)

    # Ten searches asked for, against a budget of max_queries + follow_up_rounds.
    _install_agent_model(
        monkeypatch,
        [[("search_the_web", {"query": f"query {index}"})] for index in range(10)]
        + ["Done."],
    )

    await deep_research.research_one_topic(
        "Name: A stonemason",
        ResearchTopic(topic="history", assignment="Find the dates."),
        context=_context(),
        max_queries=2,
        max_sources=4,
        concurrency=2,
        follow_up_rounds=1,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )

    assert len(searched) == 3


@pytest.mark.asyncio
async def test_the_compression_keeps_the_findings_and_drops_the_reflections(
    monkeypatch,
):
    """Reflections are the researcher's reasoning, not information about the subject.

    Letting them through would write the agent's own deliberation into the
    findings a later step reads as though it were sourced fact.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    compressed_input = {}

    class _CompressionModel:
        async def ainvoke(self, messages):
            compressed_input["human"] = messages[-1].content
            return AIMessage(content="cleaned findings")

    monkeypatch.setattr(
        "src.anubis.utils.model.init_model", lambda *a, **k: _CompressionModel()
    )

    messages = [
        AIMessage(content=""),
        ToolMessage(
            content="<source url='https://a'>the frieze was carved in 1903</source>",
            name="search_the_web",
            tool_call_id="1",
        ),
        ToolMessage(
            content="Reflection recorded: I still need the birth date.",
            name="record_reflection",
            tool_call_id="2",
        ),
    ]
    output = await deep_research.compress_topic_research(
        "Name: A stonemason",
        ResearchTopic(topic="history", assignment="Find the dates."),
        messages,
    )

    assert output == "cleaned findings"
    assert "the frieze was carved in 1903" in compressed_input["human"]
    assert "I still need the birth date" not in compressed_input["human"]


@pytest.mark.asyncio
async def test_a_topic_is_never_dropped_when_the_agent_cannot_run(monkeypatch):
    """A model failure falls back to pipeline-written queries rather than nothing.

    Returning no sources for a topic would silently narrow the research without
    anyone being told, so the fallback keeps the topic in the run.
    """
    searched = []

    async def _search(query, *, limit, context=None):
        searched.append(query)
        return [SearchResult(url="https://a", title="t", content="c")]

    monkeypatch.setattr(deep_research, "search_web", _search)

    async def _queries(subject, topic, queries_already_run, *, max_queries):
        return ["fallback query"]

    monkeypatch.setattr(deep_research, "write_topic_queries", _queries)

    async def _extract(sources, *, subject, concurrency, topic=""):
        return []

    monkeypatch.setattr(deep_research, "extract_facts", _extract)
    monkeypatch.setattr(deep_research, "compress_topic_research", _no_compression)

    def _broken_model(*args, **kwargs):
        raise RuntimeError("no model configured")

    monkeypatch.setattr("src.anubis.utils.model.init_model", _broken_model)

    result = await deep_research.research_one_topic(
        "Name: A stonemason",
        ResearchTopic(topic="history", assignment="Find the dates."),
        context=_context(),
        max_queries=2,
        max_sources=4,
        concurrency=2,
        follow_up_rounds=1,
        emit=lambda payload: None,
        is_cancelled=lambda: False,
    )

    assert searched == ["fallback query"]
    assert [source.url for source in result["sources"]] == ["https://a"]


async def _no_compression(subject, topic, messages):
    return ""


# ── resolving what the creator decided ──────────────────────────────────────


@pytest.mark.asyncio
async def test_resolving_writes_the_creators_wording_and_drops_the_ignored():
    """Accept keeps the researched fact, edit keeps the correction, ignore forgets."""
    store = InMemoryStore()
    namespace = research_proposal_namespace(CREATOR_ID, ASSISTANT_ID)
    for fact_id, fact in (
        ("keep", "I was born in 1979."),
        ("fix", "I trained in Leeds."),
        ("drop", "I never left the county."),
    ):
        document = deep_research.build_proposal_document(
            {
                "status": "inconsistent",
                "proposed_fact": fact,
                "fact_context": "context",
                "supporting_source_urls": ["https://a"],
                "conflicting_statements": ["something else"],
                "statements": [fact],
                "existing_fact": None,
            },
            creator_id=CREATOR_ID,
            assistant_id=ASSISTANT_ID,
            subject_name="A stonemason",
        )
        document.metadata["fact_id"] = fact_id
        await store.aput(namespace, key=fact_id, value={"document": document.to_json()})

    result = await resolve_proposals(
        store,
        CREATOR_ID,
        ASSISTANT_ID,
        [
            ProposalResolution(fact_id="keep", action="accept"),
            ProposalResolution(
                fact_id="fix", action="edit", corrected_text="I trained in Bradford."
            ),
            ProposalResolution(fact_id="drop", action="ignore"),
        ],
    )

    assert result == {
        "accepted": 1,
        "edited": 1,
        "ignored": 1,
        "missing": 0,
        "accepted_source_urls": ["https://a"],
    }
    learned = await store.asearch(identity_namespace(CREATOR_ID, ASSISTANT_ID))
    facts = {item.value["document"]["kwargs"]["metadata"]["fact"] for item in learned}
    assert facts == {"I was born in 1979.", "I trained in Bradford."}
    assert await list_proposals(store, CREATOR_ID, ASSISTANT_ID) == []


def test_a_researched_fact_names_its_sources_and_its_verification():
    """The stored document carries what the settings screen shows about the fact."""
    document = build_identity_document(
        {
            "status": "consistent",
            "proposed_fact": "I carved the courthouse frieze.",
            "fact_context": "From the county archive.",
            "supporting_source_urls": ["https://a", "https://b"],
        },
        creator_id=CREATOR_ID,
        assistant_id=ASSISTANT_ID,
        subject_name="A stonemason",
    )

    assert document.metadata["source"] == deep_research.RESEARCH_FACT_SOURCE
    assert document.metadata["verification_status"] == "consistent"
    assert document.metadata["source_urls"] == ["https://a", "https://b"]
    assert "I carved the courthouse frieze." in document.page_content


# ── the configurable emotion-media tier gate ────────────────────────────────


def test_the_emotion_media_minimum_tier_fails_closed_to_premium():
    """An unset or misspelt minimum never hands the feature to everyone."""
    assert minimum_tier_from_value(None) is SubscriptionTier.PREMIUM
    assert minimum_tier_from_value("") is SubscriptionTier.PREMIUM
    assert minimum_tier_from_value("enterprise") is SubscriptionTier.PREMIUM
    assert minimum_tier_from_value("pro") is SubscriptionTier.PRO


def test_only_tiers_at_or_above_the_minimum_generate_emotion_media():
    """Premium is the enterprise-grade tier today, so pro and free are refused."""
    assert tier_meets_minimum(SubscriptionTier.PREMIUM, SubscriptionTier.PREMIUM)
    assert not tier_meets_minimum(SubscriptionTier.PRO, SubscriptionTier.PREMIUM)
    assert not tier_meets_minimum(SubscriptionTier.FREE, SubscriptionTier.PREMIUM)
    assert tier_meets_minimum(SubscriptionTier.PRO, SubscriptionTier.PRO)


def test_the_media_graph_skips_generation_below_the_minimum_tier():
    """A reference-image upload from a lower tier stores the image and spends nothing."""
    assert not emotion_media_tier_allows_generation("pro", "premium")
    assert emotion_media_tier_allows_generation("premium", "premium")
    assert emotion_media_tier_allows_generation("pro", "pro")
    # A run with no tier at all (Studio, a script, a test) is not gated.
    assert emotion_media_tier_allows_generation(None, "premium")
