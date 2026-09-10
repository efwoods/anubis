"""The legal documents reach the prompts that are supposed to be reading them.

This file exists because they did not. ``src/anubis/utils/prompts/legal/`` had no
``__init__.py``, so it was an implicit namespace package and
``from src.anubis.utils.prompts.legal import TERMS_OF_SERVICE`` bound the
*module* of that name rather than the string inside it. Every caller then
interpolated a module repr into its prompt, and the content-moderation judge —
the thing that bans accounts — was deciding violations with

    <module 'src.anubis.utils.prompts.legal.TERMS_OF_SERVICE' from '/home/...'>

where the terms were meant to be. It could never populate ``violated_clauses``,
because it had never been shown a clause. Nothing caught it: the judge's own
tests replace the model call, and an end-to-end check on clean content returns
"no violation" whether the judge works or not.

So the assertions here are deliberately about the *rendered prompt text*: that
the real documents are in it, and that no module repr is. Those two hold
whatever the import style is, and they fail loudly the next time somebody
reaches for these documents the wrong way.
"""

import pytest

# The judge lives in a package that is landing alongside this file. Skipping
# rather than failing keeps a clean checkout collectable, and the moment the
# package is committed these assertions start running with no edit here.
content_moderation = pytest.importorskip(
    "src.anubis.utils.moderation.content_moderation",
    reason="the content-moderation package has not landed yet",
)
build_moderation_system_prompt = content_moderation.build_moderation_system_prompt

from src.anubis.utils.prompts.legal import (  # noqa: E402 - after the skip guard above
    PRIVACY_POLICY,
    TERMS_OF_SERVICE,
    known_platforms,
    platform_policy,
    render_platform_policies,
)

# A line from each real document. If either document is rewritten these change,
# which is correct: the point is that real text is present, not a length.
A_REAL_TERMS_CLAUSE = "You must be at least 13 years old"
A_REAL_PRIVACY_CLAUSE = "Privacy Policy"


def test_the_documents_are_text_and_not_modules():
    assert isinstance(TERMS_OF_SERVICE, str)
    assert isinstance(PRIVACY_POLICY, str)
    assert len(TERMS_OF_SERVICE) > 1000
    assert len(PRIVACY_POLICY) > 1000


def test_the_judge_is_shown_the_real_documents():
    """The assertion that would have caught the bug this file is named for."""
    prompt = build_moderation_system_prompt()
    assert "<module " not in prompt, (
        "a module repr reached the judge's prompt: the legal documents are being "
        "imported as modules rather than as text"
    )
    assert ".py'>" not in prompt, "a file path reached the judge's prompt"
    assert A_REAL_TERMS_CLAUSE in prompt
    assert A_REAL_PRIVACY_CLAUSE in prompt
    # The whole point of the judge quoting clauses is that clauses are present.
    assert len(prompt) > 8000


def test_the_room_triage_is_shown_the_real_documents():
    from src.anubis.utils.groups.triage import GROUP_TRIAGE_SYSTEM_PROMPT

    prompt = GROUP_TRIAGE_SYSTEM_PROMPT.format(
        owner_name="Evan",
        owner_rules="(none)",
        past_decisions="(none)",
        available_actions="warn",
        available_decision_actions="ignore, notify, respond",
        recent_events="(nothing)",
        platform_terms=TERMS_OF_SERVICE
        + "\n\n"
        + render_platform_policies(["twitch"]),
    )
    assert "<module " not in prompt
    assert A_REAL_TERMS_CLAUSE in prompt


# --------------------------------------------------------------------------
# The other companies' rules
# --------------------------------------------------------------------------


def test_every_platform_the_avatar_speaks_on_has_rules_on_file():
    for platform in ("slack", "discord", "twitch"):
        assert platform in known_platforms()
        policy = platform_policy(platform)
        assert policy is not None
        assert policy.terms_url.startswith("https://")
        assert policy.privacy_url.startswith("https://")
        assert policy.obligations


def test_the_connected_account_providers_have_rules_on_file():
    for provider in ("google", "github", "x"):
        assert platform_policy(provider) is not None


def test_a_platform_is_found_however_the_name_is_cased():
    assert platform_policy("Twitch") is platform_policy("twitch")
    assert platform_policy("  TWITCH  ") is platform_policy("twitch")


def test_an_unknown_company_is_skipped_rather_than_raising():
    """A new integration must reach the judge before anybody writes its policy."""
    assert platform_policy("nosuchplace") is None
    assert render_platform_policies(["nosuchplace"]) == ""
    assert render_platform_policies(None) == ""
    # A known company alongside an unknown one still renders.
    rendered = render_platform_policies(["nosuchplace", "slack"])
    assert "Slack" in rendered


def test_the_rules_that_cost_this_project_an_account_are_written_down():
    """Twitch suspended the project for exactly these three things."""
    rendered = render_platform_policies(["twitch"])
    assert "gql.twitch.tv" in rendered
    assert "session cookie" in rendered
    assert "client identifier" in rendered


def test_every_platform_carries_the_obligations_common_to_all_of_them():
    rendered = render_platform_policies(["discord"])
    # Answering "am I talking to a person?" honestly is required everywhere.
    assert "human being" in rendered
    assert "impersonate" in rendered


def test_the_verbatim_document_wins_over_the_summary_when_one_is_supplied():
    from src.anubis.utils.prompts.legal.platform_policies import (
        PlatformPolicy,
        render_platform_policy,
    )

    summarized = PlatformPolicy(
        name="Example",
        terms_url="https://example.test/terms",
        privacy_url="https://example.test/privacy",
        obligations=("A summarized obligation.",),
    )
    verbatim = PlatformPolicy(
        name="Example",
        terms_url="https://example.test/terms",
        privacy_url="https://example.test/privacy",
        obligations=("A summarized obligation.",),
        full_text="1. The verbatim clause of the real document.",
    )
    assert "A summarized obligation." in render_platform_policy(summarized)
    rendered = render_platform_policy(verbatim)
    assert "1. The verbatim clause of the real document." in rendered
    assert "A summarized obligation." not in rendered
    # The URLs stay either way: they are what makes the record checkable.
    assert "https://example.test/terms" in rendered


# --------------------------------------------------------------------------
# The judge's third-party block
# --------------------------------------------------------------------------


def test_the_third_party_block_appears_only_when_a_company_is_named():
    assert "<THIRD_PARTY_PLATFORM_POLICIES>" not in build_moderation_system_prompt()
    with_platforms = build_moderation_system_prompt(["twitch", "slack"])
    assert "<THIRD_PARTY_PLATFORM_POLICIES>" in with_platforms
    assert "Twitch" in with_platforms and "Slack" in with_platforms


def test_the_judge_is_told_not_to_quote_a_summary_as_a_clause():
    """A ban record and its appeal rest on violated_clauses being real lines."""
    prompt = build_moderation_system_prompt(["twitch"])
    assert "never copy one into violated_clauses" in prompt

    fields = content_moderation.TermsAndServicesContentModeration.model_fields
    assert "violated_platform_rules" in fields, (
        "a third-party rule needs somewhere to go that is not violated_clauses"
    )


@pytest.mark.asyncio
async def test_the_platforms_reach_the_model_call(monkeypatch):
    """Whatever a caller names has to actually arrive in the prompt."""
    seen = {}

    class _Model:
        async def ainvoke(self, messages, **kwargs):
            seen["prompt"] = str(messages[0].content)
            return content_moderation.TermsAndServicesContentModeration(
                violation=False, reasoning="", violated_clauses=[]
            )

    from src.anubis.utils import model as model_module

    monkeypatch.setattr(model_module, "init_model", lambda **kwargs: _Model())
    verdict = await content_moderation.judge_text("hello", platforms=["discord"])
    assert verdict["violation"] is False
    assert "Discord" in seen["prompt"]
    assert "<module " not in seen["prompt"]
