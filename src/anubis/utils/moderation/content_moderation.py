"""The terms-of-service / privacy-policy judge shared by messages and uploads.

One structured-output model call decides whether a piece of text violates the
TERMS_OF_SERVICE or the PRIVACY_POLICY, returning the exact clauses violated.
``judge_text`` judges one text; ``judge_documents`` judges many concurrently
(bounded) and returns the first violation.

Both are fail-OPEN: when the judge itself fails (model outage, malformed
output) the text is treated as clean and the failure is logged, because a
moderation outage must never lock every person out of the product — the ban
that follows a violation is far too heavy a consequence to hand out on a
classifier error.

This module holds the DEEP judge only. The cheap first pass that runs inline on
every chat turn lives in ``src.anubis.utils.moderation.fast_screen``; the graph
that sequences the two lives in ``src.subgraphs.moderation_graph.graph``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.anubis.utils.prompts.legal import PRIVACY_POLICY, TERMS_OF_SERVICE

logger = logging.getLogger(__name__)

# Uploaded documents can run to hundreds of thousands of characters; the judge
# reads a bounded head of each so a single upload cannot cost an unbounded
# classification call. Long documents are judged in windows of this size.
DEFAULT_MAX_CHARACTERS = 6000
# Upper bound on how many windows of one long document are judged.
MAX_WINDOWS_PER_DOCUMENT = 8


class TermsAndServicesContentModeration(BaseModel):
    """The judge's verdict on one piece of content."""

    violation: bool = Field(
        description=(
            "TRUE when the CONTENT violates the TERMS_OF_SERVICE or the PRIVACY_POLICY. "
            "FALSE when the CONTENT violates neither."
        )
    )
    reasoning: str = Field(
        description=(
            "The clear reason the CONTENT violates the terms of service or the privacy "
            "policy. Empty when there is no violation."
        )
    )
    violated_clauses: list[str] = Field(
        default_factory=list,
        description=(
            "Every exact line of the TERMS_OF_SERVICE or the PRIVACY_POLICY that the "
            "CONTENT violates, quoted verbatim and unaltered. Empty when there is no "
            "violation. Never quote anything from THIRD_PARTY_PLATFORM_POLICIES here."
        ),
    )
    violated_platform_rules: list[str] = Field(
        default_factory=list,
        description=(
            "Every rule from THIRD_PARTY_PLATFORM_POLICIES that the CONTENT violates, "
            "each written as 'Company: the rule'. These are summaries of another "
            "company's documents rather than quotations, so they belong here and never "
            "in violated_clauses. Empty when there is no violation."
        ),
    )


CONTENT_MODERATION_SYSTEM_PROMPT = """
<ROLE>
You are an expert judge of violations of the terms of service and the privacy policy in content that a person sends to or uploads into the Neural Nexus platform.
</ROLE>

<INSTRUCTIONS>
Determine whether the CONTENT violates the TERMS_OF_SERVICE or the PRIVACY_POLICY.
Return violation TRUE only when the CONTENT itself clearly violates a specific clause: unlawful, harmful, defamatory, or infringing material; attempts to gain unauthorized access; automated abuse; interference with the service; or misuse of another person's private data.
Return violation FALSE for ordinary conversation, personal stories, opinions, creative writing, questions, and any content that merely mentions a sensitive topic without violating a clause.
When there is a violation, give a clear reason and quote EVERY exact clause of the TERMS_OF_SERVICE or the PRIVACY_POLICY that the CONTENT violates, unaltered.
When there is no violation, leave the reason empty and the list of clauses empty.
THIRD_PARTY_PLATFORM_POLICIES, when present, are the rules of the other companies whose services this content passes through. Treat a violation of those rules as a violation too, and record each one in violated_platform_rules as 'Company: the rule'. Those rules are summaries of another company's documents, not quotations, so never copy one into violated_clauses: violated_clauses holds verbatim lines of the TERMS_OF_SERVICE and the PRIVACY_POLICY above and nothing else.
</INSTRUCTIONS>

<TERMS_OF_SERVICE>
{terms_of_service}
</TERMS_OF_SERVICE>

<PRIVACY_POLICY>
{privacy_policy}
</PRIVACY_POLICY>
{third_party_platform_policies}
"""

THIRD_PARTY_POLICY_BLOCK = """
<THIRD_PARTY_PLATFORM_POLICIES>
{platform_policies}
</THIRD_PARTY_PLATFORM_POLICIES>
"""


def build_moderation_system_prompt(platforms: list[str] | None = None) -> str:
    """Build the judge's system prompt from the documents that bind this content.

    ``platforms`` names the other companies whose services the content passes
    through — the platform a group message came from, the provider of a
    connected account. Their rules are added in their own block, because an
    avatar that keeps our terms can still break Slack's or Twitch's, and the
    consequence of that lands on the owner's account rather than on ours.
    """
    from src.anubis.utils.prompts.legal import render_platform_policies

    rendered = render_platform_policies(platforms)
    return CONTENT_MODERATION_SYSTEM_PROMPT.format(
        terms_of_service=TERMS_OF_SERVICE,
        privacy_policy=PRIVACY_POLICY,
        third_party_platform_policies=(
            THIRD_PARTY_POLICY_BLOCK.format(platform_policies=rendered)
            if rendered
            else ""
        ),
    )


def clean_verdict() -> dict[str, Any]:
    """Build the verdict shape used everywhere: no violation, nothing to report."""
    return {
        "violation": False,
        "reasoning": "",
        "violated_clauses": [],
        "violated_platform_rules": [],
    }


def moderation_flag_enabled(value: object, default: bool = True) -> bool:
    """Read one of the TRUE/FALSE string flags on ``GlobalContext``."""
    if value is None:
        return default
    return str(value).strip().upper() == "TRUE"


async def invoke_moderation_model(
    content: str, platforms: list[str] | None = None
) -> TermsAndServicesContentModeration:
    """One judge call. Isolated so tests can replace the model."""
    from src.anubis.utils.model import init_model

    model = init_model(response_format=TermsAndServicesContentModeration)
    response = await model.ainvoke(
        [
            SystemMessage(content=build_moderation_system_prompt(platforms)),
            HumanMessage(content=f"<CONTENT>\n{content}\n</CONTENT>"),
        ]
    )
    if isinstance(response, tuple):
        response = response[0]
    if isinstance(response, TermsAndServicesContentModeration):
        return response
    return TermsAndServicesContentModeration.model_validate(response)


async def judge_text(
    text: str,
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    """Judge one text; returns ``{violation, reasoning, violated_clauses, excerpt}``."""
    text = (text or "").strip()
    if not text:
        return clean_verdict()
    windows = [
        text[start : start + max_characters]
        for start in range(0, len(text), max(1, max_characters))
    ][:MAX_WINDOWS_PER_DOCUMENT]
    for window in windows:
        try:
            verdict = await invoke_moderation_model(window, platforms)
        except Exception as judge_error:  # noqa: BLE001 - fail open, see module docstring
            logger.error(
                "Content moderation judge failed (treating as clean): %s", judge_error
            )
            return {**clean_verdict(), "judge_error": str(judge_error)}
        if verdict.violation:
            return {
                "violation": True,
                "reasoning": verdict.reasoning,
                "violated_clauses": list(verdict.violated_clauses),
                "violated_platform_rules": list(
                    getattr(verdict, "violated_platform_rules", []) or []
                ),
                "excerpt": window[:500],
            }
    return clean_verdict()


async def judge_documents(
    documents: list[Document],
    *,
    concurrency: int = 4,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    """Judge many documents concurrently; the first violation found is returned."""
    documents = [
        document for document in documents if (document.page_content or "").strip()
    ]
    if not documents:
        return clean_verdict()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _judge_one(document: Document) -> dict[str, Any]:
        async with semaphore:
            verdict = await judge_text(
                document.page_content,
                max_characters=max_characters,
                platforms=platforms,
            )
        if verdict.get("violation"):
            metadata = document.metadata or {}
            verdict["source"] = metadata.get("source_filename") or metadata.get(
                "filename"
            )
        return verdict

    verdicts = await asyncio.gather(*(_judge_one(document) for document in documents))
    for verdict in verdicts:
        if verdict.get("violation"):
            return verdict
    return clean_verdict()


# The state key the process-media graph writes its verdict into.
MEDIA_MODERATION_STATE_KEY = "moderation_violation"


__all__ = [
    "DEFAULT_MAX_CHARACTERS",
    "MEDIA_MODERATION_STATE_KEY",
    "MAX_WINDOWS_PER_DOCUMENT",
    "TermsAndServicesContentModeration",
    "build_moderation_system_prompt",
    "clean_verdict",
    "invoke_moderation_model",
    "judge_documents",
    "judge_text",
    "moderation_flag_enabled",
]
