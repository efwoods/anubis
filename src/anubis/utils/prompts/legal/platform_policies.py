"""The rules of the other companies whose services this platform integrates with.

Neural Nexus does not only have to keep its own terms. An avatar that posts in
a Slack workspace, a Discord guild, or a Twitch chat is also bound by *that*
company's terms of service, developer policy, and privacy policy, and so is
every account a person connects. A judge that reads only our own documents
cannot see a violation of theirs — it would let the avatar do something that
gets the owner's Slack app removed or their Twitch account suspended, which has
already happened once on this project.

Each entry carries the canonical URLs and, in ``obligations``, the rules that
actually bind an automated participant, written plainly for a model to apply.

**``obligations`` is a summary written for this judge, not a quotation.** The
authoritative document is the one at the URL, and a summary is what makes a
judge usable: the real documents run to tens of thousands of characters each,
and putting six of them in every moderation call would cost more than the reply
being judged. Where the verbatim document is wanted — for a platform under
review, or a company whose terms are unusual — paste it into ``full_text`` and
the judge uses that instead, with no other change.

Adding a company is one ``PlatformPolicy`` entry in ``PLATFORM_POLICIES``. The
key is the platform name used everywhere else in the codebase (the
``source_kind`` of an inbox item, the ``platform`` a bot posts, the ``provider``
of a connected account), so the right policies are found without a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformPolicy:
    """One company's rules, as the moderation judge and the room triage read them."""

    name: str
    terms_url: str
    privacy_url: str
    developer_terms_url: str = ""
    obligations: tuple[str, ...] = ()
    # Paste the verbatim document here to have the judge read that instead of
    # the summary above. Empty by default, because the real documents are long
    # enough to dominate the cost of every call that carries them.
    full_text: str = ""
    notes: str = ""


# Obligations shared by every platform an automated participant speaks on. They
# are stated once and prepended to each platform's own, so a new entry does not
# have to restate them and cannot forget one.
COMMON_AUTOMATED_PARTICIPANT_OBLIGATIONS: tuple[str, ...] = (
    "Do not present the avatar as a human being when somebody asks whether they "
    "are speaking to a person or to software. Answering that question honestly is "
    "required by every platform here.",
    "Do not impersonate another person, another company, or a platform employee.",
    "Do not collect, store, or repeat anybody else's personal data beyond what the "
    "conversation itself needs, and never a credential, a payment detail, or a "
    "private address.",
    "Do not send unsolicited advertising, referral links, or bulk identical "
    "messages to people who did not ask for them.",
    "Do not use the platform's data to train a model on the people in the room "
    "without their knowledge, and never treat a private room as public material.",
    "Respect the platform's rate limits and never work around a block, a "
    "suspension, or a permission the account was not granted.",
    "Reach the platform only through the documented public API, with credentials "
    "issued to this application. Never a borrowed client identifier, an "
    "undocumented endpoint, or an automated session driving the private client.",
)


PLATFORM_POLICIES: dict[str, PlatformPolicy] = {
    "slack": PlatformPolicy(
        name="Slack",
        terms_url="https://slack.com/terms-of-service",
        privacy_url="https://slack.com/trust/privacy/privacy-policy",
        developer_terms_url="https://slack.com/terms-of-service/api",
        obligations=(
            "Post only in the channels the workspace installed this app into, and "
            "only with the scopes the workspace approved.",
            "Treat the contents of a workspace as that customer's confidential data. "
            "Do not move a message out of the workspace it was written in.",
            "Disclose that messages come from an app, which Slack shows through the "
            "app's own identity — never dress a message as a human member's.",
            "Do not use workspace data for advertising or profiling of the members.",
        ),
    ),
    "discord": PlatformPolicy(
        name="Discord",
        terms_url="https://discord.com/terms",
        privacy_url="https://discord.com/privacy",
        developer_terms_url="https://discord.com/developers/docs/policies-and-agreements/developer-terms-of-service",
        obligations=(
            "Use the official API and gateway with the bot's own token. A user "
            "account driven by software (a self-bot) is forbidden outright and is "
            "grounds for termination.",
            "Request only the gateway intents the feature genuinely needs, and use "
            "message content only for the purpose the server was told about.",
            "Do not store message content longer than the feature needs, and delete "
            "what a server or a member asks to have deleted.",
            "Moderation actions are the server's to define: act only where the bot "
            "holds the permission and the server's own rules allow.",
        ),
    ),
    "twitch": PlatformPolicy(
        name="Twitch",
        terms_url="https://www.twitch.tv/p/legal/terms-of-service/",
        privacy_url="https://www.twitch.tv/p/legal/privacy-notice/",
        developer_terms_url="https://www.twitch.tv/p/legal/developer-agreement/",
        obligations=(
            "Use only the documented Helix API and EventSub with a client "
            "identifier registered to this application. Never send Twitch's own "
            "first-party client identifier, and never call an undocumented endpoint "
            "such as the GraphQL or usher hosts.",
            "Do not automate a login, reuse a captured session cookie, or drive the "
            "website with an anti-detection browser.",
            "Follow the channel's own rules and the Community Guidelines when "
            "moderating, and hold the moderator scopes the broadcaster granted.",
            "Do not record or republish a broadcast, and do not build a profile of a "
            "viewer from chat.",
        ),
        notes=(
            "This platform is the reason these rules are written down: the project's "
            "Twitch account was indefinitely suspended for Fraud after a "
            "browser-automated login, a reused session cookie, and calls to "
            "gql.twitch.tv carrying Twitch's own client identifier. See "
            "f-anubis-twitch/REPORT.md."
        ),
    ),
    "google": PlatformPolicy(
        name="Google",
        terms_url="https://policies.google.com/terms",
        privacy_url="https://policies.google.com/privacy",
        developer_terms_url="https://developers.google.com/terms/api-services-user-data-policy",
        obligations=(
            "Use data from a Google account only to provide the feature the person "
            "granted the scope for, and never for advertising or model training.",
            "Keep the scopes granted to the minimum the feature needs, and stop using "
            "the data when the person disconnects the account.",
            "Do not transfer a person's Google data to anybody else except as the "
            "person directed or the law requires.",
        ),
    ),
    "github": PlatformPolicy(
        name="GitHub",
        terms_url="https://docs.github.com/site-policy/github-terms/github-terms-of-service",
        privacy_url="https://docs.github.com/site-policy/privacy-policies/github-privacy-statement",
        developer_terms_url="https://docs.github.com/site-policy/github-terms/github-terms-for-additional-products-and-features",
        obligations=(
            "Act only with the permissions the person granted, on the repositories "
            "they chose.",
            "Do not scrape user profiles or use the API to build a directory of "
            "people.",
            "Do not publish somebody's private repository content or their private "
            "email address.",
        ),
    ),
    "x": PlatformPolicy(
        name="X",
        terms_url="https://x.com/en/tos",
        privacy_url="https://x.com/en/privacy",
        developer_terms_url="https://developer.x.com/en/developer-terms/agreement-and-policy",
        obligations=(
            "Do not post duplicate or bulk automated content, and do not automate "
            "engagement (mass following, liking, or replying).",
            "Post automated content only from an account whose automation is "
            "disclosed as the developer policy requires.",
            "Do not use the API to track a person's location or to build a profile "
            "of somebody's private characteristics.",
        ),
    ),
}


def platform_policy(platform: str) -> PlatformPolicy | None:
    """Return one company's policy record, or ``None`` when the company is unknown."""
    return PLATFORM_POLICIES.get(str(platform or "").strip().lower())


def render_platform_policy(policy: PlatformPolicy) -> str:
    """Render one company's rules for a prompt."""
    if policy.full_text.strip():
        body = policy.full_text.strip()
    else:
        obligations = COMMON_AUTOMATED_PARTICIPANT_OBLIGATIONS + tuple(policy.obligations)
        body = "\n".join(f"- {line}" for line in obligations)
    lines = [
        f"{policy.name} — terms of service: {policy.terms_url}",
        f"{policy.name} — privacy policy: {policy.privacy_url}",
    ]
    if policy.developer_terms_url:
        lines.append(f"{policy.name} — developer terms: {policy.developer_terms_url}")
    if policy.notes:
        lines.append(f"Note: {policy.notes}")
    lines.append(body)
    return "\n".join(lines)


def render_platform_policies(platforms: list[str] | tuple[str, ...] | None) -> str:
    """Render the rules of every named company, for a prompt.

    Unknown names are skipped rather than raising: a new integration must be
    able to reach the judge before anybody has written its policy record, and
    the common obligations above still apply to it through Neural Nexus's own
    terms.
    """
    rendered = []
    for platform in platforms or ():
        policy = platform_policy(platform)
        if policy is None:
            continue
        rendered.append(render_platform_policy(policy))
    return "\n\n".join(rendered)


def known_platforms() -> tuple[str, ...]:
    """Every company whose rules are on file."""
    return tuple(sorted(PLATFORM_POLICIES))


__all__ = [
    "COMMON_AUTOMATED_PARTICIPANT_OBLIGATIONS",
    "PLATFORM_POLICIES",
    "PlatformPolicy",
    "known_platforms",
    "platform_policy",
    "render_platform_policies",
    "render_platform_policy",
]
