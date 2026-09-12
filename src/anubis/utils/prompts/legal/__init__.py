"""The legal documents every judge and every room prompt reads.

``TERMS_OF_SERVICE`` and ``PRIVACY_POLICY`` are Neural Nexus's own documents, as
text. ``render_platform_policies`` renders the rules of the other companies this
platform integrates with — Slack, Discord, Twitch, and the providers behind
connected accounts — for the same prompts.

**Why this package has an ``__init__``.** It did not, which made it an implicit
namespace package, and so ``from src.anubis.utils.prompts.legal import
TERMS_OF_SERVICE`` imported the *module* of that name rather than the string
inside it. Every caller then interpolated a module repr into its prompt: the
content-moderation judge spent weeks deciding violations with
``<module '...TERMS_OF_SERVICE' from '/home/.../TERMS_OF_SERVICE.py'>`` where the
terms were supposed to be, and could never quote a violated clause because it
had never been shown one. The document files are therefore named after what they
are (``neural_nexus_terms_of_service.py``) rather than after the constant they
hold, so a submodule name can never shadow a string name again, and the names
below are the only supported way to reach them.
"""

from src.anubis.utils.prompts.legal.neural_nexus_privacy_policy import PRIVACY_POLICY
from src.anubis.utils.prompts.legal.neural_nexus_terms_of_service import (
    TERMS_OF_SERVICE,
)
from src.anubis.utils.prompts.legal.platform_policies import (
    COMMON_AUTOMATED_PARTICIPANT_OBLIGATIONS,
    PLATFORM_POLICIES,
    PlatformPolicy,
    known_platforms,
    platform_policy,
    render_platform_policies,
    render_platform_policy,
)

__all__ = [
    "COMMON_AUTOMATED_PARTICIPANT_OBLIGATIONS",
    "PLATFORM_POLICIES",
    "PRIVACY_POLICY",
    "TERMS_OF_SERVICE",
    "PlatformPolicy",
    "known_platforms",
    "platform_policy",
    "render_platform_policies",
    "render_platform_policy",
]
