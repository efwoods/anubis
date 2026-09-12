"""A site refusing an automated visit is reported as that, not as a lapsed session.

Cloudflare's interstitial is an ordinary HTML page. Without this, two things go
wrong, and both hand the owner a confident wrong answer:

* the keepalive sees a page with no signed-in markers, calls the session lapsed,
  and asks the owner to sign in again — which cannot help, because the session
  was never the problem;
* a read returns "Sorry, you have been blocked" as the page text, so a question
  like "what am I spending?" gets an answer shaped like a real one.

What is pinned down here is the detector's precision. A false positive is worse
than a miss: it would report a working page as blocked.
"""

import pytest

from src.anubis.utils.connected_accounts.browser_sessions import (
    bot_wall_advice,
    bot_wall_detected,
)

CLOUDFLARE_BLOCK = """
<html><head><title>Attention Required! | Cloudflare</title></head>
<body>Sorry, you have been blocked. You are unable to access x.ai.
Cloudflare Ray ID: a3909f251fea03c8</body></html>
"""

CLOUDFLARE_CHALLENGE = """
<html><head><title>Just a moment...</title></head>
<body><div class="cf-browser-verification">Checking your browser before accessing
the site.</div></body></html>
"""


""" What a wall looks like """


@pytest.mark.parametrize("status", [403, 429, 503])
def test_a_block_page_with_a_wall_status_is_a_wall(status):
    assert bot_wall_detected(CLOUDFLARE_BLOCK, status_code=status) is True


def test_a_challenge_interstitial_is_a_wall():
    assert bot_wall_detected(CLOUDFLARE_CHALLENGE, status_code=503) is True


def test_a_rendered_page_is_judged_on_its_words_alone():
    """A navigated page has no status code; the markers have to carry it."""
    assert bot_wall_detected(CLOUDFLARE_BLOCK) is True
    assert bot_wall_detected(CLOUDFLARE_CHALLENGE) is True


""" What a wall does NOT look like — the expensive mistakes """


def test_an_ordinary_dashboard_is_not_a_wall():
    page = "<html><body><h1>Usage</h1><p>You spent $64.08 this month.</p></body></html>"
    assert bot_wall_detected(page, status_code=200) is False
    assert bot_wall_detected(page) is False


def test_a_signed_out_403_is_not_a_wall():
    """A 403 alone is ordinary — plenty of real pages 403 for a signed-out reader."""
    page = "<html><body>You do not have permission to view this project.</body></html>"
    assert bot_wall_detected(page, status_code=403) is False


def test_prose_containing_a_marker_phrase_is_not_a_wall_on_a_good_status():
    """An article about blocking is still an article.

    The marker alone must not convict, or a page discussing access control reads
    as a block and the owner is told to go use an API that does not exist.
    """
    page = "<html><body><h1>Access denied</h1><p>How to debug 403s.</p></body></html>"
    assert bot_wall_detected(page, status_code=200) is False


def test_an_empty_body_is_not_a_wall():
    assert bot_wall_detected("") is False
    assert bot_wall_detected("", status_code=403) is False


def test_a_401_is_left_to_the_reconnect_path():
    """Authentication failures are a different problem with a different fix."""
    assert bot_wall_detected(CLOUDFLARE_BLOCK, status_code=401) is False


""" What the owner is told """


def test_a_vendor_with_a_key_page_is_named_specifically():
    """Naming the working route is the only actionable thing we can say."""
    advice = bot_wall_advice({}, "platform.openai.com")
    assert "platform.openai.com" in advice
    assert "key" in advice.lower()


def test_an_unknown_site_is_told_plainly_that_signing_in_again_will_not_help():
    advice = bot_wall_advice({}, "console.x.ai")
    assert "console.x.ai" in advice
    # The single most useful sentence: do not go round the sign-in loop again.
    assert "signing in again will not help" in advice.lower()
