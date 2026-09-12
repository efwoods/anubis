"""The avatar taking part in group conversations on Slack, Discord, and Twitch.

One triage brain serves every platform. A bot posts a batch of messages to
``POST /groups/{assistant_id}/events`` and receives one decision per message —
ignore, respond, notify, or moderate — which the bot then carries out itself.

* ``events`` — the wire format every bot speaks.
* ``triage`` — the decision, with the owner's rules and past decisions as precedent.
* ``precedent`` — what the avatar has learned, in the cross-thread store.
* ``runner`` — one graph run per message, on a durable thread.
* ``group_tools`` — the same thing, run from chat.
"""
