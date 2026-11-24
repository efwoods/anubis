"""Activity review: what the owner did in a period, read from data already stored.

The owner shares the whole screen with the personal avatar, and every watched
snapshot is already kept as a hidden ``ambient_observation`` message on the
thread that was open. The owner's machines already answer for git commits,
Claude Code sessions, and browser history. This package reads those sources
for one period in the owner's own time zone and returns one structured
timeline, so the avatar can answer "what did I do today", "what did I
accomplish this week", and "how was my time spent".

Nothing is written. No row is copied into the store or into a table, and no
model call runs inside the review: the avatar's ordinary turn reads the review
and writes the answer. Every source is independent, and a source that cannot
answer (no machine online, no screen share in the period) is reported as
unavailable while the other sources still answer.
"""
