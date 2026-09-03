"""The agent inbox: triage of incoming messages on the owner's behalf.

- :mod:`repository` — the ``inbox_items``, ``inbox_preferences``, and
  ``inbox_poll_state`` tables (Postgres, with an in-memory twin), published
  process-wide like the other repositories.
- :mod:`triage` — the structured-output classification, the owner-voice draft,
  and the preference-driven confidence score.
- :mod:`poller` — fetching unseen mail from every connected mailbox and running
  the inbox graph once per message.
- :mod:`inbox_tools` — the chat tools the personal avatar uses to report and
  resolve pending items in conversation.

The graph itself lives in ``src/subgraphs/inbox/graph.py``.
"""

from src.anubis.utils.inbox.repository import (
    InMemoryInboxRepository,
    PostgresInboxRepository,
    ensure_inbox_tables,
    get_inbox_repository,
    set_inbox_repository,
)

__all__ = [
    "InMemoryInboxRepository",
    "PostgresInboxRepository",
    "ensure_inbox_tables",
    "get_inbox_repository",
    "set_inbox_repository",
]
