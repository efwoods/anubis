"""Content subscriptions: what the avatar's person publishes, arriving on its own.

The package is organised by the question each module answers:

``repository``
    Where subscriptions and the events they deliver are stored.
``transports``
    How each platform is asked to push, and how its deliveries are verified.
``payloads``
    How each platform's delivery format becomes one common event shape.
``email_notifications``
    How a platform's notice to the owner's mailbox becomes the same event, for
    the platforms that offer no webhook.
``intake``
    The one door every event goes through: dedupe, ownership, billing, ingest.
``crawl``
    The initial pull — a breadth-first walk of what the account already
    published, pruned to the person it belongs to.

Ownership lives next door in ``connected_accounts/ownership.py`` because it is
a property of the account, not of the subscription.
"""

from src.anubis.utils.subscriptions.intake import (
    ingest_content_url,
    record_content_event,
)
from src.anubis.utils.subscriptions.repository import (
    EVENT_INGESTED,
    EVENT_RECEIVED,
    EVENT_REFUSED,
    SUBSCRIPTION_ACTIVE,
    SUBSCRIPTION_FAILED,
    SUBSCRIPTION_PENDING,
    ensure_subscription_tables,
    get_subscription_repository,
    set_subscription_repository,
)
from src.anubis.utils.subscriptions.transports import (
    callback_url_for,
    subscribe_to_content,
)

__all__ = [
    "EVENT_INGESTED",
    "EVENT_RECEIVED",
    "EVENT_REFUSED",
    "SUBSCRIPTION_ACTIVE",
    "SUBSCRIPTION_FAILED",
    "SUBSCRIPTION_PENDING",
    "callback_url_for",
    "ensure_subscription_tables",
    "get_subscription_repository",
    "ingest_content_url",
    "record_content_event",
    "set_subscription_repository",
    "subscribe_to_content",
]
