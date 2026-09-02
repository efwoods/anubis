"""Symmetric encryption for third-party credentials held on a user's behalf.

This is the first encryption-at-rest in the codebase, and it exists because the
connected-account feature is the first thing that must keep a *reusable* secret.
Every credential the platform held before this module was either hashed and
never recovered (the API key, stored as a SHA-256 digest in Auth0
``app_metadata`` and only ever compared) or held in memory for the life of a
request (an Auth0 refresh token, exchanged and cached, never written down). A
mailbox app password is different in kind: the avatar has to present the
original bytes to an IMAP server on a later, unrelated turn, so the plaintext
has to survive in durable storage.

Fernet is used rather than a hand-rolled construction. It is authenticated
encryption (AES-128-CBC with an HMAC-SHA256 tag), so a record tampered with in
the store fails to decrypt instead of yielding attacker-chosen plaintext, and
``cryptography`` is already resolved in the dependency graph by
``python-jose[cryptography]`` — this module adds no new dependency.

Key management is deliberately outside this module: the key arrives through
``GlobalContext.connected_account_encryption_key`` like every other setting.
Rotating it invalidates every stored credential, which surfaces to the owner as
"reconnect your mailbox" rather than as silent corruption, because
:func:`decrypt_secret` raises on a tag mismatch.

The name is generic rather than mailbox-specific on purpose. The browser-session
storage planned for social-account login (``BROWSER_SESSION_ENCRYPTION_KEY``)
and any future OAuth refresh token need exactly this helper, and a second copy
of it would be a second chance to get the construction wrong.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Raised to the caller rather than a bare ValueError so an endpoint can map a
# configuration problem (503, an operator must act) apart from a bad credential
# (400, the user must act).
class SecretEncryptionNotConfiguredError(RuntimeError):
    """The encryption key is missing or is not a usable Fernet key."""


class SecretDecryptionError(RuntimeError):
    """Stored ciphertext could not be decrypted with the configured key."""


def generate_encryption_key() -> str:
    """Return a fresh url-safe base64 Fernet key for an operator to install.

    Offered so setting ``CONNECTED_ACCOUNT_ENCRYPTION_KEY`` does not require an
    operator to find the right one-liner: ``python -c "from
    src.anubis.utils.secret_store import generate_encryption_key;
    print(generate_encryption_key())"``.
    """
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("utf-8")


def _fernet(context: Any) -> Any:
    """Build the Fernet cipher from the configured key.

    The import is function-local per the repository's cold-start rule, and the
    key is read from the passed context rather than the environment so this
    module obeys the "never ``os.environ`` in business logic" convention.
    """
    from cryptography.fernet import Fernet

    key = getattr(context, "connected_account_encryption_key", None)
    if not key or not str(key).strip():
        raise SecretEncryptionNotConfiguredError(
            "CONNECTED_ACCOUNT_ENCRYPTION_KEY is not set, so credentials cannot "
            "be stored. Generate one with "
            "src.anubis.utils.secret_store.generate_encryption_key()."
        )
    try:
        return Fernet(str(key).strip().encode("utf-8"))
    except Exception:
        # The exception text can carry the key material on some versions, so the
        # cause is deliberately not chained into the message the caller sees.
        raise SecretEncryptionNotConfiguredError(
            "CONNECTED_ACCOUNT_ENCRYPTION_KEY is not a valid Fernet key. It must "
            "be 32 url-safe base64-encoded bytes."
        ) from None


def encrypt_secret(plaintext: str, context: Any) -> str:
    """Encrypt one credential for storage.

    Args:
        plaintext: The credential exactly as the user supplied it.
        context: The ``GlobalContext`` carrying the encryption key.

    Returns:
        Url-safe base64 ciphertext, safe to place in a store record.

    Raises:
        SecretEncryptionNotConfiguredError: The key is missing or malformed.
    """
    return _fernet(context).encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(ciphertext: str, context: Any) -> str:
    """Recover one credential written by :func:`encrypt_secret`.

    Raises:
        SecretEncryptionNotConfiguredError: The key is missing or malformed.
        SecretDecryptionError: The ciphertext does not authenticate under the
            configured key — the usual cause is a rotated key, and the honest
            remedy is for the owner to reconnect the account rather than for the
            caller to guess at the plaintext.
    """
    cipher = _fernet(context)
    try:
        return cipher.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception:
        raise SecretDecryptionError(
            "A stored credential could not be decrypted with the configured "
            "CONNECTED_ACCOUNT_ENCRYPTION_KEY. If the key was rotated, the "
            "account must be connected again."
        ) from None
