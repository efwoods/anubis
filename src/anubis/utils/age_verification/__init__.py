"""Age verification for adult-only avatar discovery.

Adult-only avatars stay out of search until the signed-in account confirms
a date of birth that meets ``AGE_VERIFICATION_MINIMUM_YEARS``, except for
the platform administrator, who still sees those avatars in search. Boot-time
entry points for the FastAPI lifespan: ``ensure_age_verification_table``
creates the table and ``publish_age_verification_repository`` binds the
repository to the pool.
"""

from __future__ import annotations

from typing import Any


async def ensure_age_verification_table(pool: Any) -> None:
    """Create the age verification table when absent."""
    from src.anubis.utils.age_verification.repository import (
        ensure_age_verification_table as _ensure,
    )

    await _ensure(pool)


def publish_age_verification_repository(pool: Any) -> Any:
    """Bind a Postgres repository to ``pool`` and publish the repository."""
    from src.anubis.utils.age_verification.repository import (
        PostgresAgeVerificationRepository,
        set_age_verification_repository,
    )

    repository = PostgresAgeVerificationRepository(pool)
    set_age_verification_repository(repository)
    return repository


def get_age_verification_repository() -> Any | None:
    """Return the published repository, or None before the lifespan ran."""
    from src.anubis.utils.age_verification.repository import (
        get_age_verification_repository as _get,
    )

    return _get()
