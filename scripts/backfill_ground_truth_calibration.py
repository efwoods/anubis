#!/usr/bin/env python
"""Fit an avatar's direct-quote cloud from quotes that are already in the store.

An avatar's replies are scored against a cloud fitted from the target's own direct
quotes: an empirical Mahalanobis threshold plus an IsolationForest, both derived
from the ``quote`` namespace and read back by the message path to produce
``comparison_to_direct_quote_response_analysis``.

That fit used to be triggered from only two ingestion branches — quotes-per-line
text and CSV/JSON statements. Uploads that arrive as audio or video (every YouTube
link is downloaded and diarized) wrote their quotes correctly and never fitted
anything, so those avatars accumulated a full direct-quote corpus while the message
path silently reported no direct-quote comparison at all. New uploads now fit once
per batch, but avatars ingested before that fix stay uncalibrated until something
refits them from the quotes already stored. That is what this script does.

Examples:
    # One avatar.
    python scripts/backfill_ground_truth_calibration.py \
        --assistant-id 0f92b031-9ab6-4c02-9ca7-b7527ebb3238

    # Every avatar that has enough quotes but no fitted model, listed only.
    python scripts/backfill_ground_truth_calibration.py --all --dry-run

The store URI committed to ``.env`` names ``host.docker.internal``, which resolves
inside the compose network but not on the host, so the usual invocation is:

    docker compose exec langgraph-api-dev \
        python scripts/backfill_ground_truth_calibration.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

# Allow running as a plain script from the repo root without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anubis.utils.context import GlobalContext  # noqa: E402

# The five store rows a completed calibration leaves behind, in the order the
# summary prints them. The first three are what the message path's gate requires;
# the last two feed the key_phrase_rate feature and the <STYLE> prompt block.
CALIBRATION_ARTIFACT_KEYS: Tuple[str, ...] = (
    "ground_truth_text_features_by_doc_id_dict_str",
    "ground_truth_text_empirical_threshold_list_str",
    "ground_truth_text_features_model_b64_pkl",
    "key_phrase_profile",
    "style_profile",
)

# The row whose absence means "this avatar was never successfully calibrated".
# Used to pick candidates for --all: the per-document feature dict is written even
# when the corpus is below the calibration floor, so keying on the dict would
# re-select avatars that are correctly deferred. The fitted model is only ever
# written once a real fit completed.
FITTED_MODEL_KEY = "ground_truth_text_features_model_b64_pkl"


class StoreUnreachableError(RuntimeError):
    """The Postgres store backing the LangGraph namespaces could not be reached."""


def redact_credentials(store_uri: str) -> str:
    """Strip user:password from a connection URI so it is safe to print."""
    return re.sub(r"://[^@/]*@", "://***:***@", store_uri)


def unreachable_store_message(store_uri: str, cause: Exception) -> str:
    """Explain an unreachable store, naming the host that most often causes it.

    The committed ``ASYNC_POSTGRES_STORE_URI`` points at ``host.docker.internal``.
    That name is resolvable from inside the compose network and NOT from the host,
    so running this script on the host against the committed configuration fails
    at connect time with a DNS error that does not obviously say so.
    """
    lines = [
        f"Cannot reach the store at {redact_credentials(store_uri)}: {cause}",
    ]
    if "host.docker.internal" in store_uri:
        lines.append(
            "The URI names 'host.docker.internal', which resolves inside the compose "
            "network but not on the host."
        )
    lines.append(
        "Either run this script inside the API container "
        "(docker compose exec langgraph-api-dev python scripts/"
        "backfill_ground_truth_calibration.py ...) or repoint "
        "ASYNC_POSTGRES_STORE_URI at a host-reachable address such as "
        "localhost:5432."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Owner resolution
# ---------------------------------------------------------------------------
# Every per-avatar artifact is stored under a namespace whose FIRST element is the
# avatar owner's user id, not the id of whoever is conversing. Writing under the
# wrong first element produces rows the message path will never read, so this
# resolves the owner from recorded state and refuses to guess.

SQL_CREATOR_ID_MARKER = """
SELECT value->>'value'
  FROM store
 WHERE trim(split_part(prefix, '.', 1)) = %s
   AND trim(split_part(prefix, '.', 2)) = 'creator_id'
   AND key = 'creator_id'
 LIMIT 1;
"""

SQL_QUOTE_PREFIX_OWNERS = """
SELECT DISTINCT trim(split_part(prefix, '.', 1))
  FROM store
 WHERE trim(split_part(prefix, '.', 2)) = %s
   AND trim(split_part(prefix, '.', 3)) = 'quote';
"""

SQL_QUOTE_ROW_COUNT = """
SELECT count(*)
  FROM store
 WHERE trim(split_part(prefix, '.', 1)) = %s
   AND trim(split_part(prefix, '.', 2)) = %s
   AND trim(split_part(prefix, '.', 3)) = 'quote';
"""

SQL_PRESENT_ARTIFACT_KEYS = """
SELECT key, length(value::text)
  FROM store
 WHERE trim(split_part(prefix, '.', 1)) = %s
   AND trim(split_part(prefix, '.', 2)) = %s
   AND key = ANY(%s);
"""

# Candidates for --all: an avatar with a quote corpus at or above the floor and no
# fitted model. Grouping on the first three dot-segments reconstructs the
# (owner, assistant) pair straight from the namespace, which is the same identity
# the writer uses, so no join against the assistant table is needed.
SQL_UNCALIBRATED_CANDIDATES = """
SELECT quote_corpus.owner_user_id,
       quote_corpus.assistant_id,
       quote_corpus.quote_row_count
  FROM (
    SELECT trim(split_part(prefix, '.', 1)) AS owner_user_id,
           trim(split_part(prefix, '.', 2)) AS assistant_id,
           count(*)                         AS quote_row_count
      FROM store
     WHERE trim(split_part(prefix, '.', 3)) = 'quote'
     GROUP BY 1, 2
  ) AS quote_corpus
 WHERE quote_corpus.quote_row_count >= %s
   AND NOT EXISTS (
     SELECT 1
       FROM store AS fitted_model
      WHERE fitted_model.key = %s
        AND trim(split_part(fitted_model.prefix, '.', 1)) = quote_corpus.owner_user_id
        AND trim(split_part(fitted_model.prefix, '.', 2)) = quote_corpus.assistant_id
   )
 ORDER BY quote_corpus.quote_row_count DESC;
"""


def resolve_owner_user_id(connection, assistant_id: str) -> str:
    """Return the owner user id for an avatar, or raise rather than guess.

    Two independent sources, in order of authority:

    1. The ``creator_id`` marker written at avatar creation. This is the value the
       API itself treats as the owner, so it is the same id the message path will
       use when it reads the calibrated rows back.
    2. The first namespace segment of the avatar's own quote rows. Whatever wrote
       those quotes already committed to an owner id, and the calibration must be
       stored under the same one to be found.

    An avatar whose quote rows disagree about the owner is ambiguous; guessing
    would write a cloud into a namespace nothing reads, so this raises instead.
    """
    with connection.cursor() as cursor:
        cursor.execute(SQL_CREATOR_ID_MARKER, (assistant_id,))
        row = cursor.fetchone()
        if row and row[0]:
            return str(row[0]).strip()

        cursor.execute(SQL_QUOTE_PREFIX_OWNERS, (assistant_id,))
        owner_user_ids = sorted({str(r[0]).strip() for r in cursor.fetchall() if r[0]})

    if not owner_user_ids:
        raise ValueError(
            f"No creator_id marker and no quote rows found for assistant "
            f"{assistant_id}; pass --user-id explicitly if you know the owner."
        )
    if len(owner_user_ids) > 1:
        raise ValueError(
            f"Quote rows for assistant {assistant_id} name more than one owner "
            f"({', '.join(owner_user_ids)}); refusing to guess. Pass --user-id."
        )
    return owner_user_ids[0]


def count_quote_rows(connection, owner_user_id: str, assistant_id: str) -> int:
    """Return how many direct-quote rows this avatar has in the store."""
    with connection.cursor() as cursor:
        cursor.execute(SQL_QUOTE_ROW_COUNT, (owner_user_id, assistant_id))
        row = cursor.fetchone()
    return int(row[0]) if row else 0


def present_artifact_sizes(
    connection, owner_user_id: str, assistant_id: str
) -> Dict[str, int]:
    """Return ``{artifact key: serialized byte length}`` for the rows that exist."""
    with connection.cursor() as cursor:
        cursor.execute(
            SQL_PRESENT_ARTIFACT_KEYS,
            (owner_user_id, assistant_id, list(CALIBRATION_ARTIFACT_KEYS)),
        )
        return {str(key): int(size or 0) for key, size in cursor.fetchall()}


def select_uncalibrated_candidates(
    connection, minimum_quote_rows: int
) -> List[Tuple[str, str, int]]:
    """Return ``(owner_user_id, assistant_id, quote_row_count)`` needing a fit."""
    with connection.cursor() as cursor:
        cursor.execute(
            SQL_UNCALIBRATED_CANDIDATES, (minimum_quote_rows, FITTED_MODEL_KEY)
        )
        return [
            (str(owner).strip(), str(assistant).strip(), int(count))
            for owner, assistant, count in cursor.fetchall()
        ]


def open_enumeration_connection(store_uri: str):
    """Connect with plain psycopg for the read-only SQL above.

    Deliberately NOT an ``AsyncPostgresStore``: enumeration and verification are
    pure SQL over the ``store`` table and need none of the vector machinery, so
    building a store here would load the 640-dimension embedding model just to run
    a few counting queries.
    """
    import psycopg

    try:
        return psycopg.connect(store_uri, autocommit=True, connect_timeout=15)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        raise StoreUnreachableError(unreachable_store_message(store_uri, exc)) from exc


async def calibrate_one_avatar(store, owner_user_id: str, assistant_id: str) -> None:
    """Fit one avatar's direct-quote cloud from the quotes already in the store."""
    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        calibrate_ground_truth_from_stored_corpus,
    )

    await calibrate_ground_truth_from_stored_corpus(
        store=store, assistant_id=assistant_id, user_id=owner_user_id
    )


async def run_backfill(
    targets: Sequence[Tuple[str, str, int]],
    store_uri: str,
    context: GlobalContext,
    connection,
) -> int:
    """Calibrate every target, reporting per avatar. Returns the failure count.

    A real store IS needed here (unlike enumeration): the corpus read inside
    ``calibrate_ground_truth`` is a semantic search, which embeds its query, so the
    same IndexConfig the API uses must be in place or the read comes back empty.
    """
    from langgraph.store.base import IndexConfig
    from langgraph.store.postgres import AsyncPostgresStore

    failure_count = 0
    async with AsyncPostgresStore.from_conn_string(
        store_uri,
        index=IndexConfig(
            dims=640,
            embed="huggingface:" + context.embedding_model,
            fields=["document.kwargs.page_content"],
        ),
    ) as store:
        await store.setup()
        for index, (owner_user_id, assistant_id, quote_row_count) in enumerate(
            targets, start=1
        ):
            print(
                f"\n[{index}/{len(targets)}] assistant={assistant_id} "
                f"owner={owner_user_id} quote_rows={quote_row_count}"
            )
            try:
                await calibrate_one_avatar(store, owner_user_id, assistant_id)
            except Exception as exc:  # noqa: BLE001 - one avatar must not abort the fleet
                failure_count += 1
                print(f"  FAILED: {exc}")
                continue

            written = present_artifact_sizes(connection, owner_user_id, assistant_id)
            for artifact_key in CALIBRATION_ARTIFACT_KEYS:
                if artifact_key in written:
                    print(f"  wrote   {artifact_key} ({written[artifact_key]} bytes)")
                else:
                    print(f"  ABSENT  {artifact_key}")
            # The first three are the message path's gate. Their absence after a
            # successful call is the designed deferral, not a failure: the corpus
            # was below the calibration floor once unusable rows were dropped.
            if FITTED_MODEL_KEY not in written:
                print(
                    "  NOTE: no fitted model written — the usable corpus was below "
                    "the calibration floor. The direct-quote comparison stays off "
                    "for this avatar until more quotes are ingested."
                )
    return failure_count


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line; ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit an avatar's direct-quote cloud from quotes already in the store."
        )
    )
    parser.add_argument(
        "--assistant-id",
        action="append",
        default=[],
        help="Avatar to calibrate. Repeatable. Mutually exclusive with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Calibrate every avatar that has at least --min-quote-rows direct "
            "quotes and no fitted model."
        ),
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help=(
            "Owner user id, overriding resolution from the creator_id marker and "
            "the quote namespaces. Only valid with a single --assistant-id."
        ),
    )
    parser.add_argument(
        "--min-quote-rows",
        type=int,
        default=None,
        help=(
            "Minimum direct-quote rows for --all selection. Defaults to the "
            "calibration floor the writer itself enforces."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be calibrated and exit without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backfill and return the process exit code."""
    arguments = parse_arguments(argv)

    if arguments.all and arguments.assistant_id:
        print("Pass either --all or --assistant-id, not both.", file=sys.stderr)
        return 2
    if not arguments.all and not arguments.assistant_id:
        print("Pass --all or at least one --assistant-id.", file=sys.stderr)
        return 2
    if arguments.user_id and len(arguments.assistant_id) != 1:
        print("--user-id applies to exactly one --assistant-id.", file=sys.stderr)
        return 2

    context = GlobalContext()
    store_uri = context.async_postgres_store_uri
    if not store_uri:
        print("ASYNC_POSTGRES_STORE_URI is not set.", file=sys.stderr)
        return 2

    from src.subgraphs.process_media_graph.utils.calibrate_ground_truth import (
        MIN_ROWS_FOR_CALIBRATION,
    )

    minimum_quote_rows = (
        arguments.min_quote_rows
        if arguments.min_quote_rows is not None
        else MIN_ROWS_FOR_CALIBRATION
    )

    print(f"Store: {redact_credentials(store_uri)}")

    try:
        connection = open_enumeration_connection(store_uri)
    except StoreUnreachableError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        if arguments.all:
            targets = select_uncalibrated_candidates(connection, minimum_quote_rows)
        else:
            targets = []
            for assistant_id in arguments.assistant_id:
                try:
                    owner_user_id = (
                        arguments.user_id
                        or resolve_owner_user_id(connection, assistant_id)
                    )
                except ValueError as exc:
                    print(f"{assistant_id}: {exc}", file=sys.stderr)
                    return 1
                targets.append(
                    (
                        owner_user_id,
                        assistant_id,
                        count_quote_rows(connection, owner_user_id, assistant_id),
                    )
                )

        if not targets:
            print("Nothing to calibrate.")
            return 0

        print(f"\n{len(targets)} avatar(s) to calibrate:")
        for owner_user_id, assistant_id, quote_row_count in targets:
            print(
                f"  {assistant_id}  owner={owner_user_id}  "
                f"quote_rows={quote_row_count}"
            )

        if arguments.dry_run:
            print("\n--dry-run: nothing written.")
            return 0

        failure_count = asyncio.run(
            run_backfill(targets, store_uri, context, connection)
        )
    finally:
        connection.close()

    if failure_count:
        print(
            f"\n{failure_count} of {len(targets)} avatar(s) failed to calibrate.",
            file=sys.stderr,
        )
        return 1
    print(f"\nCalibrated {len(targets)} avatar(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
