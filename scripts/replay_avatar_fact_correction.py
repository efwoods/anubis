#!/usr/bin/env python
r"""Replay an owner's correction of an avatar fact through ``edit_identity_fact``'s code path.

An owner's correction that an earlier build filed in the wrong place (a user
preference, a fact about the user, or a new fact stored beside the wrong one) left the
wrong value in the avatar's stored facts, where every viewer still read the wrong value.
This script runs the same ``find_fact_matches`` -> per-match suggestion ->
``apply_resolved_corrections`` steps the tool runs, without the chat approval panel:
the dry run (the default) prints every match and the suggested edit, and ``--apply``
writes only the matches whose suggestion says the stored text states the wrong claim.

Every document the script changes or deletes is first written to ``--backup-path`` as
JSON, so a replay can be undone by putting those values back.

Examples (inside the API container, where ``POSTGRES_URI`` resolves):

    docker exec anubis-langgraph-api-prod-1 python scripts/replay_avatar_fact_correction.py \\
        --owner-user-id 6a64d1ef4e063740350632ae \\
        --assistant-id 47cfdaa2-1196-4519-9127-31cb13ff9d3a \\
        --inaccurate "linkedin.com/in/evanwoods" \\
        --corrected "My LinkedIn profile is https://www.linkedin.com/in/evanfwoods." \\
        --context "I was told my LinkedIn profile is linkedin.com/in/evanfwoods."

    # Remove copies the wrong tools wrote, after reading each one in the dry run:
    ... --delete "6a64d1ef4e063740350632ae/47cfdaa2-.../preference=eef3fde4-..." --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments of the replay."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--owner-user-id", required=True)
    parser.add_argument("--assistant-id", required=True)
    parser.add_argument("--inaccurate", default="", help="The wrong claim, as the stored fact states the claim.")
    parser.add_argument("--corrected", default="", help="The corrected fact in the first person.")
    parser.add_argument("--context", default="", help="Context recorded with the corrected fact.")
    parser.add_argument(
        "--delete",
        action="append",
        default=[],
        help="namespace/parts/joined/by/slash=key of a document to delete (repeatable).",
    )
    parser.add_argument("--backup-path", default="/tmp/avatar_fact_correction_backup.json")
    parser.add_argument("--apply", action="store_true", help="Write the changes (default: dry run).")
    return parser.parse_args(argv)


def _document_text(item) -> str:
    kwargs = ((getattr(item, "value", None) or {}).get("document") or {}).get("kwargs") or {}
    return str(kwargs.get("page_content") or "")


async def replay(arguments: argparse.Namespace) -> int:
    """Print every match and suggested edit; with ``--apply``, back up and write the edits."""
    from langgraph.store.base import IndexConfig
    from langgraph.store.postgres import AsyncPostgresStore

    from src.anubis.utils.context import GlobalContext
    from src.anubis.utils.tools.identity import identity_tools

    context = GlobalContext()
    backup_documents: list[dict] = []

    async with AsyncPostgresStore.from_conn_string(
        context.async_postgres_store_uri,
        index=IndexConfig(
            dims=640,
            embed="huggingface:" + context.embedding_model,
            fields=["document.kwargs.page_content"],
        ),
    ) as store:
        resolved_corrections: list = []
        if arguments.inaccurate:
            matches = await identity_tools.find_fact_matches(
                store,
                creator_id=arguments.owner_user_id,
                assistant_id=arguments.assistant_id,
                user_id=arguments.owner_user_id,
                query=arguments.inaccurate,
            )
            print(f"{len(matches)} match(es) for {arguments.inaccurate!r}")
            for match in matches:
                suggest = (
                    identity_tools._suggest_source_url_correction
                    if match.kind == "source_url"
                    else identity_tools._suggest_correction
                )
                suggestion = identity_tools._suggest_restated_url_correction(
                    match,
                    inaccurate_information=arguments.inaccurate,
                    corrected_information=arguments.corrected,
                    correction_context=arguments.context,
                    is_deletion=not arguments.corrected,
                ) or await suggest(
                    match,
                    inaccurate_information=arguments.inaccurate,
                    corrected_information=arguments.corrected,
                    correction_context=arguments.context,
                    is_deletion=not arguments.corrected,
                )
                corrected_text, corrected_context, asserts_inaccurate_fact = suggestion
                include = bool(asserts_inaccurate_fact) and (
                    bool(corrected_text.strip()) or not arguments.corrected
                )
                print(
                    json.dumps(
                        {
                            "include": include,
                            "kind": match.kind,
                            "namespace": list(match.namespace),
                            "key": match.key,
                            "score": round(match.score, 4),
                            "matched_text": (match.matched_text or "")[:400],
                            "suggested_text": corrected_text[:400],
                        },
                        ensure_ascii=False,
                    )
                )
                if include:
                    resolved_corrections.append(
                        identity_tools.ResolvedCorrection(
                            match=match,
                            corrected_text=corrected_text,
                            corrected_context=corrected_context,
                            include=True,
                        )
                    )
                    backup_documents.append(
                        {"namespace": list(match.namespace), "key": match.key, "value": match.item.value}
                    )

        deletions: list[tuple[tuple, str]] = []
        for delete_argument in arguments.delete:
            namespace_text, _, key = delete_argument.partition("=")
            namespace = tuple(part for part in namespace_text.split("/") if part)
            item = await store.aget(namespace, key)
            if item is None:
                print(f"delete: {namespace}/{key} not found")
                continue
            print(f"delete: {namespace}/{key}: {_document_text(item)[:300]!r}")
            deletions.append((namespace, key))
            backup_documents.append({"namespace": list(namespace), "key": key, "value": item.value})

        if not arguments.apply:
            print("\ndry run: nothing written.")
            return 0

        Path(arguments.backup_path).write_text(json.dumps(backup_documents, ensure_ascii=False, indent=2))
        print(f"backed up {len(backup_documents)} document(s) to {arguments.backup_path}")
        changes = await identity_tools.apply_resolved_corrections(store, resolved_corrections)
        for change in changes:
            print(f"{change.action}: {list(change.namespace)}/{change.key}: {(change.new_text or '')[:200]!r}")
        for namespace, key in deletions:
            await store.adelete(namespace, key)
            print(f"deleted: {list(namespace)}/{key}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay and return the process exit code."""
    return asyncio.run(replay(parse_arguments(argv)))


if __name__ == "__main__":
    sys.exit(main())
