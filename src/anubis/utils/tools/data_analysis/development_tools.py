"""LangChain tools that read the owner's development activity from their machines.

The Neural Nexus daemon (``anubis-mcp-server-ubuntu-desktop``,
``src/server/dev_tools.py``) exposes read-only git and Claude Code session
tools next to the file tools. The wrappers below make those tools callable by
the personal avatar's deep agent so the owner can ask "what did I ship since
Monday", "what is still in progress", and "how long did the billing feature
take" and receive an answer grounded in commits and coding sessions rather
than in the avatar's memory of the conversation.

Every wrapper follows the multi-machine shape of ``analysis_tools``: an
optional ``device_label`` names one machine, and omitting the label fans the
call out to every live machine concurrently. Results are merged into one
``rows`` list where every row carries a ``device_label`` field, so the avatar
can always say which machine a commit or session came from. A machine that
did not answer is listed under ``unreached`` with the reason instead of
failing the whole call.

The daemon tools themselves are not re-implemented here: this module only
selects machines, forwards arguments, and merges results, reusing the
selection, timeout, and single-device call helpers from ``analysis_tools``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.tools import tool

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis.analysis_tools import (
    _call_one_device,
    _device_fanout_timeout_seconds,
    _select_devices,
)
from src.anubis.utils.tools.data_analysis.discovery import McpConnection

# Daemon tool names, spelled once so the wrappers and the tests agree.
DAEMON_LIST_GIT_REPOSITORIES = "list_git_repositories"
DAEMON_GIT_LOG = "git_log"
DAEMON_GIT_DIFF_STAT = "git_diff_stat"
DAEMON_GIT_STATUS = "git_status"
DAEMON_LIST_CLAUDE_CODE_SESSIONS = "list_claude_code_sessions"
DAEMON_READ_CLAUDE_CODE_SESSION = "read_claude_code_session"

DEVELOPMENT_TOOL_NAMES = (
    DAEMON_LIST_GIT_REPOSITORIES,
    DAEMON_GIT_LOG,
    DAEMON_GIT_DIFF_STAT,
    DAEMON_GIT_STATUS,
    DAEMON_LIST_CLAUDE_CODE_SESSIONS,
    DAEMON_READ_CLAUDE_CODE_SESSION,
)


def _rows_from_device_result(device_label: str, result: Any) -> list[dict[str, Any]]:
    """Turn one machine's raw tool result into rows tagged with the machine name.

    A list result (commits, repositories, sessions) becomes one row per entry;
    a mapping result (a status, a diff summary, a transcript) becomes a single
    row. Anything else is wrapped so the shape stays uniform for the model.
    """
    if isinstance(result, list):
        rows: list[dict[str, Any]] = []
        for entry in result:
            if isinstance(entry, dict):
                rows.append({"device_label": device_label, **entry})
            else:
                rows.append({"device_label": device_label, "value": entry})
        return rows
    if isinstance(result, dict):
        return [{"device_label": device_label, **result}]
    return [{"device_label": device_label, "value": result}]


def _sort_rows(
    rows: list[dict[str, Any]], key: str | None, *, descending: bool
) -> None:
    """Order merged rows by ``key`` in place; rows lacking the key go last."""
    if key is None:
        return
    with_key = [row for row in rows if row.get(key) is not None]
    without_key = [row for row in rows if row.get(key) is None]
    with_key.sort(key=lambda row: str(row[key]), reverse=descending)
    rows[:] = with_key + without_key


async def fan_out_development_tool(
    connections: list[McpConnection],
    device_label: str | None,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float,
    *,
    sort_key: str | None = None,
    sort_descending: bool = True,
) -> dict[str, Any]:
    """Call one daemon tool on the selected machines and merge the results.

    Returns ``{"rows": [...], "unreached": [...]}``. Every row carries
    ``device_label``; ``unreached`` holds ``{"device_label", "detail"}`` for
    each machine that did not answer (asleep, or the tool refused the
    arguments — for example a repository path that machine does not hold).
    When ``device_label`` names no connected machine the error mapping from
    ``_select_devices`` is returned unchanged so the model can correct the
    name.
    """
    selected, error = _select_devices(connections, device_label)
    if error is not None:
        return error

    async def call_one(connection: McpConnection) -> tuple[str, Any]:
        result = await _call_one_device(
            connection, tool_name, arguments, timeout_seconds
        )
        return connection.device_label, result

    outcomes = await asyncio.gather(*(call_one(connection) for connection in selected))
    rows: list[dict[str, Any]] = []
    unreached: list[dict[str, Any]] = []
    for label, result in outcomes:
        if isinstance(result, dict) and result.get("status") == "offline":
            unreached.append({"device_label": label, "detail": result.get("detail")})
            continue
        rows.extend(_rows_from_device_result(label, result))
    _sort_rows(rows, sort_key, descending=sort_descending)
    return {"rows": rows, "unreached": unreached}


def build_development_tools(
    context: GlobalContext, connections: list[McpConnection]
) -> list[Any]:
    """Build the per-turn development-analytics tool set across every live machine.

    ``connections`` are the live, avatar-bound machines for this turn (the same
    list ``build_data_analysis_tools`` receives). Each tool forwards to the
    daemon tool of the same name and merges the answers; see the module
    docstring for the fan-out shape.
    """
    fanout_timeout_seconds = _device_fanout_timeout_seconds(context)

    async def _fan_out(
        device_label: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        sort_key: str | None = None,
        sort_descending: bool = True,
    ) -> dict[str, Any]:
        return await fan_out_development_tool(
            connections,
            device_label,
            tool_name,
            arguments,
            fanout_timeout_seconds,
            sort_key=sort_key,
            sort_descending=sort_descending,
        )

    @tool
    async def list_git_repositories(
        root: str = "",
        max_depth: int = 3,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """List the git repositories on the owner's machines.

        Call this tool first whenever a question concerns code, commits,
        projects, or "what am I working on": the rows give the exact
        repository path to pass to git_log, git_diff_stat, and git_status.
        Each row carries device_label, path, name, branch, last_commit_at, and
        remote_url. Machines that did not answer are listed under unreached.

        Args:
            root: Optional absolute directory on the machine to search inside;
                omit to search every shared folder and configured repository
                root.
            max_depth: How many directory levels below each root to search.
            device_label: Optional name of a single machine, for example
                "Ubuntu". Omit to search every connected machine.
        """
        return await _fan_out(
            device_label,
            DAEMON_LIST_GIT_REPOSITORIES,
            {"root": root, "max_depth": max_depth},
            sort_key="path",
            sort_descending=False,
        )

    @tool
    async def git_log(
        repository: str,
        since: str | None = None,
        until: str | None = None,
        max_commits: int = 200,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """Read the commit history of one repository, newest first.

        Call this tool to answer "what happened since <date>", "what did I
        ship this week", "what changed in <project>", or to count how many
        commits a piece of work took. Each row carries device_label, sha,
        author, authored_at, subject, body, files_changed, insertions,
        deletions, and files. Sum insertions and deletions to describe the
        size of the work; group by authored_at to describe the pace.

        Args:
            repository: Absolute path of the repository on the machine, as
                returned by list_git_repositories.
            since: Optional lower bound in any git date form, for example
                "2026-09-01" or "2 weeks ago".
            until: Optional upper bound in any git date form.
            max_commits: Maximum number of commits to return per machine.
            device_label: Optional name of the machine holding the
                repository. Omit to ask every connected machine; machines
                without that repository appear under unreached.
        """
        return await _fan_out(
            device_label,
            DAEMON_GIT_LOG,
            {
                "repository": repository,
                "since": since,
                "until": until,
                "max_commits": max_commits,
            },
            sort_key="authored_at",
            sort_descending=True,
        )

    @tool
    async def git_diff_stat(
        repository: str,
        from_ref: str,
        to_ref: str = "HEAD",
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """Summarize how much changed between two git references of one repository.

        Call this tool to size a body of work between two points — a release
        tag and HEAD, a branch and main, or two commit shas from git_log —
        when the question is "how big was <feature>" or "what changed between
        <a> and <b>". Each row carries device_label, from_ref, to_ref,
        files_changed, insertions, deletions, and the per-file counts.

        Args:
            repository: Absolute path of the repository on the machine.
            from_ref: The older reference (commit sha, tag, or branch).
            to_ref: The newer reference; defaults to HEAD.
            device_label: Optional name of the machine holding the repository.
        """
        return await _fan_out(
            device_label,
            DAEMON_GIT_DIFF_STAT,
            {"repository": repository, "from_ref": from_ref, "to_ref": to_ref},
        )

    @tool
    async def git_status(
        repository: str,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """Read the uncommitted state of one repository.

        Call this tool to answer "what is in progress", "what have I not
        committed yet", or "is <project> pushed". Each row carries
        device_label, branch, upstream, ahead, behind, staged, modified, and
        untracked. Work is in progress when staged, modified, or untracked is
        non-empty or ahead is greater than zero.

        Args:
            repository: Absolute path of the repository on the machine.
            device_label: Optional name of the machine holding the repository.
        """
        return await _fan_out(
            device_label, DAEMON_GIT_STATUS, {"repository": repository}
        )

    @tool
    async def list_claude_code_sessions(
        since: str | None = None,
        until: str | None = None,
        max_sessions: int = 200,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """List the owner's Claude Code coding sessions, newest first.

        Call this tool to answer "how long did <feature> take", "what was I
        working on <day>", "how many hours did I code this week", or to pair
        coding time with the commits from git_log (a session belongs to a
        repository when the session cwd lies inside the repository path).
        Each row carries device_label, session_id, cwd, git_branch,
        started_at, ended_at, duration_minutes, first_prompt, message_count,
        tool_use_counts, and summary. A row with disabled=true means the owner
        switched session sharing off on that machine; say so and stop.

        Args:
            since: Optional ISO-8601 lower bound compared with the session
                start, for example "2026-09-01".
            until: Optional ISO-8601 upper bound compared with the session
                start.
            max_sessions: Maximum number of sessions to return per machine.
            device_label: Optional name of a single machine. Omit to list
                sessions from every connected machine.
        """
        return await _fan_out(
            device_label,
            DAEMON_LIST_CLAUDE_CODE_SESSIONS,
            {"since": since, "until": until, "max_sessions": max_sessions},
            sort_key="started_at",
            sort_descending=True,
        )

    @tool
    async def read_claude_code_session(
        session_id: str,
        max_characters: int = 20000,
        device_label: str | None = None,
    ) -> dict[str, Any]:
        """Read the conversation text of one Claude Code session.

        Call this tool after list_claude_code_sessions when the first prompt
        or summary is not enough to say what a session accomplished — for
        example to name the feature a session worked on, or to explain why a
        piece of work took as long as the session shows. Each row carries
        device_label, session_id, text (user and assistant turns only; tool
        calls and credential lines are removed), and truncated.

        Args:
            session_id: The session_id from list_claude_code_sessions.
            max_characters: Upper bound on the returned text length.
            device_label: Optional name of the machine holding the session.
                Omit to ask every connected machine; machines without that
                session appear under unreached.
        """
        return await _fan_out(
            device_label,
            DAEMON_READ_CLAUDE_CODE_SESSION,
            {"session_id": session_id, "max_characters": max_characters},
        )

    return [
        list_git_repositories,
        git_log,
        git_diff_stat,
        git_status,
        list_claude_code_sessions,
        read_claude_code_session,
    ]
