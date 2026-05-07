"""Reference connector: local git -> sandbox parquet.

Real implementation lives at src/anubis/utils/tools/git/git_tools.py.
"""

from __future__ import annotations

import io
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langchain.tools import tool

GIT_LOG_FORMAT = "%H%x09%an%x09%ae%x09%at%x09%s"


def _repos(root: Path) -> list[Path]:
    return [p.parent for p in root.rglob(".git") if p.is_dir()]


def _git_log(repo: Path, since: datetime) -> pd.DataFrame:
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            f"--since={since.isoformat()}",
            f"--pretty=format:{GIT_LOG_FORMAT}",
            "--shortstat",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return pd.DataFrame()

    rows: list[dict] = []
    pending: dict | None = None
    for line in out.stdout.splitlines():
        if "\t" in line:
            if pending is not None:
                rows.append(pending)
            sha, name, email, ts, subject = line.split("\t", 4)
            pending = {
                "sha": sha,
                "repo": repo.name,
                "author_name": name,
                "author_email": email,
                "committed_at": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
                "message": subject,
                "files_changed": 0,
                "insertions": 0,
                "deletions": 0,
            }
        elif "file" in line and pending is not None:
            for token in line.split(","):
                token = token.strip()
                if "file" in token:
                    pending["files_changed"] = int(token.split()[0])
                elif "insertion" in token:
                    pending["insertions"] = int(token.split()[0])
                elif "deletion" in token:
                    pending["deletions"] = int(token.split()[0])
    if pending is not None:
        rows.append(pending)
    return pd.DataFrame(rows)


def make_tool(backend):
    @tool(parse_docstring=True)
    async def ingest_git_local(days: int = 7) -> str:
        """Walk GIT_LOCAL_REPO_ROOT, run git log on each repo, write parquet.

        Args:
            days: Lookback window in days.
        """
        from src.anubis.utils.context import GlobalContext

        context = GlobalContext()
        root = Path(context.git_local_repo_root or "/home/user/gh")
        since = datetime.now(tz=timezone.utc) - timedelta(days=int(days))
        frames = [_git_log(repo, since) for repo in _repos(root)]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()

        run_id = uuid4().hex[:8]
        path = f"/data/git/clean/{datetime.utcnow().date().isoformat()}__{run_id}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        await backend.upload_files([(path, buf.getvalue())])
        return path

    return ingest_git_local
