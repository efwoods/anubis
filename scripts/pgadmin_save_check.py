#!/usr/bin/env python3
# region agent log
"""Verify pgAdmin save blocker hypotheses (NDJSON). Session 391447."""
import json
import sqlite3
import time
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-391447.log"
SYS_CFG = Path("/etc/pgadmin/config_system.py")


def log(hid: str, msg: str, data: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "sessionId": "391447",
                    "hypothesisId": hid,
                    "location": "scripts/pgadmin_save_check.py",
                    "message": msg,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                    "runId": "save-diagnose",
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    db = Path.home() / ".pgadmin" / "pgadmin4.db"
    n_servers = None
    if db.is_file():
        try:
            n_servers = sqlite3.connect(db).execute("SELECT COUNT(*) FROM server").fetchone()[0]
        except Exception as e:
            n_servers = f"err:{e}"

    mp_disabled = None
    cfg_txt = ""
    if SYS_CFG.is_file():
        cfg_txt = SYS_CFG.read_text(encoding="utf-8", errors="replace")
        mp_disabled = "MASTER_PASSWORD_REQUIRED" in cfg_txt and "False" in cfg_txt.split(
            "MASTER_PASSWORD_REQUIRED", 1
        )[-1][:80]

    log("H1", "Saved servers count (0 => never persisted)", {"count": n_servers, "db_exists": db.is_file()})
    log(
        "H2",
        "System override disables master password requirement",
        {
            "config_system_exists": SYS_CFG.is_file(),
            "looks_like_master_password_false": mp_disabled,
        },
    )


if __name__ == "__main__":
    main()
# endregion agent log
