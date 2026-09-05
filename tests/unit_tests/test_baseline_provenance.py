"""The startup ladder that keeps the style baseline in step with MODEL.

Everything here runs against an in-memory stand-in for the ``store`` table and a
recorded stand-in for ``subprocess.Popen``: what matters is WHICH decision the
ladder takes and that the retrain is started at most once per model, never the
retrain itself.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.dataset import baseline_provenance as provenance_module
from src.anubis.utils.dataset.baseline_provenance import (
    BASELINE_PROVENANCE_STORE_KEY,
    BASELINE_RETRAIN_LOCK_STORE_KEY,
    baseline_is_stale_for_model,
    ensure_baseline_matches_model,
)
from src.anubis.utils.dataset.style_features import BASELINE_PROVENANCE_PATH


# ---------------------------------------------------------------------------
# An in-memory ``store`` table that honours exactly the statements the module issues
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, table):
        self.table = table
        self.rowcount = 0
        self._result = None

    async def execute(self, statement, parameters):
        self._result = None
        if statement == provenance_module.SQL_SELECT_STORE_ROW:
            prefix, key = parameters
            row = self.table.get((prefix, key))
            self._result = (row,) if row is not None else None
        elif statement == provenance_module.SQL_INSERT_STORE_ROW_IF_ABSENT:
            prefix, key, value = parameters
            if (prefix, key) in self.table:
                self.rowcount = 0
            else:
                self.table[(prefix, key)] = json.loads(value)
                self.rowcount = 1
        elif statement == provenance_module.SQL_TAKE_OVER_STORE_ROW:
            new_value, prefix, key, previous_value = parameters
            if self.table.get((prefix, key)) == json.loads(previous_value):
                self.table[(prefix, key)] = json.loads(new_value)
                self.rowcount = 1
            else:
                self.rowcount = 0
        elif statement == provenance_module.SQL_DELETE_STORE_ROW:
            prefix, key = parameters
            self.rowcount = 1 if self.table.pop((prefix, key), None) is not None else 0
        else:
            raise AssertionError(f"unexpected statement {statement!r}")

    async def fetchone(self):
        return self._result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return False


class _FakeConnection:
    def __init__(self, table):
        self.table = table

    def cursor(self):
        return _FakeCursor(self.table)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return False


class _FakePool:
    def __init__(self):
        self.table = {}

    def connection(self):
        return _FakeConnection(self.table)

    def put(self, key, payload):
        self.table[(key, key)] = {"value": json.dumps(payload)}


class _FakeProcess:
    def __init__(self, return_code, on_wait=None):
        self.pid = 4242
        self.return_code = return_code
        self.on_wait = on_wait

    def wait(self):
        if self.on_wait is not None:
            self.on_wait()
        return self.return_code


class _AppState:
    def __init__(self, context, pool):
        self.context = context
        self.pool = pool


def _context(model="gpt-5.6-luna", **overrides) -> GlobalContext:
    values = {
        "model": model,
        "baseline_auto_retrain_lock_stale_after_seconds": 7200,
        "baseline_auto_retrain_poll_seconds": 60,
    }
    values.update(overrides)
    context = GlobalContext(**values)
    # ``__post_init__`` only consults the environment for fields left at their
    # default, and the gate's default is True; pin every value the ladder reads.
    context.baseline_auto_retrain_on_model_change = overrides.get(
        "baseline_auto_retrain_on_model_change", True
    )
    context.model = model
    return context


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """A throwaway checkout root with the sidecar path inside it."""
    (tmp_path / "data").mkdir()
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(provenance_module, "repository_root", lambda: tmp_path)
    monkeypatch.delenv("BASELINE_RESPONSE_THRESHOLD", raising=False)
    return tmp_path


@pytest.fixture
def spawned(monkeypatch):
    """Record every retrain-script spawn; each entry is the extra argument list."""
    calls = []
    outcomes = {"return_code": 0, "on_wait": None}

    def fake_popen(command, cwd, stdout, stderr, start_new_session):
        assert start_new_session is True
        assert command[1].endswith("retrain_chatgpt_baseline.py")
        calls.append(list(command[2:]))
        return _FakeProcess(outcomes["return_code"], outcomes["on_wait"])

    monkeypatch.setattr(provenance_module.subprocess, "Popen", fake_popen)
    calls_holder = {"calls": calls, "outcomes": outcomes}
    return calls_holder


def _write_sidecar(root, model, threshold=50.0):
    sidecar = root / BASELINE_PROVENANCE_PATH
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps({"model": model, "baseline_response_threshold": threshold}),
        encoding="utf-8",
    )


async def _run(app_state):
    task = await ensure_baseline_matches_model(app_state)
    if task is not None:
        await task
    return task


class TestStaleness:
    def test_truth_table(self):
        assert baseline_is_stale_for_model(None, "gpt-5.6-luna") is True
        assert baseline_is_stale_for_model({}, "gpt-5.6-luna") is True
        assert baseline_is_stale_for_model({"model": "gpt-5.4-nano"}, "gpt-5.6-luna") is True
        assert baseline_is_stale_for_model({"model": "gpt-5.6-luna"}, "gpt-5.6-luna") is False
        assert baseline_is_stale_for_model({"model": " gpt-5.6-luna "}, "gpt-5.6-luna") is False
        # Nothing configured means nothing to compare against or retrain with.
        assert baseline_is_stale_for_model(None, "") is False
        assert baseline_is_stale_for_model({"model": "gpt-5.4-nano"}, None) is False


class TestLadder:
    def test_disk_match_does_nothing(self, repository, spawned):
        _write_sidecar(repository, "gpt-5.6-luna")
        pool = _FakePool()
        task = asyncio.run(_run(_AppState(_context(), pool)))
        assert task is None
        assert spawned["calls"] == []
        assert pool.table == {}

    def test_unset_model_or_missing_pool_is_a_no_op(self, repository, spawned):
        _write_sidecar(repository, "gpt-5.4-nano")
        assert asyncio.run(_run(_AppState(_context(model=""), _FakePool()))) is None
        assert asyncio.run(_run(_AppState(_context(), None))) is None
        assert spawned["calls"] == []

    def test_gate_off_only_warns(self, repository, spawned, caplog):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()
        context = _context(baseline_auto_retrain_on_model_change=False)
        task = asyncio.run(_run(_AppState(context, pool)))
        assert task is None
        assert spawned["calls"] == []
        assert pool.table == {}
        assert "automatic retraining is disabled" in caplog.text

    def test_store_match_adopts_without_retraining(self, repository, spawned, monkeypatch):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()
        pool.put(
            BASELINE_PROVENANCE_STORE_KEY,
            {"model": "gpt-5.6-luna", "baseline_response_threshold": 61.5},
        )
        app_state = _AppState(_context(), pool)
        asyncio.run(_run(app_state))
        assert spawned["calls"] == [["--adopt-from-store"]]
        assert provenance_module.os.environ["BASELINE_RESPONSE_THRESHOLD"] == repr(61.5)
        assert app_state.context.baseline_response_threshold == 61.5
        assert (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY) not in pool.table

    def test_first_boot_takes_the_lock_and_retrains_once(self, repository, spawned):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()

        def script_side_effects():
            # What the real script does before exiting 0: writes the sidecar,
            # publishes provenance, releases the lock.
            _write_sidecar(repository, "gpt-5.6-luna", threshold=58.0)
            pool.put(
                BASELINE_PROVENANCE_STORE_KEY,
                {"model": "gpt-5.6-luna", "baseline_response_threshold": 58.0},
            )
            pool.table.pop((BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY), None)

        spawned["outcomes"]["on_wait"] = script_side_effects
        app_state = _AppState(_context(), pool)
        asyncio.run(_run(app_state))
        assert spawned["calls"] == [[]]
        assert app_state.context.baseline_response_threshold == 58.0
        assert (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY) not in pool.table

        # A later boot of the same checkout sees the sidecar and does nothing.
        asyncio.run(_run(_AppState(_context(), pool)))
        assert spawned["calls"] == [[]]

    def test_second_container_waits_for_the_lock_holder_then_adopts(self, repository, spawned):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()
        pool.put(
            BASELINE_RETRAIN_LOCK_STORE_KEY,
            {
                "model": "gpt-5.6-luna",
                "owner": "sibling:1",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        context = _context(
            baseline_auto_retrain_poll_seconds=1,
            baseline_auto_retrain_lock_stale_after_seconds=7200,
        )
        app_state = _AppState(context, pool)

        async def scenario():
            task = await ensure_baseline_matches_model(app_state)
            assert task is not None
            # The sibling publishes while this container is polling.
            pool.put(
                BASELINE_PROVENANCE_STORE_KEY,
                {"model": "gpt-5.6-luna", "baseline_response_threshold": 59.0},
            )
            await task

        asyncio.run(scenario())
        # No second retrain: only the adoption of the sibling's result.
        assert spawned["calls"] == [["--adopt-from-store"]]
        assert app_state.context.baseline_response_threshold == 59.0

    def test_abandoned_lock_is_taken_over(self, repository, spawned):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()
        pool.put(
            BASELINE_RETRAIN_LOCK_STORE_KEY,
            {
                "model": "gpt-5.6-luna",
                "owner": "dead:1",
                "started_at": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
            },
        )
        spawned["outcomes"]["on_wait"] = lambda: _write_sidecar(repository, "gpt-5.6-luna", 57.0)
        asyncio.run(_run(_AppState(_context(), pool)))
        assert spawned["calls"] == [[]]

    def test_failed_retrain_releases_the_lock_for_a_later_boot(self, repository, spawned, caplog):
        _write_sidecar(repository, "gpt-5.4-nano")
        pool = _FakePool()
        spawned["outcomes"]["return_code"] = 1
        app_state = _AppState(_context(), pool)
        asyncio.run(_run(app_state))
        assert spawned["calls"] == [[]]
        assert (BASELINE_RETRAIN_LOCK_STORE_KEY, BASELINE_RETRAIN_LOCK_STORE_KEY) not in pool.table
        assert "failed (exit 1)" in caplog.text
        assert "BASELINE_RESPONSE_THRESHOLD" not in provenance_module.os.environ
