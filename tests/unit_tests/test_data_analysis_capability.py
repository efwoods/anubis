"""Unit tests for the data-analysis capability (no network, no model).

The full Model-Context-Protocol round trip (discover → ingest → execute →
persist) is exercised against the live filesystem server by the manual
harness; these tests cover the pure policies: namespace isolation, the
store byte/age quota, workspace lifecycle, key collision hashing, and the
per-user single connection / per-avatar binding + decline model that gates
the capability.
"""

import asyncio

import pytest
from langgraph.store.memory import InMemoryStore

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis import (
    McpConnection,
    bound_connection_for,
    build_analysis_backend,
    build_connect_tool,
    build_data_analysis_tools,
    cleanup_analysis_workspace,
    clear_user_connection,
    enforce_ingested_quota,
    is_declined,
    mark_declined,
    read_user_connection,
    save_user_connection,
)
from src.anubis.utils.tools.data_analysis.analysis_tools import (
    _decode_store_content,
    _store_key_for_source,
)
from src.anubis.utils.tools.data_analysis.backend import (
    created_namespace,
    ingested_namespace,
    mcp_connection_namespace,
)

# A stand-in resolved connection used wherever a tool set must be built.
_TEST_CONNECTION = McpConnection(
    url="http://localhost:8000/mcp",
    transport="streamable_http",
    server_name="Ubuntu-OS-Filesystem",
    allowed_roots=("/data",),
)


class _FakeToolRuntime:
    """Stand-in for the injected ``ToolRuntime``: only ``.store`` is used."""

    def __init__(self, store):
        self.store = store


@pytest.fixture
def context(tmp_path):
    ctx = GlobalContext()
    ctx.data_analysis_workspace_root = str(tmp_path / "analysis")
    return ctx


def test_namespaces_are_per_user_per_avatar():
    assert ingested_namespace("u1", "a1") == ("u1", "a1", "data_ingested")
    assert created_namespace("u1", "a1") == ("u1", "a1", "data_created")
    # Distinct user or avatar → disjoint namespaces.
    assert ingested_namespace("u1", "a1") != ingested_namespace("u2", "a1")
    assert ingested_namespace("u1", "a1") != ingested_namespace("u1", "a2")
    assert ingested_namespace("u1", "a1") != created_namespace("u1", "a1")


def test_store_key_plain_name_and_collision_hash():
    # Same source path keeps the plain basename key.
    assert _store_key_for_source("/data/health1.json", None) == "/health1.json"
    assert (
        _store_key_for_source("/data/health1.json", "/data/health1.json")
        == "/health1.json"
    )
    # A different source path with the same basename gets a deterministic suffix.
    collided = _store_key_for_source("/other/health1.json", "/data/health1.json")
    assert collided != "/health1.json"
    assert collided.startswith("/health1-") and collided.endswith(".json")
    assert collided == _store_key_for_source("/other/health1.json", "/data/health1.json")


def test_decode_store_content_text_and_base64():
    assert _decode_store_content({"content": "hello", "encoding": "utf-8"}) == b"hello"
    assert _decode_store_content({"content": "aGVsbG8=", "encoding": "base64"}) == b"hello"


def test_workspace_lifecycle(context):
    bundle = build_analysis_backend(context, "u1", "a1", store=InMemoryStore())
    assert bundle.workspace_path.is_dir()
    assert (bundle.workspace_path / "work").is_dir()
    cleanup_analysis_workspace(bundle)
    assert not bundle.workspace_path.exists()


def test_unsupported_execution_backend_raises(context):
    context.data_analysis_execution_backend = "daytona"
    with pytest.raises(NotImplementedError):
        build_analysis_backend(context, "u1", "a1", store=InMemoryStore())


def test_tool_set_names(context):
    bundle = build_analysis_backend(context, "u1", "a1", store=InMemoryStore())
    tool_names = {t.name for t in build_data_analysis_tools(context, bundle, _TEST_CONNECTION)}
    assert tool_names == {
        "discover_data_files",
        "preview_data_file",
        "ingest_data_files",
        "hydrate_ingested_data",
        "persist_created_artifact",
        "list_persisted_data",
        "check_data_server_connection",
        "disconnect_data_server",
    }
    cleanup_analysis_workspace(bundle)


def test_quota_evicts_least_recently_updated_first(context):
    async def run():
        store = InMemoryStore()
        namespace = ingested_namespace("u1", "a1")
        # Three 10-byte items; quota of 20 forces one eviction (oldest first).
        for name in ("old", "mid", "new"):
            await store.aput(namespace, f"/{name}.json", {"content": "x" * 10})
            await asyncio.sleep(0.01)  # distinct updated_at ordering
        context.data_analysis_store_max_bytes = 20
        evicted = await enforce_ingested_quota(store, "u1", "a1", context)
        remaining = {item.key for item in await store.asearch(namespace, limit=10)}
        assert evicted == ["/old.json"]
        assert remaining == {"/mid.json", "/new.json"}

    asyncio.run(run())


def test_quota_age_backstop(context):
    async def run():
        store = InMemoryStore()
        namespace = ingested_namespace("u1", "a1")
        await store.aput(namespace, "/fresh.json", {"content": "x"})
        # InMemoryStore timestamps cannot be back-dated, so make the age
        # window negative: every item is then older than the limit and the
        # backstop must evict all of them.
        context.data_analysis_store_max_age_days = -1
        evicted = await enforce_ingested_quota(store, "u1", "a1", context)
        assert evicted == ["/fresh.json"]

    asyncio.run(run())


def test_quota_ignores_other_namespaces(context):
    async def run():
        store = InMemoryStore()
        await store.aput(ingested_namespace("u1", "a1"), "/a.json", {"content": "x" * 100})
        await store.aput(ingested_namespace("u2", "a1"), "/b.json", {"content": "x" * 100})
        await store.aput(created_namespace("u1", "a1"), "/r.md", {"content": "x" * 100})
        context.data_analysis_store_max_bytes = 0
        evicted = await enforce_ingested_quota(store, "u1", "a1", context)
        assert evicted == ["/a.json"]
        # Other user's buffer and the created-artifacts namespace are untouched.
        assert await store.aget(ingested_namespace("u2", "a1"), "/b.json") is not None
        assert await store.aget(created_namespace("u1", "a1"), "/r.md") is not None

    asyncio.run(run())


def test_mcp_connection_namespace_is_single_per_user():
    # Two-element namespace keyed by user only — exactly one connection record
    # can exist per user, regardless of how many avatars the user owns.
    assert mcp_connection_namespace("u1") == ("u1", "mcp_connection")
    assert mcp_connection_namespace("u1") != mcp_connection_namespace("u2")
    assert len(mcp_connection_namespace("u1")) == 2


def test_connection_save_read_clear_roundtrip():
    async def run():
        store = InMemoryStore()
        assert await read_user_connection(store, "u1") is None

        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        record = await read_user_connection(store, "u1")
        assert record is not None
        assert record["status"] == "connected"
        assert record["assistant_id"] == "a1"
        assert record["url"] == _TEST_CONNECTION.url

        assert await clear_user_connection(store, "u1") is True
        assert await read_user_connection(store, "u1") is None
        # Clearing again reports nothing existed.
        assert await clear_user_connection(store, "u1") is False

    asyncio.run(run())


def test_bound_connection_matches_only_the_bound_avatar():
    async def run():
        store = InMemoryStore()
        # No connection yet → capability off for any avatar.
        assert await bound_connection_for(store, "u1", "a1") is None

        # Connect, binding the single connection to avatar a1 (the personal avatar).
        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        bound = await bound_connection_for(store, "u1", "a1")
        assert bound is not None and bound.url == _TEST_CONNECTION.url

        # The user's OTHER avatar (e.g. a test avatar) shares the single
        # connection record but is NOT the bound avatar → no capability.
        assert await bound_connection_for(store, "u1", "a2") is None

        # A different user never sees this connection.
        assert await bound_connection_for(store, "u2", "a1") is None

    asyncio.run(run())


def test_decline_marker_is_per_avatar():
    async def run():
        store = InMemoryStore()
        assert await is_declined(store, "u1", "a1") is False
        assert await is_declined(store, "u1", "a2") is False

        await mark_declined(store, "u1", "a2", _TEST_CONNECTION)
        # Declining on the test avatar suppresses only that avatar; the
        # personal avatar can still be offered the connection.
        assert await is_declined(store, "u1", "a2") is True
        assert await is_declined(store, "u1", "a1") is False
        # And it never leaks to another user.
        assert await is_declined(store, "u2", "a2") is False

    asyncio.run(run())


def test_persist_created_artifact_rejects_traversal(context):
    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {t.name: t for t in build_data_analysis_tools(context, bundle, _TEST_CONNECTION)}
        runtime = _FakeToolRuntime(store)
        result = await tools["persist_created_artifact"].coroutine(
            workspace_file_path="../../etc/passwd", runtime=runtime
        )
        assert "error" in result
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_persist_created_artifact_roundtrip(context):
    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {t.name: t for t in build_data_analysis_tools(context, bundle, _TEST_CONNECTION)}
        runtime = _FakeToolRuntime(store)
        (bundle.workspace_path / "report.md").write_text("# Report")
        result = await tools["persist_created_artifact"].coroutine(
            workspace_file_path="report.md", runtime=runtime
        )
        assert result == {"persisted_path": "/data_created/report.md"}
        item = await store.aget(created_namespace("u1", "a1"), "/report.md")
        assert item is not None and item.value["content"] == "# Report"
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_check_and_disconnect_tools(context):
    async def run():
        store = InMemoryStore()
        # Establish a connection bound to this avatar, then build the tools.
        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {t.name: t for t in build_data_analysis_tools(context, bundle, _TEST_CONNECTION)}
        runtime = _FakeToolRuntime(store)

        # Natural-language "are you connected?" confirms the connection but —
        # privacy requirement — carries NO address or directory details.
        status = await tools["check_data_server_connection"].coroutine(runtime=runtime)
        assert status == {"connected": True, "server": "Neural Nexus MCP data server"}
        assert "url" not in status and "allowed_roots" not in status

        # Natural-language "disconnect" clears the saved connection (gate
        # closes) and suppresses the automatic re-offer for this avatar.
        result = await tools["disconnect_data_server"].coroutine(runtime=runtime)
        assert result["disconnected"] is True
        assert await bound_connection_for(store, "u1", "a1") is None
        assert await is_declined(store, "u1", "a1") is True
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_connect_tool_saves_connection_and_clears_decline(context, monkeypatch):
    async def run():
        store = InMemoryStore()
        runtime = _FakeToolRuntime(store)
        # A previously declined avatar: automatic offers are suppressed…
        await mark_declined(store, "u1", "a1", _TEST_CONNECTION)
        assert await is_declined(store, "u1", "a1") is True

        # …but an explicit natural-language connect always works.
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_discover(url, timeout, *, ignore_failure_backoff=False):
            assert ignore_failure_backoff is True
            return _TEST_CONNECTION

        monkeypatch.setattr(at, "discover_announced_server", _fake_discover)
        connect = build_connect_tool(context, "u1", "a1")
        result = await connect.coroutine(runtime=runtime)
        assert result["connected"] is True
        # Privacy: no address or directory details in the tool result.
        assert "url" not in result and "allowed_roots" not in result

        bound = await bound_connection_for(store, "u1", "a1")
        assert bound is not None and bound.url == _TEST_CONNECTION.url
        assert await is_declined(store, "u1", "a1") is False

    asyncio.run(run())


def test_ingest_passes_connection_to_mcp_calls(context, monkeypatch):
    """Regression: ingest must hand the McpConnection (not GlobalContext) to MCP calls.

    A wrong first argument raised AttributeError inside the client for every
    file, which surfaced as 'ingest failed for every health export file'.
    """

    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {t.name: t for t in build_data_analysis_tools(context, bundle, _TEST_CONNECTION)}
        runtime = _FakeToolRuntime(store)

        import base64 as b64

        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_mcp(connection, tool_name, tool_args):
            assert isinstance(connection, McpConnection), (
                f"MCP call {tool_name} received {type(connection).__name__} "
                "instead of McpConnection"
            )
            if tool_name == "get_file_info":
                return {"modified_at": "2026-07-09T00:00:00+00:00", "size_bytes": 5}
            if tool_name == "read_file_bytes":
                return b64.standard_b64encode(b"hello").decode("ascii")
            raise AssertionError(f"unexpected tool {tool_name}")

        monkeypatch.setattr(at, "call_mcp_filesystem_tool", _fake_mcp)
        result = await tools["ingest_data_files"].coroutine(
            file_paths=["/data/health1.json"], runtime=runtime
        )
        assert result["errors"] == []
        assert result["ingested"] == ["/data/health1.json"]
        item = await store.aget(ingested_namespace("u1", "a1"), "/health1.json")
        assert item is not None and item.value["content"] == "hello"
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_connect_tool_reports_unreachable_server(context, monkeypatch):
    async def run():
        store = InMemoryStore()
        runtime = _FakeToolRuntime(store)
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_discover(url, timeout, *, ignore_failure_backoff=False):
            return None

        monkeypatch.setattr(at, "discover_announced_server", _fake_discover)
        connect = build_connect_tool(context, "u1", "a1")
        result = await connect.coroutine(runtime=runtime)
        assert result["connected"] is False
        assert await bound_connection_for(store, "u1", "a1") is None

    asyncio.run(run())
