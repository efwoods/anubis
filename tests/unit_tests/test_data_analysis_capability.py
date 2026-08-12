"""Unit tests for the data-analysis capability (no network, no model).

The full Model-Context-Protocol round trip (discover → ingest → execute →
persist) is exercised against the live filesystem server by the manual
harness; these tests cover the pure policies: namespace isolation, the
store byte/age quota, workspace lifecycle, key collision hashing, and the
per-device connection / per-avatar binding + suppression model that gates
the capability.

A user connects several machines at once, so the connection records are keyed
by ``device_id`` and the host-filesystem tools fan out across every connected
machine. The fan-out shape — grouped results, an offline machine reported
rather than raised, and path-to-machine resolution — is covered here too.
"""

import asyncio
import base64
import re
from datetime import UTC, datetime

import pytest
from langgraph.store.memory import InMemoryStore

from src.anubis.utils.context import GlobalContext
from src.anubis.utils.tools.data_analysis import (
    McpConnection,
    bound_connections_for,
    build_analysis_backend,
    build_connect_tool,
    build_data_analysis_tools,
    cleanup_analysis_workspace,
    clear_user_connection,
    collect_turn_artifacts,
    enforce_ingested_quota,
    is_declined,
    mark_declined,
    read_user_connections,
    resolve_device_for_path,
    save_user_connection,
)
from src.anubis.utils.tools.data_analysis.analysis_tools import (
    _decode_store_content,
    _store_key_for_source,
    timestamped_artifact_name,
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
    device_id="d-ubuntu",
    device_label="Ubuntu",
    platform="ubuntu",
)

# A second machine, so the multi-device behaviour has something to fan out to.
_SECOND_CONNECTION = McpConnection(
    url="http://localhost:8000/mcp/relay/d-macos",
    transport="streamable_http",
    server_name="macOS-Filesystem",
    allowed_roots=("/Users/evan/data",),
    device_id="d-macos",
    device_label="macOS",
    platform="macos",
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
    assert collided == _store_key_for_source(
        "/other/health1.json", "/data/health1.json"
    )


def test_decode_store_content_text_and_base64():
    assert _decode_store_content({"content": "hello", "encoding": "utf-8"}) == b"hello"
    assert (
        _decode_store_content({"content": "aGVsbG8=", "encoding": "base64"}) == b"hello"
    )


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
    tool_names = {
        t.name for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
    }
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
        await store.aput(
            ingested_namespace("u1", "a1"), "/a.json", {"content": "x" * 100}
        )
        await store.aput(
            ingested_namespace("u2", "a1"), "/b.json", {"content": "x" * 100}
        )
        await store.aput(created_namespace("u1", "a1"), "/r.md", {"content": "x" * 100})
        context.data_analysis_store_max_bytes = 0
        evicted = await enforce_ingested_quota(store, "u1", "a1", context)
        assert evicted == ["/a.json"]
        # Other user's buffer and the created-artifacts namespace are untouched.
        assert await store.aget(ingested_namespace("u2", "a1"), "/b.json") is not None
        assert await store.aget(created_namespace("u1", "a1"), "/r.md") is not None

    asyncio.run(run())


def test_mcp_connection_namespace_is_scoped_to_the_user():
    # Two-element namespace scoped to the user; records INSIDE the namespace are
    # keyed by device_id, so one user holds one record per connected machine.
    assert mcp_connection_namespace("u1") == ("u1", "mcp_connection")
    assert mcp_connection_namespace("u1") != mcp_connection_namespace("u2")
    assert len(mcp_connection_namespace("u1")) == 2


def test_connection_save_read_clear_roundtrip():
    async def run():
        store = InMemoryStore()
        assert await read_user_connections(store, "u1") == []

        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        records = await read_user_connections(store, "u1")
        assert len(records) == 1
        assert records[0]["status"] == "connected"
        assert records[0]["assistant_id"] == "a1"
        assert records[0]["url"] == _TEST_CONNECTION.url
        assert records[0]["device_id"] == "d-ubuntu"
        assert records[0]["device_label"] == "Ubuntu"

        assert await clear_user_connection(store, "u1") == ["d-ubuntu"]
        assert await read_user_connections(store, "u1") == []
        # Clearing again reports nothing existed.
        assert await clear_user_connection(store, "u1") == []

    asyncio.run(run())


def test_saving_a_second_machine_does_not_replace_the_first():
    """The singleton bug: a second machine used to overwrite the first record."""

    async def run():
        store = InMemoryStore()
        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        await save_user_connection(
            store, "u1", connection=_SECOND_CONNECTION, assistant_id="a1"
        )

        records = await read_user_connections(store, "u1")
        assert {record["device_id"] for record in records} == {"d-ubuntu", "d-macos"}

        # Disconnecting one machine leaves the other connected.
        assert await clear_user_connection(store, "u1", "d-macos") == ["d-macos"]
        remaining = await read_user_connections(store, "u1")
        assert [record["device_id"] for record in remaining] == ["d-ubuntu"]

    asyncio.run(run())


def test_a_connection_without_a_device_identifier_cannot_be_saved():
    """The record key IS the device identifier, so an unidentified device fails loudly."""

    async def run():
        store = InMemoryStore()
        anonymous = McpConnection(
            url="http://localhost:8000/mcp",
            transport="streamable_http",
            server_name="Ubuntu-OS-Filesystem",
        )
        with pytest.raises(ValueError):
            await save_user_connection(
                store, "u1", connection=anonymous, assistant_id="a1"
            )

    asyncio.run(run())


def test_bound_connections_match_only_the_bound_avatar():
    async def run():
        store = InMemoryStore()
        # No connection yet → capability off for any avatar.
        assert await bound_connections_for(store, "u1", "a1") == []

        # Bind both machines to avatar a1 (the personal avatar).
        await save_user_connection(
            store, "u1", connection=_TEST_CONNECTION, assistant_id="a1"
        )
        await save_user_connection(
            store, "u1", connection=_SECOND_CONNECTION, assistant_id="a1"
        )
        bound = await bound_connections_for(store, "u1", "a1")
        assert {connection.device_id for connection in bound} == {
            "d-ubuntu",
            "d-macos",
        }

        # The user's OTHER avatar (e.g. a test avatar) shares the namespace but
        # is NOT the bound avatar → no capability.
        assert await bound_connections_for(store, "u1", "a2") == []

        # A different user never sees these connections.
        assert await bound_connections_for(store, "u2", "a1") == []

    asyncio.run(run())


def test_suppression_marker_is_per_avatar_and_per_device():
    async def run():
        store = InMemoryStore()
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is False
        assert await is_declined(store, "u1", "a2", "d-ubuntu") is False

        await mark_declined(store, "u1", "a2", _TEST_CONNECTION)
        # Suppressing on the test avatar affects only that avatar; the personal
        # avatar still adopts the machine automatically.
        assert await is_declined(store, "u1", "a2", "d-ubuntu") is True
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is False
        # And only that machine: the user's other machines are unaffected.
        assert await is_declined(store, "u1", "a2", "d-macos") is False
        # And it never leaks to another user.
        assert await is_declined(store, "u2", "a2", "d-ubuntu") is False

    asyncio.run(run())


def test_resolve_device_for_path_picks_the_owning_machine():
    connections = [_TEST_CONNECTION, _SECOND_CONNECTION]
    assert resolve_device_for_path("/data/health.json", connections) is _TEST_CONNECTION
    assert (
        resolve_device_for_path("/Users/evan/data/health.json", connections)
        is _SECOND_CONNECTION
    )
    # A path under no allow-listed root is ambiguous rather than guessed at.
    assert resolve_device_for_path("/elsewhere/health.json", connections) is None
    # A root prefix must be a DIRECTORY boundary, not a bare string prefix, so
    # "/database" never resolves to the machine exposing "/data".
    assert resolve_device_for_path("/database/health.json", connections) is None


def test_persist_created_artifact_rejects_traversal(context):
    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
        }
        runtime = _FakeToolRuntime(store)
        result = await tools["persist_created_artifact"].coroutine(
            workspace_file_path="../../etc/passwd", runtime=runtime
        )
        assert "error" in result
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
        tools = {
            t.name: t
            for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
        }
        runtime = _FakeToolRuntime(store)

        # Natural-language "are you connected?" confirms the connection and
        # names the machines but — privacy requirement — carries NO address or
        # directory details.
        status = await tools["check_data_server_connection"].coroutine(runtime=runtime)
        assert status["connected"] is True
        assert status["device_count"] == 1
        assert status["devices"][0]["device_label"] == "Ubuntu"
        assert "url" not in status and "allowed_roots" not in status
        assert all("url" not in device for device in status["devices"])

        # Natural-language "disconnect" clears the saved connection (gate
        # closes) and suppresses automatic re-adoption of that machine.
        result = await tools["disconnect_data_server"].coroutine(runtime=runtime)
        assert result["disconnected"] is True
        assert result["disconnected_devices"] == ["Ubuntu"]
        assert await bound_connections_for(store, "u1", "a1") == []
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is True
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_disconnect_one_named_machine_leaves_the_other_connected(context):
    async def run():
        store = InMemoryStore()
        for connection in (_TEST_CONNECTION, _SECOND_CONNECTION):
            await save_user_connection(
                store, "u1", connection=connection, assistant_id="a1"
            )
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(
                context, bundle, [_TEST_CONNECTION, _SECOND_CONNECTION]
            )
        }
        runtime = _FakeToolRuntime(store)

        result = await tools["disconnect_data_server"].coroutine(
            device_label="macOS", runtime=runtime
        )
        assert result["disconnected_devices"] == ["macOS"]

        remaining = await bound_connections_for(store, "u1", "a1")
        assert [connection.device_id for connection in remaining] == ["d-ubuntu"]
        # Only the named machine is suppressed.
        assert await is_declined(store, "u1", "a1", "d-macos") is True
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is False
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_unknown_device_label_reports_the_valid_names(context):
    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(
                context, bundle, [_TEST_CONNECTION, _SECOND_CONNECTION]
            )
        }
        result = await tools["discover_data_files"].coroutine(device_label="Toaster")
        assert "error" in result
        # Naming the valid machines lets the model correct itself in one step.
        assert "Ubuntu" in result["error"] and "macOS" in result["error"]
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_connect_tool_saves_connection_and_clears_decline(context, monkeypatch):
    async def run():
        store = InMemoryStore()
        runtime = _FakeToolRuntime(store)
        # A previously disconnected machine: automatic adoption is suppressed…
        await mark_declined(store, "u1", "a1", _TEST_CONNECTION)
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is True

        # …but an explicit natural-language connect always works.
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_resolve(
            store, user_id, context, *, ignore_failure_backoff=False
        ):
            assert ignore_failure_backoff is True
            return [_TEST_CONNECTION]

        monkeypatch.setattr(at, "resolve_available_connections", _fake_resolve)
        connect = build_connect_tool(context, "u1", "a1")
        result = await connect.coroutine(runtime=runtime)
        assert result["connected"] is True
        assert result["connected_devices"] == ["Ubuntu"]
        # Privacy: no address or directory details in the tool result.
        assert "url" not in result and "allowed_roots" not in result

        bound = await bound_connections_for(store, "u1", "a1")
        assert len(bound) == 1 and bound[0].url == _TEST_CONNECTION.url
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is False

    asyncio.run(run())


def test_connect_tool_can_target_one_named_machine(context, monkeypatch):
    """ "Reconnect my Mac" must not also re-adopt a machine the user disconnected."""

    async def run():
        store = InMemoryStore()
        runtime = _FakeToolRuntime(store)
        await mark_declined(store, "u1", "a1", _TEST_CONNECTION)
        await mark_declined(store, "u1", "a1", _SECOND_CONNECTION)

        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_resolve(
            store, user_id, context, *, ignore_failure_backoff=False
        ):
            return [_TEST_CONNECTION, _SECOND_CONNECTION]

        monkeypatch.setattr(at, "resolve_available_connections", _fake_resolve)
        connect = build_connect_tool(context, "u1", "a1")
        result = await connect.coroutine(device_label="macOS", runtime=runtime)

        assert result["connected_devices"] == ["macOS"]
        bound = await bound_connections_for(store, "u1", "a1")
        assert [connection.device_id for connection in bound] == ["d-macos"]
        # The machine that was not named stays suppressed.
        assert await is_declined(store, "u1", "a1", "d-ubuntu") is True
        assert await is_declined(store, "u1", "a1", "d-macos") is False

    asyncio.run(run())


def test_ingest_passes_connection_to_mcp_calls(context, monkeypatch):
    """Regression: ingest must hand the McpConnection (not GlobalContext) to MCP calls.

    A wrong first argument raised AttributeError inside the client for every
    file, which surfaced as 'ingest failed for every health export file'.
    """

    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
        }
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


def test_timestamped_artifact_name_appends_underscored_date_and_time():
    moment = datetime(2026, 7, 27, 14, 30, 5, tzinfo=UTC)
    assert (
        timestamped_artifact_name("report.md", moment)
        == "report_2026_07_27_14_30_05.md"
    )
    assert (
        timestamped_artifact_name("plot.png", moment) == "plot_2026_07_27_14_30_05.png"
    )
    # Idempotent: a name the model already timestamped is not stamped twice.
    assert (
        timestamped_artifact_name("report_2026_07_27_14_30_05.md", moment)
        == "report_2026_07_27_14_30_05.md"
    )
    # A name that merely contains digits is still stamped.
    assert (
        timestamped_artifact_name("health_2026.md", moment)
        == "health_2026_2026_07_27_14_30_05.md"
    )


def test_persist_created_artifact_saves_under_a_timestamped_name(context):
    """The turn remembers what it saved, under a name carrying the date and time."""

    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
        }
        (bundle.workspace_path / "report.md").write_text("# Report")
        result = await tools["persist_created_artifact"].coroutine(
            workspace_file_path="report.md", runtime=_FakeToolRuntime(store)
        )

        saved_name = bundle.persisted_artifacts[0]["name"]
        assert re.fullmatch(
            r"report_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.md", saved_name
        )
        assert result == {"persisted_path": f"/data_created/{saved_name}"}
        assert bundle.persisted_artifacts[0]["workspace_name"] == "report.md"
        assert bundle.persisted_artifacts[0]["mime_type"] == "text/markdown"
        assert bundle.persisted_artifacts[0]["content"] == "# Report"
        # The store is keyed by the timestamped name, so nothing is overwritten.
        item = await store.aget(created_namespace("u1", "a1"), f"/{saved_name}")
        assert item is not None and item.value["content"] == "# Report"
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_reports_from_different_turns_do_not_overwrite_each_other(context):
    """Two turns each writing report.md must leave two distinct stored reports."""

    async def run():
        store = InMemoryStore()
        for turn_body in ("# First turn", "# Second turn"):
            bundle = build_analysis_backend(context, "u1", "a1", store=store)
            tools = {
                t.name: t
                for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
            }
            (bundle.workspace_path / "report.md").write_text(turn_body)
            await tools["persist_created_artifact"].coroutine(
                workspace_file_path="report.md", runtime=_FakeToolRuntime(store)
            )
            cleanup_analysis_workspace(bundle)
            await asyncio.sleep(1.01)  # distinct whole-second timestamps

        stored = await store.asearch(created_namespace("u1", "a1"), limit=10)
        assert len(stored) == 2
        assert {item.value["content"] for item in stored} == {
            "# First turn",
            "# Second turn",
        }

    asyncio.run(run())


def test_collect_turn_artifacts_sweeps_unpersisted_workspace_files(context):
    """A plot the model wrote but forgot to persist is still saved and displayed."""

    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        # Binary artifact never passed to persist_created_artifact.
        png_bytes = b"\x89PNG\r\n\x1a\n binary plot bytes"
        (bundle.workspace_path / "plot.png").write_bytes(png_bytes)
        # Ingested input data lives under work/ and is NOT an artifact of this turn.
        (bundle.workspace_path / "work" / "health1.json").write_text("{}")

        artifacts = await collect_turn_artifacts(context, bundle)

        assert [record["workspace_name"] for record in artifacts] == ["plot.png"]
        plot = artifacts[0]
        assert re.fullmatch(
            r"plot_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.png", plot["name"]
        )
        assert plot["mime_type"] == "image/png"
        assert plot["encoding"] == "base64"
        assert base64.standard_b64decode(plot["content"]) == png_bytes
        assert plot["persisted_path"] == f"/data_created/{plot['name']}"
        # The sweep persists as well as reports: the artifact survives the turn.
        item = await store.aget(created_namespace("u1", "a1"), f"/{plot['name']}")
        assert item is not None and item.value["encoding"] == "base64"
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_collect_turn_artifacts_does_not_duplicate_persisted_files(context):
    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(context, bundle, [_TEST_CONNECTION])
        }
        (bundle.workspace_path / "report.md").write_text("# Report")
        await tools["persist_created_artifact"].coroutine(
            workspace_file_path="report.md", runtime=_FakeToolRuntime(store)
        )
        artifacts = await collect_turn_artifacts(context, bundle)
        # The sweep must recognize the already-saved file by its WORKSPACE name;
        # the saved name differs because it carries the date and time.
        assert [record["workspace_name"] for record in artifacts] == ["report.md"]
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_collect_turn_artifacts_omits_content_above_the_inline_cap(context):
    """Oversized artifacts stay in durable storage but never bloat the reply."""

    async def run():
        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        (bundle.workspace_path / "plot.png").write_bytes(b"\x89PNG" + b"x" * 4096)
        context.data_analysis_inline_artifact_max_bytes = 100

        artifacts = await collect_turn_artifacts(context, bundle)

        assert artifacts[0]["content"] is None
        assert artifacts[0]["omitted_reason"] == "too_large"
        assert artifacts[0]["size_bytes"] > 100
        # Durable storage still holds the full artifact.
        item = await store.aget(
            created_namespace("u1", "a1"), f"/{artifacts[0]['name']}"
        )
        assert item is not None and item.value["content"]
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_connect_tool_reports_unreachable_server(context, monkeypatch):
    async def run():
        store = InMemoryStore()
        runtime = _FakeToolRuntime(store)
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_resolve(
            store, user_id, context, *, ignore_failure_backoff=False
        ):
            return []

        monkeypatch.setattr(at, "resolve_available_connections", _fake_resolve)
        connect = build_connect_tool(context, "u1", "a1")
        result = await connect.coroutine(runtime=runtime)
        assert result["connected"] is False
        assert await bound_connections_for(store, "u1", "a1") == []

    asyncio.run(run())


def test_discover_fans_out_and_groups_results_by_machine(context, monkeypatch):
    async def run():
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_call(connection, tool_name, tool_args):
            assert tool_name == "list_all_files"
            if connection.device_id == "d-ubuntu":
                assert tool_args["directory"] == "/data"
                return ["/data/steps.json", "/data/sleep.json"]
            assert tool_args["directory"] == "/Users/evan/data"
            return ["/Users/evan/data/weight.csv"]

        monkeypatch.setattr(at, "call_mcp_filesystem_tool", _fake_call)

        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(
                context, bundle, [_TEST_CONNECTION, _SECOND_CONNECTION]
            )
        }

        result = await tools["discover_data_files"].coroutine()
        # Each machine lists its OWN connected directory, and results are
        # attributable to a machine by name.
        assert set(result["devices"]) == {"Ubuntu", "macOS"}
        assert result["devices"]["Ubuntu"]["total_files"] == 2
        assert result["devices"]["macOS"]["total_files"] == 1
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_offline_machine_is_reported_not_raised(context, monkeypatch):
    """One sleeping laptop must never cost the user an answer from the others."""

    async def run():
        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        async def _fake_call(connection, tool_name, tool_args):
            if connection.device_id == "d-macos":
                raise RuntimeError("relay device is offline")
            return ["/data/steps.json"]

        monkeypatch.setattr(at, "call_mcp_filesystem_tool", _fake_call)

        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(
                context, bundle, [_TEST_CONNECTION, _SECOND_CONNECTION]
            )
        }

        result = await tools["discover_data_files"].coroutine()
        assert result["devices"]["Ubuntu"]["total_files"] == 1
        # The offline machine is PRESENT in the result, so the model can say the
        # machine was unreachable instead of reporting partial data as complete.
        assert result["devices"]["macOS"]["status"] == "offline"
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())


def test_ingest_resolves_each_path_to_its_own_machine(context, monkeypatch):
    """One ingest call may pull files from two different machines."""

    async def run():
        import base64 as b64

        import src.anubis.utils.tools.data_analysis.analysis_tools as at

        calls: list[tuple[str, str]] = []

        async def _fake_call(connection, tool_name, tool_args):
            calls.append((connection.device_id, tool_args["file_path"]))
            if tool_name == "get_file_info":
                return {"modified_at": "2026-08-01T00:00:00+00:00", "size_bytes": 3}
            return b64.standard_b64encode(b"{}\n").decode("ascii")

        monkeypatch.setattr(at, "call_mcp_filesystem_tool", _fake_call)

        store = InMemoryStore()
        bundle = build_analysis_backend(context, "u1", "a1", store=store)
        tools = {
            t.name: t
            for t in build_data_analysis_tools(
                context, bundle, [_TEST_CONNECTION, _SECOND_CONNECTION]
            )
        }
        runtime = _FakeToolRuntime(store)

        result = await tools["ingest_data_files"].coroutine(
            file_paths=["/data/steps.json", "/Users/evan/data/weight.json"],
            runtime=runtime,
        )
        assert result["errors"] == []
        # Each path went to the machine whose allow-listed root contains it.
        assert ("d-ubuntu", "/data/steps.json") in calls
        assert ("d-macos", "/Users/evan/data/weight.json") in calls
        assert ("d-macos", "/data/steps.json") not in calls

        # The buffered copy records which machine the data came from.
        items = await store.asearch(ingested_namespace("u1", "a1"), limit=10)
        labels = {(item.value or {}).get("device_label") for item in items}
        assert labels == {"Ubuntu", "macOS"}
        cleanup_analysis_workspace(bundle)

    asyncio.run(run())
