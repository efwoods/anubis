"""The LangSmith trace tally admits one slot per trace and drops traces past the limit."""

from datetime import UTC, datetime, timedelta

from src.anubis.utils import trace_tally as trace_tally_module
from src.anubis.utils.trace_tally import TraceTally


class FakeSharedTally:
    """Stands in for the Postgres tally row: grants slots up to the limit."""

    def __init__(self, trace_limit: int, already_reserved: int = 0) -> None:
        self.trace_limit = trace_limit
        self.reserved_traces = already_reserved
        self.reservation_calls = 0
        self.unreachable = False

    def reserve(self, requested_slots: int) -> int:
        self.reservation_calls += 1
        if self.unreachable:
            raise ConnectionError("postgres unreachable")
        granted_slots = max(
            0, min(requested_slots, self.trace_limit - self.reserved_traces)
        )
        self.reserved_traces += granted_slots
        return granted_slots


def tally_with(
    fake_tally: FakeSharedTally, block_size: int = 3, period_open: bool = True
) -> TraceTally:
    trace_tally = TraceTally("postgresql://unused", fake_tally.trace_limit, block_size)
    trace_tally.organization_id = "organization"
    trace_tally.period_end = datetime.now(UTC) + (
        timedelta(days=10) if period_open else -timedelta(seconds=1)
    )
    trace_tally._reserve_block = lambda: fake_tally.reserve(trace_tally.block_size)  # type: ignore[method-assign]
    return trace_tally


def test_every_run_of_one_trace_takes_one_slot():
    fake_tally = FakeSharedTally(trace_limit=5)
    trace_tally = tally_with(fake_tally)
    assert all(trace_tally.admit_trace("trace-a") for _ in range(50))
    assert trace_tally.unreported_sent_traces == 1


def test_traces_past_the_limit_are_dropped_and_stay_dropped():
    fake_tally = FakeSharedTally(trace_limit=5)
    trace_tally = tally_with(fake_tally, block_size=3)
    admitted = [trace_tally.admit_trace(f"trace-{i}") for i in range(8)]
    assert admitted == [True] * 5 + [False] * 3
    assert fake_tally.reserved_traces == 5
    assert trace_tally.admit_trace("trace-6") is False
    assert trace_tally.admit_trace("trace-0") is True


def test_two_processes_share_one_limit():
    fake_tally = FakeSharedTally(trace_limit=10)
    first_process = tally_with(fake_tally, block_size=4)
    second_process = tally_with(fake_tally, block_size=4)
    admitted = sum(
        first_process.admit_trace(f"first-{i}")
        + second_process.admit_trace(f"second-{i}")
        for i in range(20)
    )
    assert admitted == 10


def test_seeded_traces_already_at_the_limit_admit_nothing():
    fake_tally = FakeSharedTally(trace_limit=5000, already_reserved=5200)
    trace_tally = tally_with(fake_tally)
    assert trace_tally.admit_trace("trace-a") is False


def test_an_unreachable_tally_drops_traces_and_backs_off():
    fake_tally = FakeSharedTally(trace_limit=100)
    fake_tally.unreachable = True
    trace_tally = tally_with(fake_tally)
    assert trace_tally.admit_trace("trace-a") is False
    assert trace_tally.admit_trace("trace-b") is False
    assert fake_tally.reservation_calls == 1
    fake_tally.unreachable = False
    trace_tally.reservation_retry_after = datetime.now(UTC) - timedelta(seconds=1)
    assert trace_tally.admit_trace("trace-c") is True


def test_unknown_or_ended_billing_period_drops_traces():
    fake_tally = FakeSharedTally(trace_limit=100)
    ended_period_tally = tally_with(fake_tally, period_open=False)
    assert ended_period_tally.admit_trace("trace-a") is False
    unknown_period_tally = TraceTally("postgresql://unused", 100, 10)
    assert unknown_period_tally.admit_trace("trace-a") is False
    assert fake_tally.reservation_calls == 0


def test_a_new_billing_period_forgets_old_slots_and_decisions(monkeypatch):
    fake_tally = FakeSharedTally(trace_limit=1)
    trace_tally = tally_with(fake_tally, block_size=1)
    assert trace_tally.admit_trace("trace-a") is True
    assert trace_tally.admit_trace("trace-b") is False
    monkeypatch.setattr(trace_tally, "_connection", lambda: None)
    new_period_end = trace_tally.period_end + timedelta(days=30)
    trace_tally.start_period("organization", new_period_end, seeded_traces=None)
    fake_tally.reserved_traces = 0
    assert trace_tally.admit_trace("trace-b") is True


def test_trace_id_of_reads_trace_id_dotted_order_or_run_id():
    assert trace_tally_module.trace_id_of({"trace_id": "abc", "id": "def"}) == "abc"
    assert (
        trace_tally_module.trace_id_of(
            {"dotted_order": "20261003T194557406000Zroot-id.20261003T194558Zchild-id"}
        )
        == "root-id"
    )
    assert trace_tally_module.trace_id_of({"id": "only-id"}) == "only-id"


def test_the_installed_filter_drops_every_run_of_a_dropped_trace(monkeypatch):
    import langsmith

    class FakeClient:
        def _filter_for_sampling(self, runs):
            return list(runs)

    monkeypatch.setattr(langsmith, "Client", FakeClient)
    fake_tally = FakeSharedTally(trace_limit=1)
    trace_tally = tally_with(fake_tally, block_size=1)
    assert trace_tally_module.install_trace_tally_filter(trace_tally) is True
    client = FakeClient()
    sent_runs = client._filter_for_sampling(
        [
            {"id": "root-1", "trace_id": "trace-1"},
            {"id": "child-1", "trace_id": "trace-1"},
            {"id": "root-2", "trace_id": "trace-2"},
            {"id": "child-2", "trace_id": "trace-2"},
        ]
    )
    assert [run["id"] for run in sent_runs] == ["root-1", "child-1"]


def test_a_langsmith_release_without_the_filter_turns_tracing_off(monkeypatch):
    import langsmith

    class ClientWithoutFilter:
        pass

    monkeypatch.setattr(langsmith, "Client", ClientWithoutFilter)
    tracing_switches: list[bool] = []
    monkeypatch.setattr(
        trace_tally_module, "set_process_tracing", tracing_switches.append
    )

    class Context:
        langsmith_tracing = "true"
        langsmith_monthly_trace_limit = 5000
        async_postgres_store_uri = "postgresql://unused"

    import asyncio

    asyncio.run(trace_tally_module.run_trace_tally(Context()))
    assert tracing_switches == [False]


def test_period_start_is_one_month_before_period_end():
    assert trace_tally_module.period_start_for(
        datetime(2026, 11, 1, tzinfo=UTC)
    ) == datetime(2026, 10, 1, tzinfo=UTC)
    assert trace_tally_module.period_start_for(
        datetime(2027, 1, 1, tzinfo=UTC)
    ) == datetime(2026, 12, 1, tzinfo=UTC)


def test_the_api_key_fingerprint_never_contains_the_key():
    class Context:
        langsmith_api_key = "lsv2_pt_secret_value"

    fingerprint = trace_tally_module.api_key_fingerprint_of(Context())
    assert len(fingerprint) == 16
    assert "secret" not in fingerprint
