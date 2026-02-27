"""Tests for kovrin_safety.audit — Merkle hash chain audit trail.

Tests the ImmutableTraceLog and HashedTrace classes that provide
tamper-evident, append-only cryptographic audit trails. Validates:
  - Basic append/read/len operations
  - SHA-256 hash chain linking (genesis -> N events)
  - Deterministic hashing (same input -> same hash)
  - Integrity verification (detects tampering)
  - Event querying with filters
  - JSON export for external audit
  - Async append with subscriber notifications
  - Edge cases and boundary conditions
"""

import json
import tempfile
from pathlib import Path

import pytest

from kovrin_safety.audit import HashedTrace, ImmutableTraceLog
from kovrin_safety.models import RiskLevel, Trace

# --- Fixtures ------------------------------------------------


@pytest.fixture
def log() -> ImmutableTraceLog:
    """Fresh, empty trace log for each test."""
    return ImmutableTraceLog()


@pytest.fixture
def populated_log() -> ImmutableTraceLog:
    """Log with 5 events already appended."""
    tlog = ImmutableTraceLog()
    for i in range(5):
        tlog.append(
            Trace(
                event_type=f"TYPE_{i}",
                description=f"Event {i}",
                intent_id=f"int-{i % 2}",
                task_id=f"task-{i % 3}",
                risk_level=RiskLevel.LOW if i < 3 else RiskLevel.HIGH,
            )
        )
    return tlog


# --- Basic Operations -----------------------------------------


class TestBasicOperations:
    """Core append, length, and state tracking."""

    def test_empty_log_length(self, log):
        assert len(log) == 0

    def test_empty_log_head_is_genesis(self, log):
        assert log.head_hash == "0" * 64

    def test_genesis_hash_constant(self):
        assert ImmutableTraceLog.GENESIS_HASH == "0" * 64

    def test_append_single_returns_hashed_trace(self, log):
        trace = Trace(event_type="TEST", description="Hello")
        result = log.append(trace)
        assert isinstance(result, HashedTrace)

    def test_append_increments_length(self, log):
        log.append(Trace(event_type="A"))
        assert len(log) == 1
        log.append(Trace(event_type="B"))
        assert len(log) == 2

    def test_append_multiple(self, log):
        for i in range(25):
            log.append(Trace(event_type=f"EVT_{i}"))
        assert len(log) == 25

    def test_head_hash_updates_on_append(self, log):
        h0 = log.head_hash
        log.append(Trace(event_type="A"))
        h1 = log.head_hash
        log.append(Trace(event_type="B"))
        h2 = log.head_hash
        assert h0 != h1
        assert h1 != h2
        assert h0 != h2

    def test_head_hash_matches_last_event_hash(self, log):
        hashed = log.append(Trace(event_type="A"))
        assert log.head_hash == hashed.hash
        hashed2 = log.append(Trace(event_type="B"))
        assert log.head_hash == hashed2.hash

    def test_sequence_numbers_monotonic(self, log):
        for i in range(10):
            hashed = log.append(Trace(event_type="TEST"))
            assert hashed.sequence == i

    def test_sequence_starts_at_zero(self, log):
        hashed = log.append(Trace(event_type="FIRST"))
        assert hashed.sequence == 0

    def test_trace_preserved_in_hashed_trace(self, log):
        trace = Trace(
            event_type="CRITIC_PIPELINE",
            description="Test desc",
            intent_id="int-abc",
            task_id="task-xyz",
            risk_level=RiskLevel.HIGH,
            l0_passed=True,
            details={"key": "value"},
        )
        hashed = log.append(trace)
        assert hashed.trace.event_type == "CRITIC_PIPELINE"
        assert hashed.trace.description == "Test desc"
        assert hashed.trace.intent_id == "int-abc"
        assert hashed.trace.task_id == "task-xyz"
        assert hashed.trace.risk_level == RiskLevel.HIGH
        assert hashed.trace.l0_passed is True
        assert hashed.trace.details == {"key": "value"}


# --- Hash Chain Integrity -------------------------------------


class TestHashChain:
    """Verify SHA-256 chain linking is correct."""

    def test_first_event_links_to_genesis(self, log):
        hashed = log.append(Trace(event_type="FIRST"))
        assert hashed.previous_hash == ImmutableTraceLog.GENESIS_HASH

    def test_second_event_links_to_first(self, log):
        h1 = log.append(Trace(event_type="A"))
        h2 = log.append(Trace(event_type="B"))
        assert h2.previous_hash == h1.hash

    def test_chain_of_three(self, log):
        h1 = log.append(Trace(event_type="A"))
        h2 = log.append(Trace(event_type="B"))
        h3 = log.append(Trace(event_type="C"))
        assert h1.previous_hash == "0" * 64
        assert h2.previous_hash == h1.hash
        assert h3.previous_hash == h2.hash

    def test_long_chain(self, log):
        """Chain of 100 events maintains proper linking."""
        prev_hash = ImmutableTraceLog.GENESIS_HASH
        for i in range(100):
            hashed = log.append(Trace(event_type=f"EVT_{i}"))
            assert hashed.previous_hash == prev_hash
            prev_hash = hashed.hash

    def test_hash_is_sha256_hex(self, log):
        hashed = log.append(Trace(event_type="TEST"))
        assert len(hashed.hash) == 64
        assert all(c in "0123456789abcdef" for c in hashed.hash)

    def test_hash_includes_event_content(self):
        """Different event content produces different hashes from genesis."""
        log1 = ImmutableTraceLog()
        log2 = ImmutableTraceLog()
        h1 = log1.append(Trace(id="tr-same", event_type="A", description="Alpha"))
        h2 = log2.append(Trace(id="tr-same", event_type="B", description="Beta"))
        assert h1.hash != h2.hash

    def test_deterministic_hashing(self):
        """Same trace content + same chain state produces identical hash."""
        trace = Trace(
            id="tr-fixed",
            event_type="DETERMINISTIC",
            description="Test",
            intent_id="int-1",
            task_id="task-1",
        )
        log1 = ImmutableTraceLog()
        log2 = ImmutableTraceLog()
        h1 = log1.append(trace)
        h2 = log2.append(trace)
        assert h1.hash == h2.hash
        assert h1.previous_hash == h2.previous_hash

    def test_hash_depends_on_previous_hash(self):
        """Same trace at different chain positions produces different hashes."""
        trace = Trace(id="tr-same", event_type="SAME")
        log1 = ImmutableTraceLog()
        log2 = ImmutableTraceLog()
        log2.append(Trace(event_type="PRECURSOR"))
        h1 = log1.append(trace)
        h2 = log2.append(trace)
        assert h1.hash != h2.hash

    def test_hash_depends_on_sequence(self, log):
        """Same trace ID/type but at different positions produces different hashes."""
        h0 = log.append(Trace(id="tr-x", event_type="X"))
        h1 = log.append(Trace(id="tr-x", event_type="X"))
        assert h0.hash != h1.hash
        assert h0.sequence == 0
        assert h1.sequence == 1


# --- Verify Integrity -----------------------------------------


class TestVerifyIntegrity:
    """Tamper detection via chain verification."""

    def test_empty_log_is_valid(self, log):
        valid, msg = log.verify_integrity()
        assert valid is True
        assert "Empty" in msg

    def test_single_event_valid(self, log):
        log.append(Trace(event_type="A"))
        valid, msg = log.verify_integrity()
        assert valid is True
        assert "1 events verified" in msg

    def test_five_events_valid(self, populated_log):
        valid, msg = populated_log.verify_integrity()
        assert valid is True
        assert "5 events verified" in msg

    def test_large_chain_valid(self, log):
        for i in range(50):
            log.append(Trace(event_type=f"EVT_{i}", details={"idx": i}))
        valid, msg = log.verify_integrity()
        assert valid is True
        assert "50 events verified" in msg

    def test_tampered_hash_detected(self, log):
        """Changing a stored hash is detected."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        log.append(Trace(event_type="C"))
        original = log._events[1]
        log._events[1] = HashedTrace(
            trace=original.trace,
            hash="deadbeef" * 8,
            previous_hash=original.previous_hash,
            sequence=original.sequence,
        )
        valid, msg = log.verify_integrity()
        assert valid is False
        assert "Tampered" in msg

    def test_tampered_hash_at_index_zero(self, log):
        """Tampering the first event hash is detected."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        original = log._events[0]
        log._events[0] = HashedTrace(
            trace=original.trace,
            hash="abcd1234" * 8,
            previous_hash=original.previous_hash,
            sequence=original.sequence,
        )
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_tampered_previous_hash_detected(self, log):
        """Changing a previous_hash link breaks the chain."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        original = log._events[1]
        log._events[1] = HashedTrace(
            trace=original.trace,
            hash=original.hash,
            previous_hash="wrong_link_value_padding_xx" + "0" * 38,
            sequence=original.sequence,
        )
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_tampered_trace_content_detected(self, log):
        """Changing trace content while keeping the hash is detected."""
        log.append(Trace(event_type="ORIGINAL"))
        original = log._events[0]
        log._events[0] = HashedTrace(
            trace=Trace(
                id=original.trace.id,
                timestamp=original.trace.timestamp,
                event_type="TAMPERED",
            ),
            hash=original.hash,
            previous_hash=original.previous_hash,
            sequence=original.sequence,
        )
        valid, msg = log.verify_integrity()
        assert valid is False
        assert "Tampered" in msg

    def test_tampered_sequence_detected(self, log):
        """Changing a sequence number breaks hash verification."""
        log.append(Trace(event_type="A"))
        original = log._events[0]
        log._events[0] = HashedTrace(
            trace=original.trace,
            hash=original.hash,
            previous_hash=original.previous_hash,
            sequence=999,
        )
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_reordered_events_detected(self, log):
        """Swapping two events breaks the chain."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        log.append(Trace(event_type="C"))
        log._events[0], log._events[1] = log._events[1], log._events[0]
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_deleted_event_detected(self, log):
        """Removing an event from the middle breaks the chain."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        log.append(Trace(event_type="C"))
        del log._events[1]
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_inserted_event_detected(self, log):
        """Inserting a foreign event breaks the chain."""
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="C"))
        forged = HashedTrace(
            trace=Trace(event_type="FORGED"),
            hash="f" * 64,
            previous_hash=log._events[0].hash,
            sequence=1,
        )
        log._events.insert(1, forged)
        valid, msg = log.verify_integrity()
        assert valid is False

    def test_integrity_valid_after_many_appends(self, log):
        """Chain stays valid after many legitimate appends."""
        for i in range(200):
            log.append(
                Trace(
                    event_type=f"TYPE_{i % 5}",
                    description=f"Description {i}",
                    intent_id=f"int-{i % 10}",
                    details={"iteration": i, "nested": {"value": i * 2}},
                )
            )
        valid, msg = log.verify_integrity()
        assert valid is True
        assert "200 events verified" in msg


# --- Query Events ---------------------------------------------


class TestQueryEvents:
    """Event filtering by intent_id, task_id, event_type."""

    def test_get_all_events(self, populated_log):
        results = populated_log.get_events()
        assert len(results) == 5

    def test_filter_by_intent_id(self, populated_log):
        results = populated_log.get_events(intent_id="int-0")
        assert len(results) == 3

    def test_filter_by_intent_id_other(self, populated_log):
        results = populated_log.get_events(intent_id="int-1")
        assert len(results) == 2

    def test_filter_by_task_id(self, populated_log):
        results = populated_log.get_events(task_id="task-0")
        assert len(results) == 2

    def test_filter_by_event_type(self, populated_log):
        results = populated_log.get_events(event_type="TYPE_0")
        assert len(results) == 1
        assert results[0].trace.event_type == "TYPE_0"

    def test_filter_combined_intent_and_type(self, log):
        log.append(Trace(event_type="L0_CHECK", intent_id="int-1"))
        log.append(Trace(event_type="ROUTING", intent_id="int-1"))
        log.append(Trace(event_type="L0_CHECK", intent_id="int-2"))
        results = log.get_events(intent_id="int-1", event_type="L0_CHECK")
        assert len(results) == 1

    def test_filter_combined_all_three(self, log):
        log.append(Trace(event_type="A", intent_id="i1", task_id="t1"))
        log.append(Trace(event_type="A", intent_id="i1", task_id="t2"))
        log.append(Trace(event_type="B", intent_id="i1", task_id="t1"))
        log.append(Trace(event_type="A", intent_id="i2", task_id="t1"))
        results = log.get_events(intent_id="i1", task_id="t1", event_type="A")
        assert len(results) == 1

    def test_filter_no_matches(self, populated_log):
        results = populated_log.get_events(intent_id="nonexistent")
        assert len(results) == 0

    def test_filter_empty_log(self, log):
        results = log.get_events(event_type="ANY")
        assert len(results) == 0

    def test_filter_by_task_id_only(self, log):
        log.append(Trace(event_type="X", task_id="t-alpha"))
        log.append(Trace(event_type="Y", task_id="t-beta"))
        log.append(Trace(event_type="Z", task_id="t-alpha"))
        results = log.get_events(task_id="t-alpha")
        assert len(results) == 2

    def test_events_property_returns_copy(self, log):
        log.append(Trace(event_type="A"))
        events = log.events
        assert len(events) == 1
        events.clear()
        assert len(log) == 1
        assert len(log.events) == 1

    def test_events_property_empty_log(self, log):
        assert log.events == []

    def test_events_property_preserves_order(self, log):
        log.append(Trace(event_type="FIRST"))
        log.append(Trace(event_type="SECOND"))
        log.append(Trace(event_type="THIRD"))
        types = [e.trace.event_type for e in log.events]
        assert types == ["FIRST", "SECOND", "THIRD"]


# --- JSON Export ----------------------------------------------


class TestExportJson:
    """Export to JSON file for external audit tools."""

    def test_export_creates_file(self, log):
        log.append(Trace(event_type="TEST"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert "events" in data
            assert "total_events" in data
            assert "chain_head" in data
            assert "exported_at" in data
        finally:
            path.unlink(missing_ok=True)

    def test_export_total_events(self, log):
        for i in range(3):
            log.append(Trace(event_type=f"EVT_{i}"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            assert data["total_events"] == 3
            assert len(data["events"]) == 3
        finally:
            path.unlink(missing_ok=True)

    def test_export_chain_head_matches(self, log):
        log.append(Trace(event_type="A"))
        log.append(Trace(event_type="B"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            assert data["chain_head"] == log.head_hash
        finally:
            path.unlink(missing_ok=True)

    def test_export_event_structure(self, log):
        log.append(
            Trace(
                event_type="CRITIC_PIPELINE",
                description="Test",
                intent_id="int-1",
                task_id="task-1",
                risk_level=RiskLevel.CRITICAL,
                l0_passed=False,
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            event = data["events"][0]
            assert event["sequence"] == 0
            assert len(event["hash"]) == 64
            assert event["previous_hash"] == "0" * 64
            trace = event["trace"]
            assert trace["event_type"] == "CRITIC_PIPELINE"
            assert trace["description"] == "Test"
            assert trace["intent_id"] == "int-1"
            assert trace["task_id"] == "task-1"
            assert trace["risk_level"] == "CRITICAL"
            assert trace["l0_passed"] is False
        finally:
            path.unlink(missing_ok=True)

    def test_export_preserves_hash_chain(self, log):
        """Exported JSON preserves correct chain linking."""
        for i in range(5):
            log.append(Trace(event_type=f"EVT_{i}"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            events = data["events"]
            assert events[0]["previous_hash"] == "0" * 64
            for i in range(1, len(events)):
                assert events[i]["previous_hash"] == events[i - 1]["hash"]
        finally:
            path.unlink(missing_ok=True)

    def test_export_empty_log(self, log):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            assert data["total_events"] == 0
            assert data["events"] == []
            assert data["chain_head"] == "0" * 64
        finally:
            path.unlink(missing_ok=True)

    def test_export_risk_level_none(self, log):
        """Trace without risk_level exports as null."""
        log.append(Trace(event_type="NO_RISK"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            assert data["events"][0]["trace"]["risk_level"] is None
        finally:
            path.unlink(missing_ok=True)

    def test_export_accepts_string_path(self, log):
        """export_json accepts both str and Path."""
        log.append(Trace(event_type="TEST"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path_str = f.name
        try:
            log.export_json(path_str)
            data = json.loads(Path(path_str).read_text())
            assert data["total_events"] == 1
        finally:
            Path(path_str).unlink(missing_ok=True)

    def test_export_with_details(self, log):
        """Trace details dict is included in export."""
        log.append(
            Trace(
                event_type="WITH_DETAILS",
                details={"action": "AUTO_EXECUTE", "profile": "DEFAULT"},
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            data = json.loads(path.read_text())
            assert data["events"][0]["trace"]["details"] == {
                "action": "AUTO_EXECUTE",
                "profile": "DEFAULT",
            }
        finally:
            path.unlink(missing_ok=True)

    def test_export_is_valid_json(self, log):
        """Exported file is parseable JSON."""
        for i in range(10):
            log.append(
                Trace(
                    event_type=f"EVT_{i}",
                    details={"nested": {"list": [1, 2, 3]}, "bool": True},
                )
            )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            log.export_json(path)
            text = path.read_text()
            data = json.loads(text)
            assert isinstance(data, dict)
        finally:
            path.unlink(missing_ok=True)


# --- Async Operations -----------------------------------------


class TestAsyncAppend:
    """Async append with subscriber notification."""

    @pytest.mark.asyncio
    async def test_append_async_returns_hashed_trace(self, log):
        hashed = await log.append_async(Trace(event_type="ASYNC"))
        assert isinstance(hashed, HashedTrace)
        assert hashed.sequence == 0
        assert len(log) == 1

    @pytest.mark.asyncio
    async def test_append_async_chain_integrity(self, log):
        """Async append maintains correct hash chain."""
        h1 = await log.append_async(Trace(event_type="A"))
        h2 = await log.append_async(Trace(event_type="B"))
        assert h2.previous_hash == h1.hash
        valid, msg = log.verify_integrity()
        assert valid is True

    @pytest.mark.asyncio
    async def test_append_async_multiple(self, log):
        for i in range(10):
            await log.append_async(Trace(event_type=f"EVT_{i}"))
        assert len(log) == 10
        valid, _ = log.verify_integrity()
        assert valid is True

    @pytest.mark.asyncio
    async def test_append_async_head_hash_updates(self, log):
        genesis = log.head_hash
        await log.append_async(Trace(event_type="A"))
        after_one = log.head_hash
        assert genesis != after_one
        await log.append_async(Trace(event_type="B"))
        after_two = log.head_hash
        assert after_one != after_two


class TestSubscribers:
    """Subscriber notification on async append."""

    @pytest.mark.asyncio
    async def test_subscriber_notified(self, log):
        received = []

        async def callback(hashed):
            received.append(hashed)

        log.subscribe(callback)
        await log.append_async(Trace(event_type="TEST"))
        assert len(received) == 1
        assert received[0].trace.event_type == "TEST"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, log):
        received_a = []
        received_b = []

        async def callback_a(hashed):
            received_a.append(hashed)

        async def callback_b(hashed):
            received_b.append(hashed)

        log.subscribe(callback_a)
        log.subscribe(callback_b)
        await log.append_async(Trace(event_type="MULTI"))
        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_subscriber_receives_multiple_events(self, log):
        received = []

        async def callback(hashed):
            received.append(hashed)

        log.subscribe(callback)
        for i in range(5):
            await log.append_async(Trace(event_type=f"EVT_{i}"))
        assert len(received) == 5
        for i, r in enumerate(received):
            assert r.trace.event_type == f"EVT_{i}"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, log):
        received = []

        async def callback(hashed):
            received.append(hashed)

        log.subscribe(callback)
        await log.append_async(Trace(event_type="BEFORE"))
        assert len(received) == 1
        log.unsubscribe(callback)
        await log.append_async(Trace(event_type="AFTER"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_break_log(self, log):
        """A failing subscriber must not prevent the event from being logged."""

        async def bad_callback(hashed):
            raise ValueError("Subscriber crashed!")

        log.subscribe(bad_callback)
        hashed = await log.append_async(Trace(event_type="SURVIVES"))
        assert len(log) == 1
        assert hashed.trace.event_type == "SURVIVES"

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_block_others(self, log):
        """A failing subscriber should not prevent other subscribers from being notified."""
        received = []

        async def bad_callback(hashed):
            raise RuntimeError("Crash")

        async def good_callback(hashed):
            received.append(hashed)

        log.subscribe(bad_callback)
        log.subscribe(good_callback)
        await log.append_async(Trace(event_type="TEST"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_sync_append_does_not_notify(self, log):
        """Sync append() does not trigger subscriber notifications."""
        received = []

        async def callback(hashed):
            received.append(hashed)

        log.subscribe(callback)
        log.append(Trace(event_type="SYNC"))
        assert len(received) == 0
        assert len(log) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_is_safe(self, log):
        """Unsubscribing a callback that was never subscribed does not raise."""

        async def never_subscribed(hashed):
            pass

        log.unsubscribe(never_subscribed)
        assert len(log._subscribers) == 0

    @pytest.mark.asyncio
    async def test_subscriber_receives_correct_hashed_trace(self, log):
        """Subscriber receives the same HashedTrace that was returned."""
        received = []

        async def callback(hashed):
            received.append(hashed)

        log.subscribe(callback)
        returned = await log.append_async(Trace(event_type="CHECK"))
        assert len(received) == 1
        assert received[0].hash == returned.hash
        assert received[0].sequence == returned.sequence


# --- HashedTrace Model ----------------------------------------


class TestHashedTraceModel:
    """Pydantic model validation for HashedTrace."""

    def test_default_previous_hash(self):
        hashed = HashedTrace(
            trace=Trace(event_type="X"),
            hash="a" * 64,
        )
        assert hashed.previous_hash == "GENESIS"

    def test_default_sequence(self):
        hashed = HashedTrace(
            trace=Trace(event_type="X"),
            hash="b" * 64,
        )
        assert hashed.sequence == 0

    def test_serialization(self, log):
        hashed = log.append(Trace(event_type="SERIALIZE"))
        data = hashed.model_dump()
        assert "trace" in data
        assert "hash" in data
        assert "previous_hash" in data
        assert "sequence" in data

    def test_json_round_trip(self, log):
        hashed = log.append(
            Trace(event_type="RT", risk_level=RiskLevel.MEDIUM, l0_passed=True)
        )
        json_str = hashed.model_dump_json()
        restored = HashedTrace.model_validate_json(json_str)
        assert restored.hash == hashed.hash
        assert restored.previous_hash == hashed.previous_hash
        assert restored.sequence == hashed.sequence
        assert restored.trace.event_type == "RT"
        assert restored.trace.risk_level == RiskLevel.MEDIUM

    def test_hash_field_description(self):
        schema = HashedTrace.model_json_schema()
        hash_props = schema["properties"]["hash"]
        assert "SHA-256" in hash_props.get("description", "")

    def test_previous_hash_field_description(self):
        schema = HashedTrace.model_json_schema()
        prev_props = schema["properties"]["previous_hash"]
        assert "previous" in prev_props.get("description", "").lower()


# --- Edge Cases & Invariants ----------------------------------


class TestEdgeCases:
    """Boundary conditions and safety invariants."""

    def test_trace_with_empty_strings(self, log):
        log.append(
            Trace(event_type="", description="", intent_id="", task_id="")
        )
        assert len(log) == 1
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_with_large_details(self, log):
        large_details = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}
        log.append(Trace(event_type="LARGE", details=large_details))
        assert len(log) == 1
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_with_nested_details(self, log):
        nested = {"level1": {"level2": {"level3": [1, 2, {"deep": True}]}}}
        log.append(Trace(event_type="NESTED", details=nested))
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_with_unicode(self, log):
        log.append(Trace(event_type="UNICODE", description="Bezpecnost AI agentov"))
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_with_special_characters(self, log):
        log.append(Trace(
            event_type="SPECIAL",
            description="Quotes and backslash test",
        ))
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_with_all_risk_levels(self, log):
        for level in RiskLevel:
            log.append(Trace(event_type="RISK", risk_level=level))
        assert len(log) == 4
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_independent_log_instances(self):
        """Two log instances are completely independent."""
        log1 = ImmutableTraceLog()
        log2 = ImmutableTraceLog()
        log1.append(Trace(event_type="A"))
        assert len(log1) == 1
        assert len(log2) == 0
        log2.append(Trace(event_type="B"))
        log2.append(Trace(event_type="C"))
        assert len(log1) == 1
        assert len(log2) == 2

    def test_verify_after_verify(self, populated_log):
        """Calling verify multiple times is idempotent."""
        v1, m1 = populated_log.verify_integrity()
        v2, m2 = populated_log.verify_integrity()
        assert v1 == v2
        assert m1 == m2

    def test_append_after_verify(self, log):
        """Appending after verify keeps the chain valid."""
        log.append(Trace(event_type="A"))
        valid1, _ = log.verify_integrity()
        assert valid1 is True
        log.append(Trace(event_type="B"))
        valid2, _ = log.verify_integrity()
        assert valid2 is True

    def test_head_hash_never_genesis_after_append(self, log):
        """After any append, head_hash must differ from genesis."""
        log.append(Trace(event_type="X"))
        assert log.head_hash != ImmutableTraceLog.GENESIS_HASH

    def test_all_hashes_unique(self, log):
        """Every event in the chain has a unique hash."""
        for i in range(50):
            log.append(Trace(event_type=f"EVT_{i}"))
        hashes = [e.hash for e in log.events]
        assert len(hashes) == len(set(hashes))

    def test_trace_with_none_risk_and_l0(self, log):
        """Default None values for risk_level and l0_passed."""
        hashed = log.append(Trace(event_type="MINIMAL"))
        assert hashed.trace.risk_level is None
        assert hashed.trace.l0_passed is None
        valid, _ = log.verify_integrity()
        assert valid is True

    def test_trace_id_auto_generated(self, log):
        hashed = log.append(Trace(event_type="AUTO_ID"))
        assert hashed.trace.id.startswith("tr-")
        assert len(hashed.trace.id) > 3

    def test_trace_timestamp_auto_set(self, log):
        hashed = log.append(Trace(event_type="AUTO_TS"))
        assert hashed.trace.timestamp is not None


# --- Append-Only Invariant ------------------------------------


class TestAppendOnlyInvariant:
    """The log is append-only — verify the API does not support deletion or modification."""

    def test_no_delete_method(self, log):
        assert not hasattr(log, "delete")
        assert not hasattr(log, "remove")
        assert not hasattr(log, "pop")
        assert not hasattr(log, "clear")

    def test_no_modify_method(self, log):
        assert not hasattr(log, "update")
        assert not hasattr(log, "set")
        assert not hasattr(log, "replace")

    def test_events_property_returns_new_list(self, log):
        """Each call to .events returns a fresh list."""
        log.append(Trace(event_type="A"))
        list1 = log.events
        list2 = log.events
        assert list1 is not list2
        assert list1 == list2

    def test_len_matches_events_count(self, log):
        for i in range(7):
            log.append(Trace(event_type=f"EVT_{i}"))
            assert len(log) == i + 1
            assert len(log.events) == i + 1


# --- Isolation ------------------------------------------------


class TestIsolation:
    """Multiple ImmutableTraceLog instances do not share state."""

    def test_separate_chains(self):
        """Same trace appended to two logs produces same hash (identical content)."""
        from datetime import UTC, datetime
        fixed_ts = datetime(2026, 1, 1, tzinfo=UTC)
        shared_trace = Trace(id="tr-shared", event_type="SAME", timestamp=fixed_ts)
        log_a = ImmutableTraceLog()
        log_b = ImmutableTraceLog()
        ha = log_a.append(shared_trace)
        hb = log_b.append(shared_trace)
        assert ha.hash == hb.hash
        log_a.append(Trace(event_type="ONLY_A"))
        log_b.append(Trace(event_type="ONLY_B"))
        assert log_a.head_hash != log_b.head_hash
        valid_a, _ = log_a.verify_integrity()
        valid_b, _ = log_b.verify_integrity()
        assert valid_a is True
        assert valid_b is True

    def test_separate_subscriber_lists(self):
        log_a = ImmutableTraceLog()
        log_b = ImmutableTraceLog()

        async def cb(h):
            pass

        log_a.subscribe(cb)
        assert len(log_a._subscribers) == 1
        assert len(log_b._subscribers) == 0

    def test_separate_event_storage(self):
        log_a = ImmutableTraceLog()
        log_b = ImmutableTraceLog()
        log_a.append(Trace(event_type="A"))
        log_a.append(Trace(event_type="B"))
        assert len(log_a) == 2
        assert len(log_b) == 0
