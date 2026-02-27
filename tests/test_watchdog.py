"""Tests for kovrin_safety.watchdog — temporal rules and drift detection."""

import pytest

from kovrin_safety.audit import HashedTrace, ImmutableTraceLog
from kovrin_safety.models import (
    ContainmentLevel,
    DriftLevel,
    Trace,
)
from kovrin_safety.watchdog import (
    DEFAULT_RULES,
    AgentCompetencyDrift,
    AgentDriftTracker,
    CrossAgentConsistency,
    ExcessiveFailureRate,
    ExcessiveToolCallRate,
    NoExecutionAfterRejection,
    ToolCallAfterBlock,
    ToolEscalationDetection,
    UnexpectedEventSequence,
    WatchdogMonitor,
    make_drift_rules,
)


def _make_hashed(trace: Trace, seq: int = 0) -> HashedTrace:
    """Helper to create a HashedTrace for testing."""
    return HashedTrace(trace=trace, hash=f"hash_{seq}", previous_hash="prev", sequence=seq)


# ─── NoExecutionAfterRejection ───────────────────────────────


class TestNoExecutionAfterRejection:
    def test_no_alert_on_normal_execution(self):
        rule = NoExecutionAfterRejection()
        event = _make_hashed(Trace(event_type="EXECUTION_START", task_id="t1"))
        history = [
            _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=True)),
        ]
        assert rule.check(event, history) is None

    def test_alert_on_execution_after_rejection(self):
        rule = NoExecutionAfterRejection()
        event = _make_hashed(Trace(event_type="EXECUTION_START", task_id="t1"))
        history = [
            _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=False)),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.KILL

    def test_no_alert_for_different_task(self):
        rule = NoExecutionAfterRejection()
        event = _make_hashed(Trace(event_type="EXECUTION_START", task_id="t2"))
        history = [
            _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=False)),
        ]
        assert rule.check(event, history) is None

    def test_ignores_non_execution_events(self):
        rule = NoExecutionAfterRejection()
        event = _make_hashed(Trace(event_type="RISK_ROUTING", task_id="t1"))
        history = [
            _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=False)),
        ]
        assert rule.check(event, history) is None


# ─── ExcessiveFailureRate ────────────────────────────────────


class TestExcessiveFailureRate:
    def test_no_alert_below_threshold(self):
        rule = ExcessiveFailureRate(threshold=0.5)
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE"))
        history = [
            _make_hashed(Trace(event_type="EXECUTION_COMPLETE")),
            _make_hashed(Trace(event_type="EXECUTION_COMPLETE")),
            _make_hashed(Trace(event_type="CRITIC_PIPELINE", l0_passed=False)),
        ]
        alert = rule.check(event, history)
        assert alert is None  # 1/3 = 33%, below 50%

    def test_alert_above_threshold(self):
        rule = ExcessiveFailureRate(threshold=0.5)
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE"))
        history = [
            _make_hashed(Trace(event_type="EXECUTION_COMPLETE")),
            _make_hashed(Trace(event_type="CRITIC_PIPELINE", l0_passed=False)),
            _make_hashed(Trace(event_type="CRITIC_PIPELINE", l0_passed=False)),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.PAUSE

    def test_no_alert_with_few_events(self):
        rule = ExcessiveFailureRate(threshold=0.5)
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE"))
        history = [
            _make_hashed(Trace(event_type="CRITIC_PIPELINE", l0_passed=False)),
        ]
        assert rule.check(event, history) is None  # Too few events


# ─── UnexpectedEventSequence ─────────────────────────────────


class TestUnexpectedEventSequence:
    def test_complete_without_start(self):
        rule = UnexpectedEventSequence()
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE", task_id="t1"))
        history = [
            _make_hashed(Trace(event_type="L0_CHECK", task_id="t1")),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.WARN

    def test_normal_sequence(self):
        rule = UnexpectedEventSequence()
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE", task_id="t1"))
        history = [
            _make_hashed(Trace(event_type="EXECUTION_START", task_id="t1")),
        ]
        assert rule.check(event, history) is None

    def test_no_alert_on_empty_history(self):
        rule = UnexpectedEventSequence()
        event = _make_hashed(Trace(event_type="EXECUTION_COMPLETE", task_id="t1"))
        assert rule.check(event, []) is None


# ─── ExcessiveToolCallRate ───────────────────────────────────


class TestExcessiveToolCallRate:
    def test_no_alert_below_limit(self):
        rule = ExcessiveToolCallRate(max_calls_per_minute=5)
        event = _make_hashed(Trace(event_type="TOOL_CALL", intent_id="i1"))
        history = [
            _make_hashed(Trace(event_type="TOOL_CALL", intent_id="i1")),
            _make_hashed(Trace(event_type="TOOL_CALL", intent_id="i1")),
        ]
        assert rule.check(event, history) is None

    def test_ignores_non_tool_events(self):
        rule = ExcessiveToolCallRate(max_calls_per_minute=1)
        event = _make_hashed(Trace(event_type="L0_CHECK", intent_id="i1"))
        assert rule.check(event, []) is None


# ─── ToolEscalationDetection ─────────────────────────────────


class TestToolEscalationDetection:
    def test_no_alert_low_risk(self):
        rule = ToolEscalationDetection()
        event = _make_hashed(
            Trace(event_type="TOOL_CALL", task_id="t1", details={"risk_level": "LOW"})
        )
        assert rule.check(event, []) is None

    def test_alert_on_escalation(self):
        rule = ToolEscalationDetection()
        event = _make_hashed(
            Trace(event_type="TOOL_CALL", task_id="t1", details={"risk_level": "HIGH"})
        )
        history = [
            _make_hashed(
                Trace(event_type="TOOL_CALL", task_id="t1", details={"risk_level": "LOW"})
            ),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.WARN


# ─── ToolCallAfterBlock ──────────────────────────────────────


class TestToolCallAfterBlock:
    def test_first_block_no_alert(self):
        rule = ToolCallAfterBlock()
        event = _make_hashed(
            Trace(
                event_type="TOOL_CALL_BLOCKED",
                task_id="t1",
                details={"tool_name": "file_write"},
            )
        )
        assert rule.check(event, []) is None

    def test_repeated_block_alerts(self):
        rule = ToolCallAfterBlock()
        event = _make_hashed(
            Trace(
                event_type="TOOL_CALL_BLOCKED",
                task_id="t1",
                details={"tool_name": "file_write"},
            )
        )
        history = [
            _make_hashed(
                Trace(
                    event_type="TOOL_CALL_BLOCKED",
                    task_id="t1",
                    details={"tool_name": "file_write"},
                )
            ),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.PAUSE
        assert "file_write" in alert.reason


# ─── Default Rules ───────────────────────────────────────────


class TestDefaultRules:
    def test_six_default_rules(self):
        assert len(DEFAULT_RULES) == 6

    def test_rule_names_unique(self):
        names = [r.name for r in DEFAULT_RULES]
        assert len(names) == len(set(names))


# ─── Agent Drift Tracker ────────────────────────────────────


class TestAgentDriftTracker:
    def test_empty_metrics(self):
        tracker = AgentDriftTracker()
        m = tracker.get_metrics("unknown")
        assert m.total_executions == 0
        assert m.drift_level == DriftLevel.NONE

    def test_record_success(self):
        tracker = AgentDriftTracker()
        tracker.record("agent-1", "t1", prm_score=0.8, success=True)
        m = tracker.get_metrics("agent-1")
        assert m.total_executions == 1
        assert m.success_rate == 1.0

    def test_record_failure(self):
        tracker = AgentDriftTracker()
        tracker.record("agent-1", "t1", success=False)
        m = tracker.get_metrics("agent-1")
        assert m.success_rate == 0.0

    def test_drift_none_with_good_scores(self):
        tracker = AgentDriftTracker()
        for i in range(5):
            tracker.record("agent-1", f"t{i}", prm_score=0.8, success=True)
        m = tracker.get_metrics("agent-1")
        assert m.drift_level == DriftLevel.NONE

    def test_drift_critical_with_bad_scores(self):
        tracker = AgentDriftTracker()
        for i in range(5):
            tracker.record("agent-1", f"t{i}", prm_score=0.1, success=False)
        m = tracker.get_metrics("agent-1")
        assert m.drift_level == DriftLevel.CRITICAL

    def test_drift_high(self):
        tracker = AgentDriftTracker()
        for i in range(5):
            tracker.record("agent-1", f"t{i}", prm_score=0.3, success=True)
        m = tracker.get_metrics("agent-1")
        assert m.drift_level == DriftLevel.HIGH

    def test_drift_moderate(self):
        tracker = AgentDriftTracker()
        for i in range(5):
            tracker.record("agent-1", f"t{i}", prm_score=0.45, success=True)
        m = tracker.get_metrics("agent-1")
        assert m.drift_level == DriftLevel.MODERATE

    def test_drift_low(self):
        tracker = AgentDriftTracker()
        for i in range(5):
            tracker.record("agent-1", f"t{i}", prm_score=0.6, success=True)
        m = tracker.get_metrics("agent-1")
        assert m.drift_level == DriftLevel.LOW

    def test_sliding_window(self):
        tracker = AgentDriftTracker(window_size=3)
        for i in range(10):
            tracker.record("agent-1", f"t{i}", prm_score=float(i) / 10)
        m = tracker.get_metrics("agent-1")
        assert len(m.recent_prm_scores) == 3

    def test_get_all_metrics(self):
        tracker = AgentDriftTracker()
        tracker.record("agent-1", "t1", prm_score=0.5)
        tracker.record("agent-2", "t2", prm_score=0.8)
        all_m = tracker.get_all_metrics()
        assert len(all_m) == 2


# ─── CrossAgentConsistency ───────────────────────────────────


class TestCrossAgentConsistency:
    def test_no_alert_consistent(self):
        rule = CrossAgentConsistency()
        event = _make_hashed(
            Trace(
                event_type="EXECUTION_COMPLETE",
                intent_id="i1",
                task_id="t2",
                description="Task approved and success",
            )
        )
        history = [
            _make_hashed(
                Trace(
                    event_type="EXECUTION_COMPLETE",
                    intent_id="i1",
                    task_id="t1",
                    description="Result approved and valid",
                )
            ),
        ]
        assert rule.check(event, history) is None

    def test_alert_contradictory(self):
        rule = CrossAgentConsistency()
        event = _make_hashed(
            Trace(
                event_type="EXECUTION_COMPLETE",
                intent_id="i1",
                task_id="t2",
                description="This result is rejected and invalid",
            )
        )
        history = [
            _make_hashed(
                Trace(
                    event_type="EXECUTION_COMPLETE",
                    intent_id="i1",
                    task_id="t1",
                    description="Everything is approved and valid and correct",
                )
            ),
        ]
        alert = rule.check(event, history)
        assert alert is not None
        assert alert.severity == ContainmentLevel.WARN


# ─── Watchdog Monitor ────────────────────────────────────────


class TestWatchdogMonitor:
    def test_initial_state(self):
        monitor = WatchdogMonitor()
        assert not monitor.is_paused
        assert not monitor.is_killed
        assert len(monitor.alerts) == 0

    def test_check_event_safe(self):
        monitor = WatchdogMonitor()
        event = _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=True))
        alerts = monitor.check_event(event)
        assert len(alerts) == 0

    def test_check_event_triggers_kill(self):
        monitor = WatchdogMonitor()
        # First: rejection
        rejection = _make_hashed(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=False))
        monitor.check_event(rejection)

        # Then: attempted execution after rejection
        execution = _make_hashed(Trace(event_type="EXECUTION_START", task_id="t1"), seq=1)
        alerts = monitor.check_event(execution)
        assert len(alerts) > 0
        assert monitor.is_killed

    def test_pause_and_resume(self):
        monitor = WatchdogMonitor()
        # Manually trigger pause
        monitor._paused = True
        assert monitor.is_paused
        monitor.resume()
        assert not monitor.is_paused

    def test_kill_cannot_resume(self):
        monitor = WatchdogMonitor()
        monitor._killed = True
        monitor._paused = True
        monitor.resume()
        assert monitor.is_paused  # Can't resume after kill

    def test_killed_monitor_ignores_events(self):
        monitor = WatchdogMonitor()
        monitor._killed = True
        event = _make_hashed(Trace(event_type="EXECUTION_START", task_id="t1"))
        alerts = monitor.check_event(event)
        assert len(alerts) == 0

    def test_enable_drift_rules(self):
        monitor = WatchdogMonitor(enable_drift_rules=True)
        assert monitor.drift_tracker is not None
        assert len(monitor.rules) > len(DEFAULT_RULES)

    def test_disable_drift_rules(self):
        monitor = WatchdogMonitor(enable_drift_rules=False)
        assert monitor.drift_tracker is None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        monitor = WatchdogMonitor()
        log = ImmutableTraceLog()
        await monitor.start(log)
        assert monitor._trace_log is log
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_subscriber_integration(self):
        monitor = WatchdogMonitor()
        log = ImmutableTraceLog()
        await monitor.start(log)

        # Append via async — should trigger watchdog
        await log.append_async(Trace(event_type="L0_CHECK", task_id="t1", l0_passed=False))
        await log.append_async(Trace(event_type="EXECUTION_START", task_id="t1"))

        assert monitor.is_killed
        assert len(monitor.alerts) > 0
        await monitor.stop()


class TestMakeDriftRules:
    def test_returns_two_rules(self):
        rules = make_drift_rules()
        assert len(rules) == 2

    def test_custom_tracker(self):
        tracker = AgentDriftTracker(window_size=5)
        rules = make_drift_rules(tracker)
        assert isinstance(rules[0], AgentCompetencyDrift)
        assert isinstance(rules[1], CrossAgentConsistency)
