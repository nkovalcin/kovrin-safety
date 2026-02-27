"""
Kovrin Safety — Watchdog Monitor

Independent runtime monitor that observes execution and can
intervene when agents deviate from expected behavior.

Features:
- Temporal rule engine: detects forbidden event sequences
- Agent drift tracking: monitors PRM scores and success rates
- Graduated containment: WARN -> PAUSE -> KILL

The watchdog subscribes to the audit trace log and evaluates
every new event against its rule set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kovrin_safety.audit import HashedTrace, ImmutableTraceLog
from kovrin_safety.models import (
    AgentDriftMetrics,
    ContainmentLevel,
    DriftLevel,
    Trace,
    WatchdogAlert,
)

# ─── Temporal Rules ──────────────────────────────────────────


class TemporalRule(ABC):
    """A rule that checks sequences of trace events."""

    def __init__(self, name: str, description: str, severity: ContainmentLevel) -> None:
        self.name = name
        self.description = description
        self.severity = severity

    @abstractmethod
    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        """Check if this rule is violated. Must be implemented by subclasses."""


class NoExecutionAfterRejection(TemporalRule):
    """After an action is REJECTED by L0, no EXECUTION may follow for the same task_id."""

    def __init__(self) -> None:
        super().__init__(
            name="no_exec_after_rejection",
            description="Execution must not follow L0 rejection for the same task",
            severity=ContainmentLevel.KILL,
        )

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "EXECUTION_START":
            return None

        for past in history:
            if (
                past.trace.task_id == event.trace.task_id
                and past.trace.event_type == "L0_CHECK"
                and past.trace.l0_passed is False
            ):
                return WatchdogAlert(
                    severity=self.severity,
                    reason=(
                        f"Task {event.trace.task_id} was rejected by L0 "
                        "but execution was attempted"
                    ),
                    task_id=event.trace.task_id,
                    intent_id=event.trace.intent_id,
                    rule=self.name,
                )
        return None


class ExcessiveFailureRate(TemporalRule):
    """If more than threshold% of tasks fail, something is wrong."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__(
            name="excessive_failure_rate",
            description=f"Failure rate exceeds {threshold * 100}%",
            severity=ContainmentLevel.PAUSE,
        )
        self.threshold = threshold

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type not in ("EXECUTION_COMPLETE", "EXECUTION_START"):
            return None

        completed = sum(1 for h in history if h.trace.event_type == "EXECUTION_COMPLETE")
        failed = sum(
            1
            for h in history
            if h.trace.event_type in ("HUMAN_REJECTED",)
            or (h.trace.event_type == "CRITIC_PIPELINE" and h.trace.l0_passed is False)
        )

        total = completed + failed
        if total >= 3 and failed / total > self.threshold:
            return WatchdogAlert(
                severity=self.severity,
                reason=(
                    f"Failure rate {failed}/{total} "
                    f"({failed / total:.0%}) exceeds threshold {self.threshold:.0%}"
                ),
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        return None


class UnexpectedEventSequence(TemporalRule):
    """Detects suspicious event patterns."""

    def __init__(self) -> None:
        super().__init__(
            name="unexpected_sequence",
            description="Unexpected event sequence detected",
            severity=ContainmentLevel.WARN,
        )

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if not history:
            return None

        # Rule: EXECUTION_COMPLETE without prior EXECUTION_START for same task
        if event.trace.event_type == "EXECUTION_COMPLETE":
            has_start = any(
                h.trace.task_id == event.trace.task_id
                and h.trace.event_type == "EXECUTION_START"
                for h in history
            )
            if not has_start:
                return WatchdogAlert(
                    severity=self.severity,
                    reason=(
                        f"EXECUTION_COMPLETE without prior EXECUTION_START "
                        f"for task {event.trace.task_id}"
                    ),
                    task_id=event.trace.task_id,
                    intent_id=event.trace.intent_id,
                    rule=self.name,
                )
        return None


class ExcessiveToolCallRate(TemporalRule):
    """Alert if an agent makes too many tool calls in rapid succession."""

    def __init__(self, max_calls_per_minute: int = 30) -> None:
        super().__init__(
            name="excessive_tool_calls",
            description=f"More than {max_calls_per_minute} tool calls per minute",
            severity=ContainmentLevel.PAUSE,
        )
        self._max_calls = max_calls_per_minute

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "TOOL_CALL":
            return None

        cutoff = event.trace.timestamp.timestamp() - 60.0
        recent_calls = sum(
            1
            for h in history
            if h.trace.event_type == "TOOL_CALL"
            and h.trace.intent_id == event.trace.intent_id
            and h.trace.timestamp.timestamp() > cutoff
        )

        if recent_calls >= self._max_calls:
            return WatchdogAlert(
                severity=self.severity,
                reason=(
                    f"Excessive tool call rate: {recent_calls + 1} calls "
                    f"in last 60s (max {self._max_calls})"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        return None


class ToolEscalationDetection(TemporalRule):
    """Detect escalation patterns in tool call risk levels within a single task."""

    _RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def __init__(self) -> None:
        super().__init__(
            name="tool_escalation_detection",
            description="Tool call risk level escalation detected within same task",
            severity=ContainmentLevel.WARN,
        )

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "TOOL_CALL":
            return None

        details = event.trace.details or {}
        current_risk = details.get("risk_level", "LOW")
        current_rank = self._RISK_ORDER.get(current_risk, 0)

        if current_rank < 2:  # Only alert for HIGH+ escalation
            return None

        task_tool_risks = [
            self._RISK_ORDER.get(h.trace.details.get("risk_level", "LOW"), 0)
            for h in history
            if h.trace.event_type == "TOOL_CALL"
            and h.trace.task_id == event.trace.task_id
            and h.trace.details
        ]

        if task_tool_risks and min(task_tool_risks) < current_rank:
            return WatchdogAlert(
                severity=self.severity,
                reason=(
                    f"Tool risk escalation in task {event.trace.task_id}: "
                    f"previous min risk level escalated to {current_risk}"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        return None


class ToolCallAfterBlock(TemporalRule):
    """Detect attempts to call a tool that was previously blocked."""

    def __init__(self) -> None:
        super().__init__(
            name="tool_call_after_block",
            description="Repeated tool call after safety block",
            severity=ContainmentLevel.PAUSE,
        )

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "TOOL_CALL_BLOCKED":
            return None

        details = event.trace.details or {}
        tool_name = details.get("tool_name", "")
        if not tool_name:
            return None

        previous_blocks = sum(
            1
            for h in history
            if h.trace.event_type == "TOOL_CALL_BLOCKED"
            and h.trace.task_id == event.trace.task_id
            and (h.trace.details or {}).get("tool_name") == tool_name
        )

        if previous_blocks >= 1:
            return WatchdogAlert(
                severity=self.severity,
                reason=(
                    f"Tool '{tool_name}' blocked {previous_blocks + 1} times "
                    f"for task {event.trace.task_id} — possible safety bypass attempt"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        return None


# Default rule set
DEFAULT_RULES: list[TemporalRule] = [
    NoExecutionAfterRejection(),
    ExcessiveFailureRate(),
    UnexpectedEventSequence(),
    ExcessiveToolCallRate(),
    ToolEscalationDetection(),
    ToolCallAfterBlock(),
]


# ─── Agent Drift Detection ──────────────────────────────────


class AgentDriftTracker:
    """Tracks per-agent execution history for drift detection.

    Maintains a sliding window of PRM scores and success/failure
    counts per agent.
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._data: dict[str, dict] = {}

    def record(
        self,
        agent_name: str,
        task_id: str,
        prm_score: float | None = None,
        success: bool = True,
    ) -> None:
        """Record an execution result for an agent."""
        if agent_name not in self._data:
            self._data[agent_name] = {
                "total": 0,
                "successes": 0,
                "prm_scores": [],
            }

        data = self._data[agent_name]
        data["total"] += 1
        if success:
            data["successes"] += 1
        if prm_score is not None:
            data["prm_scores"].append(prm_score)
            if len(data["prm_scores"]) > self._window_size:
                data["prm_scores"] = data["prm_scores"][-self._window_size :]

    def get_metrics(self, agent_name: str) -> AgentDriftMetrics:
        """Get drift metrics for a specific agent."""
        data = self._data.get(agent_name)
        if not data:
            return AgentDriftMetrics(agent_name=agent_name)

        total = data["total"]
        successes = data["successes"]
        scores = data["prm_scores"]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        success_rate = successes / total if total > 0 else 1.0

        drift_level = self._compute_drift_level(avg_score, success_rate, len(scores))

        return AgentDriftMetrics(
            agent_name=agent_name,
            total_executions=total,
            recent_prm_scores=list(scores),
            average_prm_score=round(avg_score, 4),
            success_rate=round(success_rate, 4),
            drift_level=drift_level,
        )

    def get_all_metrics(self) -> list[AgentDriftMetrics]:
        """Get drift metrics for all tracked agents."""
        return [self.get_metrics(name) for name in self._data]

    @staticmethod
    def _compute_drift_level(
        avg_score: float, success_rate: float, sample_size: int
    ) -> DriftLevel:
        """Determine drift level from metrics."""
        if sample_size < 3:
            return DriftLevel.NONE
        if avg_score < 0.2 and success_rate < 0.3:
            return DriftLevel.CRITICAL
        if avg_score < 0.35 or success_rate < 0.5:
            return DriftLevel.HIGH
        if avg_score < 0.5:
            return DriftLevel.MODERATE
        if avg_score < 0.65:
            return DriftLevel.LOW
        return DriftLevel.NONE


class AgentCompetencyDrift(TemporalRule):
    """Monitors per-agent PRM scores and triggers graduated containment."""

    def __init__(self, tracker: AgentDriftTracker) -> None:
        super().__init__(
            name="agent_competency_drift",
            description="Agent performance degradation detected via PRM scores",
            severity=ContainmentLevel.WARN,
        )
        self._tracker = tracker

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "PRM_EVALUATION":
            return None

        details = event.trace.details or {}
        agent_name = details.get("agent_name", "")
        score = details.get("aggregate_score", 0.5)

        if not agent_name:
            return None

        self._tracker.record(agent_name, event.trace.task_id, prm_score=score)
        metrics = self._tracker.get_metrics(agent_name)

        if metrics.drift_level == DriftLevel.CRITICAL:
            return WatchdogAlert(
                severity=ContainmentLevel.KILL,
                reason=(
                    f"Agent '{agent_name}' critically drifted: "
                    f"avg_prm={metrics.average_prm_score:.2f}, "
                    f"success_rate={metrics.success_rate:.0%}"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        elif metrics.drift_level == DriftLevel.HIGH:
            return WatchdogAlert(
                severity=ContainmentLevel.PAUSE,
                reason=(
                    f"Agent '{agent_name}' high drift: "
                    f"avg_prm={metrics.average_prm_score:.2f}, "
                    f"success_rate={metrics.success_rate:.0%}"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        elif metrics.drift_level == DriftLevel.MODERATE:
            return WatchdogAlert(
                severity=ContainmentLevel.WARN,
                reason=(
                    f"Agent '{agent_name}' moderate drift: "
                    f"avg_prm={metrics.average_prm_score:.2f}"
                ),
                task_id=event.trace.task_id,
                intent_id=event.trace.intent_id,
                rule=self.name,
            )
        return None


class CrossAgentConsistency(TemporalRule):
    """Detects contradictory outputs from different agents in the same intent."""

    _POSITIVE = {"success", "approved", "valid", "correct", "feasible", "recommended", "agree"}
    _NEGATIVE = {
        "failure",
        "rejected",
        "invalid",
        "incorrect",
        "infeasible",
        "not recommended",
        "disagree",
    }

    def __init__(self) -> None:
        super().__init__(
            name="cross_agent_consistency",
            description="Contradictory outputs detected between agents in the same intent",
            severity=ContainmentLevel.WARN,
        )

    def check(self, event: HashedTrace, history: list[HashedTrace]) -> WatchdogAlert | None:
        if event.trace.event_type != "EXECUTION_COMPLETE":
            return None

        current_desc = (event.trace.description or "").lower()
        current_sentiment = self._sentiment(current_desc)
        if current_sentiment == 0:
            return None

        for past in history:
            if (
                past.trace.event_type == "EXECUTION_COMPLETE"
                and past.trace.intent_id == event.trace.intent_id
                and past.trace.task_id != event.trace.task_id
            ):
                past_desc = (past.trace.description or "").lower()
                past_sentiment = self._sentiment(past_desc)
                if past_sentiment != 0 and past_sentiment != current_sentiment:
                    return WatchdogAlert(
                        severity=self.severity,
                        reason=(
                            f"Contradictory outputs: task {event.trace.task_id} "
                            f"vs {past.trace.task_id}"
                        ),
                        task_id=event.trace.task_id,
                        intent_id=event.trace.intent_id,
                        rule=self.name,
                    )
        return None

    def _sentiment(self, text: str) -> int:
        """Simple keyword-based sentiment: +1 positive, -1 negative, 0 neutral."""
        pos = sum(1 for w in self._POSITIVE if w in text)
        neg = sum(1 for w in self._NEGATIVE if w in text)
        if pos > neg:
            return 1
        if neg > pos:
            return -1
        return 0


def make_drift_rules(tracker: AgentDriftTracker | None = None) -> list[TemporalRule]:
    """Create agent drift detection rules with a shared tracker."""
    tracker = tracker or AgentDriftTracker()
    return [
        AgentCompetencyDrift(tracker),
        CrossAgentConsistency(),
    ]


# ─── Watchdog Monitor ───────────────────────────────────────


class WatchdogMonitor:
    """Independent runtime monitor for safety pipelines.

    Subscribes to the trace log and evaluates every event against
    temporal rules. Can WARN, PAUSE, or KILL the pipeline.

    Unlike WatchdogAgent in Kovrin (which includes LLM-based drift detection),
    this is purely rule-based for lightweight middleware use.
    """

    def __init__(
        self,
        rules: list[TemporalRule] | None = None,
        enable_drift_rules: bool = False,
    ) -> None:
        base_rules = rules if rules is not None else list(DEFAULT_RULES)
        if enable_drift_rules:
            self._drift_tracker = AgentDriftTracker()
            base_rules = base_rules + make_drift_rules(self._drift_tracker)
        else:
            self._drift_tracker = None
        self.rules = base_rules
        self.alerts: list[WatchdogAlert] = []
        self._history: list[HashedTrace] = []
        self._paused = False
        self._killed = False
        self._trace_log: ImmutableTraceLog | None = None
        self._callback = self._on_event

    @property
    def drift_tracker(self) -> AgentDriftTracker | None:
        return self._drift_tracker

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_killed(self) -> bool:
        return self._killed

    def resume(self) -> None:
        """Resume after a PAUSE (not possible after KILL)."""
        if not self._killed:
            self._paused = False

    async def start(self, trace_log: ImmutableTraceLog) -> None:
        """Start monitoring the trace log."""
        self._trace_log = trace_log
        trace_log.subscribe(self._callback)

    async def stop(self) -> None:
        """Stop monitoring."""
        if self._trace_log is not None:
            self._trace_log.unsubscribe(self._callback)

    def check_event(self, event: HashedTrace) -> list[WatchdogAlert]:
        """Synchronously check an event against all rules.

        Useful for non-async pipelines. Returns list of alerts fired.
        """
        if self._killed:
            return []

        fired: list[WatchdogAlert] = []
        for rule in self.rules:
            alert = rule.check(event, self._history)
            if alert:
                fired.append(alert)
                self.alerts.append(alert)
                self._handle_alert(alert)

        self._history.append(event)
        return fired

    async def _on_event(self, hashed: HashedTrace) -> None:
        """Process a new trace event (async subscriber callback)."""
        self.check_event(hashed)

    def _handle_alert(self, alert: WatchdogAlert) -> None:
        """Execute containment action based on alert severity."""
        if alert.severity == ContainmentLevel.WARN:
            pass  # Log warning, continue
        elif alert.severity == ContainmentLevel.PAUSE:
            self._paused = True
        elif alert.severity == ContainmentLevel.KILL:
            self._killed = True
            self._paused = True

        # Log the alert as a trace event
        if self._trace_log:
            self._trace_log.append(
                Trace(
                    intent_id=alert.intent_id,
                    task_id=alert.task_id,
                    event_type=f"WATCHDOG_{alert.severity.value}",
                    description=f"Watchdog {alert.severity.value}: {alert.reason}",
                    details={
                        "rule": alert.rule,
                        "severity": alert.severity.value,
                        "alert_id": alert.id,
                    },
                )
            )
