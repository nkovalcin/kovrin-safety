"""Tests for kovrin_safety.models — all enums and Pydantic models.

Covers:
- 6 enums: RiskLevel, SpeculationTier, RoutingAction, ContainmentLevel,
           AutonomyProfile, DriftLevel
- 7 models: ProofObligation, Trace, RoutingDecision, WatchdogAlert,
            AutonomySettings, AgentDriftMetrics
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kovrin_safety.models import (
    AgentDriftMetrics,
    AutonomyProfile,
    AutonomySettings,
    ContainmentLevel,
    DriftLevel,
    ProofObligation,
    RiskLevel,
    RoutingAction,
    RoutingDecision,
    SpeculationTier,
    Trace,
    WatchdogAlert,
)

# ═══════════════════════════════════════════════════════════════
# Enum Tests
# ═══════════════════════════════════════════════════════════════


class TestRiskLevel:
    """RiskLevel enum — 4 severity tiers."""

    def test_values(self) -> None:
        assert RiskLevel.LOW == "LOW"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_member_count(self) -> None:
        assert len(RiskLevel) == 4

    def test_from_string(self) -> None:
        assert RiskLevel("LOW") is RiskLevel.LOW
        assert RiskLevel("CRITICAL") is RiskLevel.CRITICAL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskLevel("INVALID")

    def test_is_str_subclass(self) -> None:
        assert isinstance(RiskLevel.LOW, str)

    def test_ordering_by_name(self) -> None:
        names = [m.name for m in RiskLevel]
        assert names == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TestSpeculationTier:
    """SpeculationTier enum — 3 reversibility tiers."""

    def test_values(self) -> None:
        assert SpeculationTier.FREE == "FREE"
        assert SpeculationTier.GUARDED == "GUARDED"
        assert SpeculationTier.NONE == "NONE"

    def test_member_count(self) -> None:
        assert len(SpeculationTier) == 3

    def test_from_string(self) -> None:
        assert SpeculationTier("FREE") is SpeculationTier.FREE
        assert SpeculationTier("NONE") is SpeculationTier.NONE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeculationTier("PARTIAL")

    def test_is_str_subclass(self) -> None:
        assert isinstance(SpeculationTier.GUARDED, str)


class TestRoutingAction:
    """RoutingAction enum — 3 possible routing decisions."""

    def test_values(self) -> None:
        assert RoutingAction.AUTO_EXECUTE == "AUTO_EXECUTE"
        assert RoutingAction.SANDBOX_REVIEW == "SANDBOX_REVIEW"
        assert RoutingAction.HUMAN_APPROVAL == "HUMAN_APPROVAL"

    def test_member_count(self) -> None:
        assert len(RoutingAction) == 3

    def test_from_string(self) -> None:
        assert RoutingAction("AUTO_EXECUTE") is RoutingAction.AUTO_EXECUTE
        assert RoutingAction("HUMAN_APPROVAL") is RoutingAction.HUMAN_APPROVAL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            RoutingAction("REJECT")

    def test_is_str_subclass(self) -> None:
        assert isinstance(RoutingAction.SANDBOX_REVIEW, str)


class TestContainmentLevel:
    """ContainmentLevel enum — graduated watchdog responses."""

    def test_values(self) -> None:
        assert ContainmentLevel.WARN == "WARN"
        assert ContainmentLevel.PAUSE == "PAUSE"
        assert ContainmentLevel.KILL == "KILL"

    def test_member_count(self) -> None:
        assert len(ContainmentLevel) == 3

    def test_from_string(self) -> None:
        assert ContainmentLevel("WARN") is ContainmentLevel.WARN
        assert ContainmentLevel("KILL") is ContainmentLevel.KILL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ContainmentLevel("SHUTDOWN")

    def test_is_str_subclass(self) -> None:
        assert isinstance(ContainmentLevel.PAUSE, str)


class TestAutonomyProfile:
    """AutonomyProfile enum — 4 autonomy presets."""

    def test_values(self) -> None:
        assert AutonomyProfile.DEFAULT == "DEFAULT"
        assert AutonomyProfile.CAUTIOUS == "CAUTIOUS"
        assert AutonomyProfile.AGGRESSIVE == "AGGRESSIVE"
        assert AutonomyProfile.LOCKED == "LOCKED"

    def test_member_count(self) -> None:
        assert len(AutonomyProfile) == 4

    def test_from_string(self) -> None:
        assert AutonomyProfile("DEFAULT") is AutonomyProfile.DEFAULT
        assert AutonomyProfile("LOCKED") is AutonomyProfile.LOCKED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            AutonomyProfile("UNSAFE")

    def test_is_str_subclass(self) -> None:
        assert isinstance(AutonomyProfile.AGGRESSIVE, str)


class TestDriftLevel:
    """DriftLevel enum — 5 severity levels for behavioral drift."""

    def test_values(self) -> None:
        assert DriftLevel.NONE == "NONE"
        assert DriftLevel.LOW == "LOW"
        assert DriftLevel.MODERATE == "MODERATE"
        assert DriftLevel.HIGH == "HIGH"
        assert DriftLevel.CRITICAL == "CRITICAL"

    def test_member_count(self) -> None:
        assert len(DriftLevel) == 5

    def test_from_string(self) -> None:
        assert DriftLevel("NONE") is DriftLevel.NONE
        assert DriftLevel("MODERATE") is DriftLevel.MODERATE
        assert DriftLevel("CRITICAL") is DriftLevel.CRITICAL

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            DriftLevel("SEVERE")

    def test_is_str_subclass(self) -> None:
        assert isinstance(DriftLevel.HIGH, str)

    def test_ordering_by_name(self) -> None:
        names = [m.name for m in DriftLevel]
        assert names == ["NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


# ═══════════════════════════════════════════════════════════════
# Pydantic Model Tests
# ═══════════════════════════════════════════════════════════════


class TestProofObligation:
    """ProofObligation — result of a single axiom/critic check."""

    def test_create_passing(self) -> None:
        po = ProofObligation(
            axiom_id=1,
            axiom_name="Human Agency",
            description="Agent must preserve human override capability",
            passed=True,
            evidence="User retains override controls",
        )
        assert po.axiom_id == 1
        assert po.axiom_name == "Human Agency"
        assert po.description == "Agent must preserve human override capability"
        assert po.passed is True
        assert po.evidence == "User retains override controls"

    def test_create_failing(self) -> None:
        po = ProofObligation(
            axiom_id=2,
            axiom_name="Harm Floor",
            description="Expected harm must not exceed threshold",
            passed=False,
            evidence="Predicted harm score 0.92 exceeds 0.5 threshold",
        )
        assert po.passed is False

    def test_default_evidence_is_empty_string(self) -> None:
        po = ProofObligation(
            axiom_id=1,
            axiom_name="Test",
            description="Desc",
            passed=True,
        )
        assert po.evidence == ""

    def test_frozen_axiom_id(self) -> None:
        po = ProofObligation(
            axiom_id=1, axiom_name="Test", description="Desc", passed=True
        )
        with pytest.raises(ValidationError):
            po.axiom_id = 2  # type: ignore[misc]

    def test_frozen_passed(self) -> None:
        po = ProofObligation(
            axiom_id=1, axiom_name="Test", description="Desc", passed=True
        )
        with pytest.raises(ValidationError):
            po.passed = False  # type: ignore[misc]

    def test_frozen_evidence(self) -> None:
        po = ProofObligation(
            axiom_id=1, axiom_name="Test", description="Desc", passed=True
        )
        with pytest.raises(ValidationError):
            po.evidence = "new"  # type: ignore[misc]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProofObligation(axiom_name="Test", description="Desc", passed=True)  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        po = ProofObligation(
            axiom_id=3,
            axiom_name="Transparency",
            description="Decisions must be traceable",
            passed=True,
            evidence="Full trace available",
        )
        data = po.model_dump()
        restored = ProofObligation(**data)
        assert restored == po

    def test_json_roundtrip(self) -> None:
        po = ProofObligation(
            axiom_id=4,
            axiom_name="Reversibility",
            description="Prefer reversible actions",
            passed=False,
            evidence="Action is irreversible",
        )
        json_str = po.model_dump_json()
        restored = ProofObligation.model_validate_json(json_str)
        assert restored == po


class TestTrace:
    """Trace — immutable audit event in the safety pipeline."""

    def test_create_with_defaults(self) -> None:
        t = Trace()
        assert t.id.startswith("tr-")
        assert len(t.id) == 11  # "tr-" + 8 hex chars
        assert t.timestamp is not None
        assert t.intent_id == ""
        assert t.task_id == ""
        assert t.event_type == ""
        assert t.description == ""
        assert t.details == {}
        assert t.risk_level is None
        assert t.l0_passed is None

    def test_create_with_all_fields(self) -> None:
        now = datetime.now(UTC)
        t = Trace(
            id="tr-custom01",
            timestamp=now,
            intent_id="intent-abc",
            task_id="task-123",
            event_type="L0_CHECK",
            description="Constitutional axiom validation",
            details={"axiom": "Human Agency", "result": "pass"},
            risk_level=RiskLevel.HIGH,
            l0_passed=True,
        )
        assert t.id == "tr-custom01"
        assert t.timestamp == now
        assert t.intent_id == "intent-abc"
        assert t.task_id == "task-123"
        assert t.event_type == "L0_CHECK"
        assert t.description == "Constitutional axiom validation"
        assert t.details["axiom"] == "Human Agency"
        assert t.risk_level is RiskLevel.HIGH
        assert t.l0_passed is True

    def test_auto_id_unique(self) -> None:
        traces = [Trace() for _ in range(100)]
        ids = {t.id for t in traces}
        assert len(ids) == 100

    def test_auto_timestamp_is_utc(self) -> None:
        t = Trace()
        assert t.timestamp.tzinfo is not None

    def test_frozen_id(self) -> None:
        t = Trace()
        with pytest.raises(ValidationError):
            t.id = "tr-new00001"  # type: ignore[misc]

    def test_frozen_event_type(self) -> None:
        t = Trace(event_type="L0_CHECK")
        with pytest.raises(ValidationError):
            t.event_type = "ROUTING"  # type: ignore[misc]

    def test_frozen_risk_level(self) -> None:
        t = Trace(risk_level=RiskLevel.LOW)
        with pytest.raises(ValidationError):
            t.risk_level = RiskLevel.HIGH  # type: ignore[misc]

    def test_frozen_l0_passed(self) -> None:
        t = Trace(l0_passed=True)
        with pytest.raises(ValidationError):
            t.l0_passed = False  # type: ignore[misc]

    def test_frozen_details(self) -> None:
        t = Trace(details={"key": "val"})
        with pytest.raises(ValidationError):
            t.details = {}  # type: ignore[misc]

    def test_risk_level_accepts_enum(self) -> None:
        t = Trace(risk_level=RiskLevel.CRITICAL)
        assert t.risk_level is RiskLevel.CRITICAL

    def test_serialization_roundtrip(self) -> None:
        t = Trace(
            event_type="ROUTING",
            risk_level=RiskLevel.MEDIUM,
            l0_passed=True,
            details={"step": 1},
        )
        data = t.model_dump()
        restored = Trace(**data)
        assert restored.event_type == t.event_type
        assert restored.risk_level == t.risk_level
        assert restored.l0_passed == t.l0_passed
        assert restored.details == t.details

    def test_json_roundtrip(self) -> None:
        t = Trace(
            event_type="AUDIT",
            description="Merkle chain append",
            risk_level=RiskLevel.LOW,
        )
        json_str = t.model_dump_json()
        restored = Trace.model_validate_json(json_str)
        assert restored.event_type == t.event_type
        assert restored.risk_level == t.risk_level


class TestRoutingDecision:
    """RoutingDecision — output of the risk router for a single action."""

    def test_create(self) -> None:
        rd = RoutingDecision(
            task_id="task-001",
            action=RoutingAction.AUTO_EXECUTE,
            risk_level=RiskLevel.LOW,
            speculation_tier=SpeculationTier.FREE,
            reason="Read-only operation, safe to auto-execute",
        )
        assert rd.task_id == "task-001"
        assert rd.action is RoutingAction.AUTO_EXECUTE
        assert rd.risk_level is RiskLevel.LOW
        assert rd.speculation_tier is SpeculationTier.FREE
        assert rd.reason == "Read-only operation, safe to auto-execute"

    def test_human_approval_decision(self) -> None:
        rd = RoutingDecision(
            task_id="task-002",
            action=RoutingAction.HUMAN_APPROVAL,
            risk_level=RiskLevel.CRITICAL,
            speculation_tier=SpeculationTier.NONE,
            reason="CRITICAL risk always requires human approval",
        )
        assert rd.action is RoutingAction.HUMAN_APPROVAL
        assert rd.risk_level is RiskLevel.CRITICAL

    def test_sandbox_review_decision(self) -> None:
        rd = RoutingDecision(
            task_id="task-003",
            action=RoutingAction.SANDBOX_REVIEW,
            risk_level=RiskLevel.MEDIUM,
            speculation_tier=SpeculationTier.GUARDED,
            reason="Reversible action with moderate risk",
        )
        assert rd.action is RoutingAction.SANDBOX_REVIEW

    def test_frozen_action(self) -> None:
        rd = RoutingDecision(
            task_id="task-001",
            action=RoutingAction.AUTO_EXECUTE,
            risk_level=RiskLevel.LOW,
            speculation_tier=SpeculationTier.FREE,
            reason="Safe",
        )
        with pytest.raises(ValidationError):
            rd.action = RoutingAction.HUMAN_APPROVAL  # type: ignore[misc]

    def test_frozen_task_id(self) -> None:
        rd = RoutingDecision(
            task_id="task-001",
            action=RoutingAction.AUTO_EXECUTE,
            risk_level=RiskLevel.LOW,
            speculation_tier=SpeculationTier.FREE,
            reason="Safe",
        )
        with pytest.raises(ValidationError):
            rd.task_id = "task-999"  # type: ignore[misc]

    def test_frozen_reason(self) -> None:
        rd = RoutingDecision(
            task_id="task-001",
            action=RoutingAction.AUTO_EXECUTE,
            risk_level=RiskLevel.LOW,
            speculation_tier=SpeculationTier.FREE,
            reason="Safe",
        )
        with pytest.raises(ValidationError):
            rd.reason = "Modified"  # type: ignore[misc]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(task_id="task-001")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        rd = RoutingDecision(
            task_id="task-004",
            action=RoutingAction.SANDBOX_REVIEW,
            risk_level=RiskLevel.HIGH,
            speculation_tier=SpeculationTier.GUARDED,
            reason="Elevated risk, sandboxing required",
        )
        data = rd.model_dump()
        restored = RoutingDecision(**data)
        assert restored == rd

    def test_json_roundtrip(self) -> None:
        rd = RoutingDecision(
            task_id="task-005",
            action=RoutingAction.HUMAN_APPROVAL,
            risk_level=RiskLevel.CRITICAL,
            speculation_tier=SpeculationTier.NONE,
            reason="Irreversible critical action",
        )
        json_str = rd.model_dump_json()
        restored = RoutingDecision.model_validate_json(json_str)
        assert restored == rd


class TestWatchdogAlert:
    """WatchdogAlert — an alert raised by the watchdog monitor."""

    def test_create_with_defaults(self) -> None:
        alert = WatchdogAlert(severity=ContainmentLevel.WARN, reason="Test alert")
        assert alert.id.startswith("alert-")
        assert len(alert.id) == 14  # "alert-" + 8 hex chars
        assert alert.timestamp is not None
        assert alert.severity is ContainmentLevel.WARN
        assert alert.reason == "Test alert"
        assert alert.task_id == ""
        assert alert.intent_id == ""
        assert alert.rule == ""

    def test_create_with_all_fields(self) -> None:
        now = datetime.now(UTC)
        alert = WatchdogAlert(
            id="alert-custom1",
            timestamp=now,
            severity=ContainmentLevel.KILL,
            reason="Agent exceeded failure threshold",
            task_id="task-abc",
            intent_id="intent-xyz",
            rule="ExcessiveFailureRate",
        )
        assert alert.id == "alert-custom1"
        assert alert.timestamp == now
        assert alert.severity is ContainmentLevel.KILL
        assert alert.reason == "Agent exceeded failure threshold"
        assert alert.task_id == "task-abc"
        assert alert.intent_id == "intent-xyz"
        assert alert.rule == "ExcessiveFailureRate"

    def test_auto_id_unique(self) -> None:
        alerts = [
            WatchdogAlert(severity=ContainmentLevel.WARN, reason="test")
            for _ in range(50)
        ]
        ids = {a.id for a in alerts}
        assert len(ids) == 50

    def test_pause_severity(self) -> None:
        alert = WatchdogAlert(severity=ContainmentLevel.PAUSE, reason="Drift detected")
        assert alert.severity is ContainmentLevel.PAUSE

    def test_frozen_severity(self) -> None:
        alert = WatchdogAlert(severity=ContainmentLevel.WARN, reason="Test")
        with pytest.raises(ValidationError):
            alert.severity = ContainmentLevel.KILL  # type: ignore[misc]

    def test_frozen_reason(self) -> None:
        alert = WatchdogAlert(severity=ContainmentLevel.WARN, reason="Test")
        with pytest.raises(ValidationError):
            alert.reason = "Modified"  # type: ignore[misc]

    def test_frozen_id(self) -> None:
        alert = WatchdogAlert(severity=ContainmentLevel.WARN, reason="Test")
        with pytest.raises(ValidationError):
            alert.id = "alert-hacked1"  # type: ignore[misc]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            WatchdogAlert(severity=ContainmentLevel.WARN)  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            WatchdogAlert(reason="Test")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        alert = WatchdogAlert(
            severity=ContainmentLevel.PAUSE,
            reason="Temporal rule violation",
            task_id="task-7",
            rule="NoExecutionAfterRejection",
        )
        data = alert.model_dump()
        restored = WatchdogAlert(**data)
        assert restored.severity == alert.severity
        assert restored.reason == alert.reason
        assert restored.task_id == alert.task_id
        assert restored.rule == alert.rule

    def test_json_roundtrip(self) -> None:
        alert = WatchdogAlert(
            severity=ContainmentLevel.KILL,
            reason="Agent killed",
            intent_id="intent-999",
        )
        json_str = alert.model_dump_json()
        restored = WatchdogAlert.model_validate_json(json_str)
        assert restored.severity == alert.severity
        assert restored.reason == alert.reason


class TestAutonomySettings:
    """AutonomySettings — runtime autonomy configuration."""

    def test_defaults(self) -> None:
        s = AutonomySettings()
        assert s.profile is AutonomyProfile.DEFAULT
        assert s.override_matrix == {}
        assert s.updated_at is not None

    def test_custom_profile(self) -> None:
        s = AutonomySettings(profile=AutonomyProfile.LOCKED)
        assert s.profile is AutonomyProfile.LOCKED

    def test_cautious_profile(self) -> None:
        s = AutonomySettings(profile=AutonomyProfile.CAUTIOUS)
        assert s.profile is AutonomyProfile.CAUTIOUS

    def test_aggressive_profile(self) -> None:
        s = AutonomySettings(profile=AutonomyProfile.AGGRESSIVE)
        assert s.profile is AutonomyProfile.AGGRESSIVE

    def test_override_matrix(self) -> None:
        overrides = {"LOW_FREE": "SANDBOX_REVIEW", "MEDIUM_GUARDED": "HUMAN_APPROVAL"}
        s = AutonomySettings(override_matrix=overrides)
        assert s.override_matrix == overrides

    def test_updated_at_is_utc(self) -> None:
        s = AutonomySettings()
        assert s.updated_at.tzinfo is not None

    def test_mutable_profile(self) -> None:
        """AutonomySettings is NOT frozen (runtime config)."""
        s = AutonomySettings()
        s.profile = AutonomyProfile.LOCKED
        assert s.profile is AutonomyProfile.LOCKED

    def test_mutable_override_matrix(self) -> None:
        s = AutonomySettings()
        s.override_matrix = {"LOW_FREE": "AUTO_EXECUTE"}
        assert "LOW_FREE" in s.override_matrix

    def test_serialization_roundtrip(self) -> None:
        s = AutonomySettings(
            profile=AutonomyProfile.CAUTIOUS,
            override_matrix={"HIGH_NONE": "HUMAN_APPROVAL"},
        )
        data = s.model_dump()
        restored = AutonomySettings(**data)
        assert restored.profile == s.profile
        assert restored.override_matrix == s.override_matrix

    def test_json_roundtrip(self) -> None:
        s = AutonomySettings(profile=AutonomyProfile.AGGRESSIVE)
        json_str = s.model_dump_json()
        restored = AutonomySettings.model_validate_json(json_str)
        assert restored.profile == s.profile


class TestAgentDriftMetrics:
    """AgentDriftMetrics — per-agent performance tracking for drift detection."""

    def test_defaults(self) -> None:
        m = AgentDriftMetrics(agent_name="test-agent")
        assert m.agent_name == "test-agent"
        assert m.total_executions == 0
        assert m.recent_prm_scores == []
        assert m.average_prm_score == 0.0
        assert m.success_rate == 1.0
        assert m.drift_level is DriftLevel.NONE
        assert m.last_updated is not None

    def test_custom_values(self) -> None:
        m = AgentDriftMetrics(
            agent_name="coordinator",
            total_executions=150,
            recent_prm_scores=[0.8, 0.75, 0.82, 0.9],
            average_prm_score=0.8175,
            success_rate=0.93,
            drift_level=DriftLevel.LOW,
        )
        assert m.total_executions == 150
        assert len(m.recent_prm_scores) == 4
        assert m.average_prm_score == 0.8175
        assert m.success_rate == 0.93
        assert m.drift_level is DriftLevel.LOW

    def test_critical_drift(self) -> None:
        m = AgentDriftMetrics(
            agent_name="rogue-agent",
            total_executions=50,
            success_rate=0.12,
            drift_level=DriftLevel.CRITICAL,
        )
        assert m.drift_level is DriftLevel.CRITICAL
        assert m.success_rate == 0.12

    def test_all_drift_levels(self) -> None:
        for level in DriftLevel:
            m = AgentDriftMetrics(agent_name="test", drift_level=level)
            assert m.drift_level is level

    def test_missing_required_agent_name(self) -> None:
        with pytest.raises(ValidationError):
            AgentDriftMetrics()  # type: ignore[call-arg]

    def test_last_updated_is_utc(self) -> None:
        m = AgentDriftMetrics(agent_name="test")
        assert m.last_updated.tzinfo is not None

    def test_mutable_total_executions(self) -> None:
        """AgentDriftMetrics is NOT frozen (tracking state)."""
        m = AgentDriftMetrics(agent_name="test")
        m.total_executions = 42
        assert m.total_executions == 42

    def test_mutable_drift_level(self) -> None:
        m = AgentDriftMetrics(agent_name="test")
        m.drift_level = DriftLevel.HIGH
        assert m.drift_level is DriftLevel.HIGH

    def test_mutable_success_rate(self) -> None:
        m = AgentDriftMetrics(agent_name="test")
        m.success_rate = 0.5
        assert m.success_rate == 0.5

    def test_mutable_recent_prm_scores(self) -> None:
        m = AgentDriftMetrics(agent_name="test")
        m.recent_prm_scores = [0.9, 0.85]
        assert m.recent_prm_scores == [0.9, 0.85]

    def test_serialization_roundtrip(self) -> None:
        m = AgentDriftMetrics(
            agent_name="serializer",
            total_executions=10,
            recent_prm_scores=[0.7, 0.8],
            average_prm_score=0.75,
            success_rate=0.9,
            drift_level=DriftLevel.MODERATE,
        )
        data = m.model_dump()
        restored = AgentDriftMetrics(**data)
        assert restored.agent_name == m.agent_name
        assert restored.total_executions == m.total_executions
        assert restored.recent_prm_scores == m.recent_prm_scores
        assert restored.drift_level == m.drift_level

    def test_json_roundtrip(self) -> None:
        m = AgentDriftMetrics(
            agent_name="json-test",
            total_executions=5,
            drift_level=DriftLevel.HIGH,
        )
        json_str = m.model_dump_json()
        restored = AgentDriftMetrics.model_validate_json(json_str)
        assert restored.agent_name == m.agent_name
        assert restored.drift_level == m.drift_level


# ═══════════════════════════════════════════════════════════════
# Cross-Model Integration Tests
# ═══════════════════════════════════════════════════════════════


class TestEnumInteroperability:
    """Verify enums work correctly when embedded in models."""

    def test_risk_level_in_trace(self) -> None:
        for level in RiskLevel:
            t = Trace(risk_level=level)
            assert t.risk_level is level

    def test_risk_level_in_routing_decision(self) -> None:
        for level in RiskLevel:
            rd = RoutingDecision(
                task_id="t",
                action=RoutingAction.AUTO_EXECUTE,
                risk_level=level,
                speculation_tier=SpeculationTier.FREE,
                reason="test",
            )
            assert rd.risk_level is level

    def test_speculation_tier_in_routing_decision(self) -> None:
        for tier in SpeculationTier:
            rd = RoutingDecision(
                task_id="t",
                action=RoutingAction.AUTO_EXECUTE,
                risk_level=RiskLevel.LOW,
                speculation_tier=tier,
                reason="test",
            )
            assert rd.speculation_tier is tier

    def test_containment_level_in_alert(self) -> None:
        for level in ContainmentLevel:
            alert = WatchdogAlert(severity=level, reason="test")
            assert alert.severity is level

    def test_autonomy_profile_in_settings(self) -> None:
        for profile in AutonomyProfile:
            s = AutonomySettings(profile=profile)
            assert s.profile is profile

    def test_drift_level_in_metrics(self) -> None:
        for level in DriftLevel:
            m = AgentDriftMetrics(agent_name="t", drift_level=level)
            assert m.drift_level is level


class TestModelSchemaExport:
    """Verify JSON schema generation works for all models."""

    def test_proof_obligation_schema(self) -> None:
        schema = ProofObligation.model_json_schema()
        assert "axiom_id" in schema["properties"]
        assert "passed" in schema["properties"]

    def test_trace_schema(self) -> None:
        schema = Trace.model_json_schema()
        assert "id" in schema["properties"]
        assert "event_type" in schema["properties"]
        assert "risk_level" in schema["properties"]

    def test_routing_decision_schema(self) -> None:
        schema = RoutingDecision.model_json_schema()
        assert "action" in schema["properties"]
        assert "risk_level" in schema["properties"]
        assert "speculation_tier" in schema["properties"]

    def test_watchdog_alert_schema(self) -> None:
        schema = WatchdogAlert.model_json_schema()
        assert "severity" in schema["properties"]
        assert "reason" in schema["properties"]

    def test_autonomy_settings_schema(self) -> None:
        schema = AutonomySettings.model_json_schema()
        assert "profile" in schema["properties"]
        assert "override_matrix" in schema["properties"]

    def test_agent_drift_metrics_schema(self) -> None:
        schema = AgentDriftMetrics.model_json_schema()
        assert "agent_name" in schema["properties"]
        assert "drift_level" in schema["properties"]
        assert "success_rate" in schema["properties"]
