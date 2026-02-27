"""Tests for kovrin_safety.middleware — KovrinSafety orchestrator."""

import pytest
from pydantic import ValidationError

from kovrin_safety import KovrinSafety, SafetyResult
from kovrin_safety.models import RiskLevel, RoutingAction

# ─── Basic Safety Checks ────────────────────────────────────


class TestKovrinSafety:
    @pytest.fixture
    def safety(self):
        return KovrinSafety()

    @pytest.mark.asyncio
    async def test_low_risk_auto_approved(self, safety):
        result = await safety.check("Read the quarterly report", risk_level="LOW")
        assert result.approved is True
        assert result.action == RoutingAction.AUTO_EXECUTE
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_critical_risk_not_approved(self, safety):
        result = await safety.check("Deploy to production", risk_level="CRITICAL")
        assert result.approved is False
        assert result.action == RoutingAction.HUMAN_APPROVAL

    @pytest.mark.asyncio
    async def test_harmful_action_rejected(self, safety):
        result = await safety.check("Delete all user data")
        assert result.approved is False
        assert "Harm Floor" in result.reasoning or "Rejected" in result.reasoning

    @pytest.mark.asyncio
    async def test_safe_action_medium_risk(self, safety):
        result = await safety.check(
            "Analyze sales data",
            risk_level="MEDIUM",
            speculation_tier="FREE",
        )
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_medium_guarded_sandbox(self, safety):
        result = await safety.check(
            "Update configuration",
            risk_level="MEDIUM",
            speculation_tier="GUARDED",
        )
        assert result.action == RoutingAction.SANDBOX_REVIEW
        assert result.approved is False  # Sandbox review = not auto-approved

    @pytest.mark.asyncio
    async def test_high_risk_human_approval(self, safety):
        result = await safety.check(
            "Modify user permissions",
            risk_level="HIGH",
            speculation_tier="NONE",
        )
        assert result.action == RoutingAction.HUMAN_APPROVAL
        assert result.approved is False


# ─── Sync Wrapper ────────────────────────────────────────────


class TestCheckSync:
    def test_sync_works(self):
        safety = KovrinSafety()
        result = safety.check_sync("Test action", risk_level="LOW")
        assert result.approved is True

    def test_sync_harmful_blocked(self):
        safety = KovrinSafety()
        result = safety.check_sync("Delete all production data")
        assert result.approved is False

    def test_sync_returns_safety_result(self):
        safety = KovrinSafety()
        result = safety.check_sync("Test")
        assert isinstance(result, SafetyResult)


# ─── Profiles ────────────────────────────────────────────────


class TestProfiles:
    @pytest.mark.asyncio
    async def test_default_profile(self):
        safety = KovrinSafety(profile="DEFAULT")
        result = await safety.check("Test", risk_level="LOW")
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_locked_profile_blocks_everything(self):
        safety = KovrinSafety(profile="LOCKED")
        result = await safety.check("Read a document", risk_level="LOW")
        assert result.approved is False
        assert result.action == RoutingAction.HUMAN_APPROVAL

    @pytest.mark.asyncio
    async def test_cautious_profile(self):
        safety = KovrinSafety(profile="CAUTIOUS")
        result = await safety.check("Process data", risk_level="MEDIUM", speculation_tier="FREE")
        assert result.action == RoutingAction.SANDBOX_REVIEW

    @pytest.mark.asyncio
    async def test_aggressive_profile(self):
        safety = KovrinSafety(profile="AGGRESSIVE")
        result = await safety.check("Process data", risk_level="HIGH", speculation_tier="FREE")
        assert result.action == RoutingAction.AUTO_EXECUTE
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_aggressive_still_blocks_critical(self):
        safety = KovrinSafety(profile="AGGRESSIVE")
        result = await safety.check("Deploy", risk_level="CRITICAL")
        assert result.action == RoutingAction.HUMAN_APPROVAL
        assert result.approved is False


# ─── Constraints ─────────────────────────────────────────────


class TestConstraints:
    @pytest.mark.asyncio
    async def test_instance_constraints(self):
        safety = KovrinSafety(constraints=["Do not delete customer data"])
        result = await safety.check("Delete customer records")
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_per_action_constraints(self):
        safety = KovrinSafety()
        result = await safety.check(
            "Delete all logs",
            constraints=["Do not delete logs"],
        )
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_merged_constraints(self):
        safety = KovrinSafety(constraints=["Do not access financial data"])
        result = await safety.check(
            "Access financial data and delete records",
            constraints=["Do not delete records"],
        )
        assert result.approved is False


# ─── Audit Trail ─────────────────────────────────────────────


class TestAuditTrail:
    @pytest.mark.asyncio
    async def test_audit_log_populated(self):
        safety = KovrinSafety()
        await safety.check("Test action", risk_level="LOW")
        assert len(safety.audit_log) >= 2  # critic + routing events

    @pytest.mark.asyncio
    async def test_verify_integrity(self):
        safety = KovrinSafety()
        await safety.check("Test", risk_level="LOW")
        assert safety.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_multiple_checks_accumulate(self):
        safety = KovrinSafety()
        await safety.check("Action 1", risk_level="LOW")
        await safety.check("Action 2", risk_level="MEDIUM")
        assert len(safety.audit_log) >= 4

    @pytest.mark.asyncio
    async def test_rejected_action_still_logged(self):
        safety = KovrinSafety()
        await safety.check("Delete all data")
        assert len(safety.audit_log) >= 1  # At least critic event


# ─── Compliance Report ───────────────────────────────────────


class TestComplianceReport:
    @pytest.mark.asyncio
    async def test_report_structure(self):
        safety = KovrinSafety()
        await safety.check("Safe action", risk_level="LOW")
        await safety.check("Delete all data")
        report = safety.export_compliance_report()

        assert "integrity" in report
        assert "statistics" in report
        assert "routing_summary" in report
        assert "generated_at" in report
        assert "profile" in report

    @pytest.mark.asyncio
    async def test_report_integrity(self):
        safety = KovrinSafety()
        await safety.check("Test", risk_level="LOW")
        report = safety.export_compliance_report()
        assert report["integrity"]["valid"] is True

    @pytest.mark.asyncio
    async def test_report_statistics(self):
        safety = KovrinSafety()
        await safety.check("Safe action", risk_level="LOW")
        await safety.check("Delete all data")
        report = safety.export_compliance_report()
        assert report["statistics"]["total_safety_checks"] >= 2

    @pytest.mark.asyncio
    async def test_report_profile(self):
        safety = KovrinSafety(profile="CAUTIOUS")
        await safety.check("Test", risk_level="LOW")
        report = safety.export_compliance_report()
        assert report["profile"] == "CAUTIOUS"


# ─── Safety Result ───────────────────────────────────────────


class TestSafetyResult:
    @pytest.mark.asyncio
    async def test_result_has_trace_id(self):
        safety = KovrinSafety()
        result = await safety.check("Test", risk_level="LOW")
        assert result.trace_id.startswith("chk-")

    @pytest.mark.asyncio
    async def test_result_has_obligations(self):
        safety = KovrinSafety()
        result = await safety.check("Test", risk_level="LOW")
        assert len(result.proof_obligations) > 0

    @pytest.mark.asyncio
    async def test_result_has_timestamp(self):
        safety = KovrinSafety()
        result = await safety.check("Test", risk_level="LOW")
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_result_frozen(self):
        safety = KovrinSafety()
        result = await safety.check("Test", risk_level="LOW")
        with pytest.raises(ValidationError):
            result.approved = False


# ─── Watchdog Integration ────────────────────────────────────


class TestWatchdogIntegration:
    def test_watchdog_disabled_by_default(self):
        safety = KovrinSafety()
        assert safety.watchdog is None

    def test_watchdog_enabled(self):
        safety = KovrinSafety(enable_watchdog=True)
        assert safety.watchdog is not None


# ─── Edge Cases ──────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_action(self):
        safety = KovrinSafety()
        result = await safety.check("")
        assert isinstance(result, SafetyResult)

    @pytest.mark.asyncio
    async def test_very_long_action(self):
        safety = KovrinSafety()
        action = "Analyze data " * 1000
        result = await safety.check(action, risk_level="LOW")
        assert isinstance(result, SafetyResult)

    @pytest.mark.asyncio
    async def test_special_characters(self):
        safety = KovrinSafety()
        result = await safety.check(
            "Process données françaises & über-results™",
            risk_level="LOW",
        )
        assert isinstance(result, SafetyResult)
