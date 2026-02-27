"""
Kovrin Safety — Main Middleware Orchestrator

The KovrinSafety class is the primary entry point for the package.
It orchestrates: constitutional check → critic pipeline → risk routing → audit.

Usage:
    from kovrin_safety import KovrinSafety

    safety = KovrinSafety()
    result = await safety.check("Send email to all customers", risk_level="HIGH")
    print(result.approved)  # False
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from kovrin_safety.audit import ImmutableTraceLog
from kovrin_safety.critics import CriticPipeline
from kovrin_safety.models import (
    AutonomyProfile,
    AutonomySettings,
    ProofObligation,
    RiskLevel,
    RoutingAction,
    SpeculationTier,
    Trace,
)
from kovrin_safety.risk_router import RiskRouter
from kovrin_safety.watchdog import WatchdogMonitor


class SafetyResult(BaseModel):
    """Result of a safety check on an action."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    action: RoutingAction
    risk_level: RiskLevel
    speculation_tier: SpeculationTier
    proof_obligations: list[ProofObligation] = Field(default_factory=list)
    trace_id: str = ""
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KovrinSafety:
    """Safety middleware for AI agents.

    Orchestrates constitutional checks, critic pipeline evaluation,
    risk-based routing, and cryptographic audit logging.

    3 lines to add safety to any AI pipeline:

        safety = KovrinSafety()
        result = safety.check_sync("Send email to all customers", risk_level="HIGH")
        if not result.approved:
            print(f"Blocked: {result.reasoning}")

    Args:
        profile: Autonomy profile — DEFAULT, CAUTIOUS, AGGRESSIVE, or LOCKED.
        anthropic_api_key: Optional API key for Claude-powered semantic checks.
            Without it, uses rule-based checking (still effective, just less nuanced).
        constraints: Organizational constraints for the PolicyCritic.
        enable_watchdog: Enable the watchdog monitor for temporal rule checking.
    """

    def __init__(
        self,
        profile: str = "DEFAULT",
        anthropic_api_key: str | None = None,
        constraints: list[str] | None = None,
        enable_watchdog: bool = False,
    ) -> None:
        self._profile = AutonomyProfile(profile)
        self._api_key = anthropic_api_key
        self._constraints = constraints or []

        # Core components
        self._router = RiskRouter()
        self._audit = ImmutableTraceLog()
        self._critics = CriticPipeline.create(
            api_key=anthropic_api_key,
        )

        # Watchdog (optional) — subscribes synchronously to the audit log
        self._watchdog: WatchdogMonitor | None = None
        if enable_watchdog:
            self._watchdog = WatchdogMonitor()
            # Register watchdog as sync subscriber via check_event
            # (audit.append_async notifies, but we also hook sync path)
            self._audit.subscribe(self._watchdog._on_event)

    async def check(
        self,
        action: str,
        risk_level: str = "MEDIUM",
        speculation_tier: str = "GUARDED",
        context: str = "",
        constraints: list[str] | None = None,
    ) -> SafetyResult:
        """Run full safety pipeline: constitutional → critics → risk router → audit.

        Args:
            action: Description of what the agent wants to do.
            risk_level: Risk classification — LOW, MEDIUM, HIGH, CRITICAL.
            speculation_tier: Reversibility tier — FREE, GUARDED, NONE.
            context: Additional context for critic evaluation.
            constraints: Per-action constraints (merged with instance constraints).

        Returns:
            SafetyResult with approval status and full audit trail.
        """
        check_id = f"chk-{uuid.uuid4().hex[:8]}"
        risk = RiskLevel(risk_level)
        tier = SpeculationTier(speculation_tier)
        all_constraints = self._constraints + (constraints or [])

        # Step 1: Critic pipeline (constitutional + feasibility + policy)
        critics_passed, obligations = await self._critics.evaluate(
            action=action,
            constraints=all_constraints,
            context=context,
        )

        # Log critic result
        critic_trace = Trace(
            intent_id=check_id,
            task_id=check_id,
            event_type="CRITIC_PIPELINE",
            description=(f"Critics: {'PASSED' if critics_passed else 'REJECTED'} — {action[:60]}"),
            details={
                "obligations": [o.model_dump() for o in obligations],
                "failed": [
                    {"name": o.axiom_name, "evidence": o.evidence}
                    for o in obligations
                    if not o.passed
                ],
            },
            risk_level=risk,
            l0_passed=critics_passed,
        )
        await self._audit.append_async(critic_trace)

        # If critics rejected, action is not approved
        if not critics_passed:
            failed_names = [o.axiom_name for o in obligations if not o.passed]
            failed_evidence = "; ".join(
                f"{o.axiom_name}: {o.evidence}" for o in obligations if not o.passed
            )
            return SafetyResult(
                approved=False,
                action=RoutingAction.HUMAN_APPROVAL,
                risk_level=risk,
                speculation_tier=tier,
                proof_obligations=obligations,
                trace_id=check_id,
                reasoning=f"Rejected by critics: {failed_names}. {failed_evidence}",
            )

        # Step 2: Risk routing
        settings = AutonomySettings(profile=self._profile)
        decision = self._router.route(
            action_id=check_id,
            risk_level=risk,
            speculation_tier=tier,
            settings=settings,
        )

        # Log routing decision
        route_trace = Trace(
            intent_id=check_id,
            task_id=check_id,
            event_type="RISK_ROUTING",
            description=f"Routed: {decision.action.value} — {action[:60]}",
            details={
                "action": decision.action.value,
                "risk_level": risk.value,
                "speculation_tier": tier.value,
                "reason": decision.reason,
                "profile": self._profile.value,
            },
            risk_level=risk,
        )
        await self._audit.append_async(route_trace)

        # Determine approval
        approved = decision.action == RoutingAction.AUTO_EXECUTE

        return SafetyResult(
            approved=approved,
            action=decision.action,
            risk_level=risk,
            speculation_tier=tier,
            proof_obligations=obligations,
            trace_id=check_id,
            reasoning=decision.reason,
        )

    def check_sync(
        self,
        action: str,
        risk_level: str = "MEDIUM",
        speculation_tier: str = "GUARDED",
        context: str = "",
        constraints: list[str] | None = None,
    ) -> SafetyResult:
        """Synchronous wrapper for check().

        For use in non-async code.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — create new loop in thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.check(action, risk_level, speculation_tier, context, constraints),
                )
                return future.result()
        else:
            return asyncio.run(
                self.check(action, risk_level, speculation_tier, context, constraints)
            )

    @property
    def audit_log(self) -> ImmutableTraceLog:
        """Access the Merkle audit trail."""
        return self._audit

    @property
    def watchdog(self) -> WatchdogMonitor | None:
        """Access the watchdog monitor (None if disabled)."""
        return self._watchdog

    def verify_integrity(self) -> bool:
        """Verify audit chain integrity."""
        valid, _ = self._audit.verify_integrity()
        return valid

    def export_compliance_report(self) -> dict:
        """Generate a compliance report.

        Returns dict with traces, integrity check, and statistics.
        Suitable for EU AI Act compliance documentation.
        """
        valid, message = self._audit.verify_integrity()
        events = self._audit.events

        total_checks = len([e for e in events if e.trace.event_type == "CRITIC_PIPELINE"])
        passed_checks = len(
            [
                e
                for e in events
                if e.trace.event_type == "CRITIC_PIPELINE" and e.trace.l0_passed is True
            ]
        )
        rejected_checks = total_checks - passed_checks

        routing_events = [e for e in events if e.trace.event_type == "RISK_ROUTING"]
        auto_count = sum(
            1
            for e in routing_events
            if e.trace.details.get("action") == RoutingAction.AUTO_EXECUTE.value
        )
        sandbox_count = sum(
            1
            for e in routing_events
            if e.trace.details.get("action") == RoutingAction.SANDBOX_REVIEW.value
        )
        human_count = sum(
            1
            for e in routing_events
            if e.trace.details.get("action") == RoutingAction.HUMAN_APPROVAL.value
        )

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "integrity": {
                "valid": valid,
                "message": message,
                "chain_head": self._audit.head_hash,
                "total_events": len(events),
            },
            "statistics": {
                "total_safety_checks": total_checks,
                "passed": passed_checks,
                "rejected": rejected_checks,
                "rejection_rate": (
                    f"{rejected_checks / total_checks:.1%}" if total_checks > 0 else "N/A"
                ),
            },
            "routing_summary": {
                "auto_execute": auto_count,
                "sandbox_review": sandbox_count,
                "human_approval": human_count,
            },
            "profile": self._profile.value,
            "watchdog_alerts": (len(self._watchdog.alerts) if self._watchdog else 0),
        }
