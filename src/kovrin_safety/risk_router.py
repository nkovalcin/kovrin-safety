"""
Kovrin Safety — Risk-Based Router

Deterministic routing matrix: (RiskLevel x SpeculationTier) -> RoutingAction.
Pure logic, zero external dependencies.

Safety invariant: CRITICAL risk ALWAYS routes to HUMAN_APPROVAL.
No profile, no override, no configuration can change this.
"""

from __future__ import annotations

import contextlib

from kovrin_safety.models import (
    AutonomyProfile,
    AutonomySettings,
    RiskLevel,
    RoutingAction,
    RoutingDecision,
    SpeculationTier,
)


class RiskRouter:
    """Routes actions based on risk level and speculation tier.

    The routing matrix is a deterministic lookup table. CRITICAL risk
    always maps to HUMAN_APPROVAL — this is a hardcoded safety guard
    that cannot be overridden by any profile or cell override.
    """

    # Routing matrix: (risk_level, speculation_tier) -> action
    _MATRIX: dict[tuple[RiskLevel, SpeculationTier], RoutingAction] = {
        (RiskLevel.LOW, SpeculationTier.FREE): RoutingAction.AUTO_EXECUTE,
        (RiskLevel.LOW, SpeculationTier.GUARDED): RoutingAction.AUTO_EXECUTE,
        (RiskLevel.LOW, SpeculationTier.NONE): RoutingAction.SANDBOX_REVIEW,
        (RiskLevel.MEDIUM, SpeculationTier.FREE): RoutingAction.AUTO_EXECUTE,
        (RiskLevel.MEDIUM, SpeculationTier.GUARDED): RoutingAction.SANDBOX_REVIEW,
        (RiskLevel.MEDIUM, SpeculationTier.NONE): RoutingAction.HUMAN_APPROVAL,
        (RiskLevel.HIGH, SpeculationTier.FREE): RoutingAction.SANDBOX_REVIEW,
        (RiskLevel.HIGH, SpeculationTier.GUARDED): RoutingAction.HUMAN_APPROVAL,
        (RiskLevel.HIGH, SpeculationTier.NONE): RoutingAction.HUMAN_APPROVAL,
        (RiskLevel.CRITICAL, SpeculationTier.FREE): RoutingAction.HUMAN_APPROVAL,
        (RiskLevel.CRITICAL, SpeculationTier.GUARDED): RoutingAction.HUMAN_APPROVAL,
        (RiskLevel.CRITICAL, SpeculationTier.NONE): RoutingAction.HUMAN_APPROVAL,
    }

    # Autonomy profile overrides
    _PROFILE_OVERRIDES: dict[
        AutonomyProfile, dict[tuple[RiskLevel, SpeculationTier], RoutingAction]
    ] = {
        AutonomyProfile.DEFAULT: {},
        AutonomyProfile.CAUTIOUS: {
            (RiskLevel.HIGH, SpeculationTier.FREE): RoutingAction.HUMAN_APPROVAL,
            (RiskLevel.MEDIUM, SpeculationTier.GUARDED): RoutingAction.HUMAN_APPROVAL,
            (RiskLevel.MEDIUM, SpeculationTier.FREE): RoutingAction.SANDBOX_REVIEW,
        },
        AutonomyProfile.AGGRESSIVE: {
            (RiskLevel.HIGH, SpeculationTier.FREE): RoutingAction.AUTO_EXECUTE,
            (RiskLevel.HIGH, SpeculationTier.GUARDED): RoutingAction.SANDBOX_REVIEW,
            (RiskLevel.MEDIUM, SpeculationTier.NONE): RoutingAction.SANDBOX_REVIEW,
            (RiskLevel.LOW, SpeculationTier.NONE): RoutingAction.AUTO_EXECUTE,
        },
        AutonomyProfile.LOCKED: {
            (risk, tier): RoutingAction.HUMAN_APPROVAL
            for risk in RiskLevel
            for tier in SpeculationTier
        },
    }

    def route(
        self,
        action_id: str,
        risk_level: RiskLevel,
        speculation_tier: SpeculationTier = SpeculationTier.GUARDED,
        settings: AutonomySettings | None = None,
    ) -> RoutingDecision:
        """Determine the routing action for an action.

        Args:
            action_id: Identifier for the action being routed.
            risk_level: Risk classification of the action.
            speculation_tier: Reversibility tier of the action.
            settings: Optional autonomy settings for runtime overrides.

        Returns:
            RoutingDecision with the determined action and reasoning.
        """
        key = (risk_level, speculation_tier)
        action = self._MATRIX.get(key, RoutingAction.HUMAN_APPROVAL)

        if settings and settings.profile != AutonomyProfile.DEFAULT:
            # Apply profile overrides
            profile_overrides = self._PROFILE_OVERRIDES.get(settings.profile, {})
            if key in profile_overrides:
                action = profile_overrides[key]

            # Apply cell-level overrides (highest priority)
            cell_key = f"{risk_level.value}:{speculation_tier.value}"
            if cell_key in settings.override_matrix:
                with contextlib.suppress(ValueError):
                    action = RoutingAction(settings.override_matrix[cell_key])

        # SAFETY GUARD: CRITICAL always requires human approval.
        # This is hardcoded and cannot be overridden.
        if risk_level == RiskLevel.CRITICAL:
            action = RoutingAction.HUMAN_APPROVAL

        reason = self._explain(risk_level, speculation_tier, action)

        return RoutingDecision(
            task_id=action_id,
            action=action,
            risk_level=risk_level,
            speculation_tier=speculation_tier,
            reason=reason,
        )

    def get_matrix(
        self, profile: AutonomyProfile = AutonomyProfile.DEFAULT
    ) -> dict[tuple[str, str], str]:
        """Return the effective routing matrix for a given profile.

        Useful for dashboard visualization and debugging.
        Returns dict of {(risk, tier): action} as strings.
        """
        result: dict[tuple[str, str], str] = {}
        profile_overrides = self._PROFILE_OVERRIDES.get(profile, {})

        for (risk, tier), action in self._MATRIX.items():
            effective = profile_overrides.get((risk, tier), action)
            # CRITICAL guard
            if risk == RiskLevel.CRITICAL:
                effective = RoutingAction.HUMAN_APPROVAL
            result[(risk.value, tier.value)] = effective.value

        return result

    @staticmethod
    def _explain(risk: RiskLevel, tier: SpeculationTier, action: RoutingAction) -> str:
        if action == RoutingAction.AUTO_EXECUTE:
            return f"Risk={risk.value}, Tier={tier.value} — safe for automatic execution"
        elif action == RoutingAction.SANDBOX_REVIEW:
            return f"Risk={risk.value}, Tier={tier.value} — requires sandbox review before commit"
        else:
            return f"Risk={risk.value}, Tier={tier.value} — requires human approval"
