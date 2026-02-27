"""
Kovrin Safety — Lightweight safety middleware for AI agents.

Constitutional checks, risk routing, and cryptographic audit trails
for any AI agent pipeline. 3 lines of code to add safety to LangGraph,
CrewAI, or any custom agent.

Usage:
    from kovrin_safety import KovrinSafety

    safety = KovrinSafety()
    result = safety.check_sync("Send email to all customers", risk_level="HIGH")
    print(result.approved)  # False — requires human approval
"""

from kovrin_safety.middleware import KovrinSafety, SafetyResult
from kovrin_safety.models import (
    AutonomyProfile,
    ContainmentLevel,
    RiskLevel,
    RoutingAction,
    SpeculationTier,
)

__all__ = [
    "KovrinSafety",
    "SafetyResult",
    "RiskLevel",
    "SpeculationTier",
    "RoutingAction",
    "AutonomyProfile",
    "ContainmentLevel",
]

__version__ = "0.1.0"
