"""
Kovrin Safety — Core Data Models

Pydantic models and enums extracted from the Kovrin framework.
Zero internal dependencies beyond pydantic.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ─── Enums ───────────────────────────────────────────────────


class RiskLevel(StrEnum):
    """Risk classification for actions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SpeculationTier(StrEnum):
    """Three-tier classification for action reversibility.

    - FREE: Read-only, idempotent — auto-execute
    - GUARDED: Reversible — sandbox with commit/rollback
    - NONE: Irreversible — human confirmation required
    """

    FREE = "FREE"
    GUARDED = "GUARDED"
    NONE = "NONE"


class RoutingAction(StrEnum):
    """Decision from the risk router."""

    AUTO_EXECUTE = "AUTO_EXECUTE"
    SANDBOX_REVIEW = "SANDBOX_REVIEW"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"


class ContainmentLevel(StrEnum):
    """Watchdog graduated containment response."""

    WARN = "WARN"
    PAUSE = "PAUSE"
    KILL = "KILL"


class AutonomyProfile(StrEnum):
    """Named autonomy presets for risk routing overrides."""

    DEFAULT = "DEFAULT"
    CAUTIOUS = "CAUTIOUS"
    AGGRESSIVE = "AGGRESSIVE"
    LOCKED = "LOCKED"


class DriftLevel(StrEnum):
    """Severity of agent behavioral drift."""

    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ─── Proof Obligations ──────────────────────────────────────


class ProofObligation(BaseModel):
    """Result of checking an action against a single axiom or critic."""

    model_config = ConfigDict(frozen=True)

    axiom_id: int
    axiom_name: str
    description: str
    passed: bool
    evidence: str = ""


# ─── Trace Event ─────────────────────────────────────────────


class Trace(BaseModel):
    """An immutable audit event in the safety pipeline."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"tr-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    intent_id: str = ""
    task_id: str = ""
    event_type: str = ""
    description: str = ""
    details: dict = Field(default_factory=dict)
    risk_level: RiskLevel | None = None
    l0_passed: bool | None = None


# ─── Routing Decision ───────────────────────────────────────


class RoutingDecision(BaseModel):
    """Output of the risk router for a single action."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    action: RoutingAction
    risk_level: RiskLevel
    speculation_tier: SpeculationTier
    reason: str


# ─── Watchdog Alert ─────────────────────────────────────────


class WatchdogAlert(BaseModel):
    """An alert raised by the watchdog monitor."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: f"alert-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: ContainmentLevel
    reason: str
    task_id: str = ""
    intent_id: str = ""
    rule: str = ""


# ─── Autonomy Settings ──────────────────────────────────────


class AutonomySettings(BaseModel):
    """Runtime autonomy configuration for the risk router."""

    profile: AutonomyProfile = AutonomyProfile.DEFAULT
    override_matrix: dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─── Agent Drift Metrics ────────────────────────────────────


class AgentDriftMetrics(BaseModel):
    """Per-agent performance tracking for drift detection."""

    agent_name: str
    total_executions: int = 0
    recent_prm_scores: list[float] = Field(default_factory=list)
    average_prm_score: float = 0.0
    success_rate: float = 1.0
    drift_level: DriftLevel = DriftLevel.NONE
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
