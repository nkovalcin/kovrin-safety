"""
Kovrin Safety — Exception Hierarchy

Structured exceptions for the kovrin-safety middleware.

Exception hierarchy:
    KovrinSafetyError
    +-- ConstitutionalViolationError  (Layer 0 axiom violation — non-recoverable)
    +-- SafetyBlockedError            (Critic pipeline rejected action)
    +-- ScopeViolationError           (Scope boundary exceeded)
"""

from __future__ import annotations


class KovrinSafetyError(Exception):
    """Base exception for all kovrin-safety errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ConstitutionalViolationError(KovrinSafetyError):
    """Raised when a Layer 0 constitutional axiom is violated.

    This is non-recoverable — the action MUST NOT proceed.
    """

    def __init__(self, axiom_name: str, evidence: str, details: dict | None = None):
        super().__init__(
            f"Constitutional violation: {axiom_name} — {evidence}",
            details={"axiom_name": axiom_name, "evidence": evidence, **(details or {})},
        )
        self.axiom_name = axiom_name
        self.evidence = evidence


class SafetyBlockedError(KovrinSafetyError):
    """Raised when the critic pipeline rejects an action.

    Contains information about which critics failed and why.
    """

    def __init__(
        self, message: str, failed_critics: list[str] | None = None, details: dict | None = None
    ):
        super().__init__(
            message,
            details={"failed_critics": failed_critics or [], **(details or {})},
        )
        self.failed_critics = failed_critics or []


class ScopeViolationError(KovrinSafetyError):
    """Raised when an operation exceeds its authorized scope boundary."""

    def __init__(self, message: str, scope_details: dict | None = None):
        super().__init__(message, details=scope_details)
