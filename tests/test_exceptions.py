"""Tests for kovrin_safety.exceptions — exception hierarchy."""

import pytest

from kovrin_safety.exceptions import (
    ConstitutionalViolationError,
    KovrinSafetyError,
    SafetyBlockedError,
    ScopeViolationError,
)


class TestKovrinSafetyError:
    def test_base_error(self):
        err = KovrinSafetyError("Test error")
        assert str(err) == "Test error"
        assert err.details == {}

    def test_base_error_with_details(self):
        err = KovrinSafetyError("Error", details={"key": "value"})
        assert err.details["key"] == "value"

    def test_is_exception(self):
        assert issubclass(KovrinSafetyError, Exception)


class TestConstitutionalViolationError:
    def test_create(self):
        err = ConstitutionalViolationError("Harm Floor", "Expected harm too high")
        assert err.axiom_name == "Harm Floor"
        assert err.evidence == "Expected harm too high"
        assert "Harm Floor" in str(err)

    def test_inherits_from_base(self):
        assert issubclass(ConstitutionalViolationError, KovrinSafetyError)

    def test_details_include_axiom(self):
        err = ConstitutionalViolationError("Scope Limit", "Out of bounds")
        assert err.details["axiom_name"] == "Scope Limit"
        assert err.details["evidence"] == "Out of bounds"


class TestSafetyBlockedError:
    def test_create(self):
        err = SafetyBlockedError("Blocked", failed_critics=["Safety", "Policy"])
        assert err.failed_critics == ["Safety", "Policy"]
        assert "Safety" in err.details["failed_critics"]

    def test_empty_critics(self):
        err = SafetyBlockedError("Blocked")
        assert err.failed_critics == []

    def test_inherits_from_base(self):
        assert issubclass(SafetyBlockedError, KovrinSafetyError)


class TestScopeViolationError:
    def test_create(self):
        err = ScopeViolationError("Exceeded scope")
        assert str(err) == "Exceeded scope"

    def test_with_scope_details(self):
        err = ScopeViolationError("Out of scope", scope_details={"max_depth": 3})
        assert err.details["max_depth"] == 3

    def test_inherits_from_base(self):
        assert issubclass(ScopeViolationError, KovrinSafetyError)


class TestExceptionHierarchy:
    def test_catch_all_with_base(self):
        """All custom exceptions should be catchable with KovrinSafetyError."""
        errors = [
            ConstitutionalViolationError("Test", "Test"),
            SafetyBlockedError("Test"),
            ScopeViolationError("Test"),
        ]
        for err in errors:
            with pytest.raises(KovrinSafetyError):
                raise err
