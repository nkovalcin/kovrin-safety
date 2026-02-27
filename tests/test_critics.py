"""Tests for kovrin_safety.critics — critic pipeline."""

import pytest

from kovrin_safety.constitutional import RuleBasedChecker
from kovrin_safety.critics import (
    CriticPipeline,
    FeasibilityCritic,
    PolicyCritic,
    SafetyCritic,
)

# ─── Safety Critic ───────────────────────────────────────────


class TestSafetyCritic:
    @pytest.fixture
    def critic(self):
        return SafetyCritic(RuleBasedChecker())

    @pytest.mark.asyncio
    async def test_safe_action_passes(self, critic):
        obligations = await critic.evaluate("Write a summary report")
        assert critic.passed(obligations)

    @pytest.mark.asyncio
    async def test_harmful_action_fails(self, critic):
        obligations = await critic.evaluate("Delete all production data")
        assert not critic.passed(obligations)

    @pytest.mark.asyncio
    async def test_returns_five_obligations(self, critic):
        obligations = await critic.evaluate("Test action")
        assert len(obligations) == 5


# ─── Feasibility Critic ─────────────────────────────────────


class TestFeasibilityCritic:
    @pytest.fixture
    def critic(self):
        return FeasibilityCritic()  # Rule-based, no API key

    @pytest.mark.asyncio
    async def test_normal_action_feasible(self, critic):
        result = await critic.evaluate("Analyze the quarterly report")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_impossible_action_infeasible(self, critic):
        result = await critic.evaluate("Travel to Mars by tomorrow")
        assert result.passed is False
        assert "infeasible" in result.evidence.lower()

    @pytest.mark.asyncio
    async def test_time_travel_infeasible(self, critic):
        result = await critic.evaluate("Use time travel to fix the bug")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_read_minds_infeasible(self, critic):
        result = await critic.evaluate("Read minds of the users to determine preferences")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_feasibility_with_context(self, critic):
        result = await critic.evaluate("Process data", context="Normal data processing")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_axiom_fields(self, critic):
        result = await critic.evaluate("Test")
        assert result.axiom_name == "Feasibility"
        assert result.axiom_id == 0


# ─── Policy Critic ───────────────────────────────────────────


class TestPolicyCritic:
    @pytest.fixture
    def critic(self):
        return PolicyCritic()  # Rule-based, no API key

    @pytest.mark.asyncio
    async def test_no_constraints_passes(self, critic):
        result = await critic.evaluate("Send emails", constraints=[])
        assert result.passed is True
        assert "auto-pass" in result.evidence.lower()

    @pytest.mark.asyncio
    async def test_compliant_action(self, critic):
        result = await critic.evaluate(
            "Send weekly newsletter to subscribers",
            constraints=["Do not delete customer data"],
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_violating_action(self, critic):
        result = await critic.evaluate(
            "Delete all customer records",
            constraints=["Do not delete customer data"],
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_multiple_constraints(self, critic):
        result = await critic.evaluate(
            "Process financial transactions",
            constraints=[
                "Do not access financial data without authorization",
                "Must log all operations",
            ],
        )
        # "financial" appears in both action and constraint
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_never_constraint(self, critic):
        result = await critic.evaluate(
            "Recommend layoffs to reduce costs",
            constraints=["Never suggest layoffs"],
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_must_not_constraint(self, critic):
        result = await critic.evaluate(
            "Share confidential information externally",
            constraints=["Must not share confidential information"],
        )
        assert result.passed is False


# ─── Critic Pipeline ────────────────────────────────────────


class TestCriticPipeline:
    @pytest.fixture
    def pipeline(self):
        return CriticPipeline.create()

    @pytest.mark.asyncio
    async def test_safe_action_passes_all(self, pipeline):
        passed, obligations = await pipeline.evaluate("Summarize the document")
        assert passed is True
        assert len(obligations) == 7  # 5 safety + 1 feasibility + 1 policy

    @pytest.mark.asyncio
    async def test_harmful_action_rejected(self, pipeline):
        passed, obligations = await pipeline.evaluate("Delete all user data")
        assert passed is False

    @pytest.mark.asyncio
    async def test_infeasible_action_rejected(self, pipeline):
        passed, obligations = await pipeline.evaluate("Travel to Mars immediately")
        assert passed is False

    @pytest.mark.asyncio
    async def test_policy_violation_rejected(self, pipeline):
        passed, obligations = await pipeline.evaluate(
            "Delete all customer records",
            constraints=["Do not delete customer data"],
        )
        assert passed is False

    @pytest.mark.asyncio
    async def test_with_context(self, pipeline):
        passed, obligations = await pipeline.evaluate(
            "Process the report",
            context="Quarterly financial analysis",
        )
        assert passed is True

    @pytest.mark.asyncio
    async def test_create_factory(self):
        pipeline = CriticPipeline.create()
        assert isinstance(pipeline.safety, SafetyCritic)
        assert isinstance(pipeline.feasibility, FeasibilityCritic)
        assert isinstance(pipeline.policy, PolicyCritic)

    @pytest.mark.asyncio
    async def test_create_with_tools(self):
        pipeline = CriticPipeline.create(available_tools=["calculator", "web_search"])
        assert pipeline.feasibility._available_tools == ["calculator", "web_search"]

    @pytest.mark.asyncio
    async def test_all_obligations_have_names(self, pipeline):
        _, obligations = await pipeline.evaluate("Test action")
        for ob in obligations:
            assert ob.axiom_name != ""
            assert ob.description != ""
