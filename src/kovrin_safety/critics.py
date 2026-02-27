"""
Kovrin Safety — Critic Pipeline

Three specialized critics validate every action:
1. SafetyCritic — validates Layer 0 compliance (constitutional axioms)
2. FeasibilityCritic — checks if the action is achievable
3. PolicyCritic — validates organizational constraints

CriticPipeline orchestrates all three. If any critic rejects,
the action is not approved.

Both rule-based (no LLM) and Claude-powered modes are supported.
"""

from __future__ import annotations

import asyncio
import json
import re

from kovrin_safety.constitutional import ConstitutionalChecker, create_checker
from kovrin_safety.models import ProofObligation

# ─── Safety Critic ───────────────────────────────────────────


class SafetyCritic:
    """Validates actions against Layer 0 constitutional axioms.

    Delegates to ConstitutionalChecker — the single source of truth
    for safety verification.
    """

    def __init__(self, checker: ConstitutionalChecker) -> None:
        self._checker = checker

    async def evaluate(self, action: str, context: str = "") -> list[ProofObligation]:
        """Evaluate action against all 5 axioms."""
        return await self._checker.check(action, context)

    @staticmethod
    def passed(obligations: list[ProofObligation]) -> bool:
        return all(o.passed for o in obligations)


# ─── Feasibility Critic ─────────────────────────────────────


# Patterns indicating clearly infeasible actions
_INFEASIBLE_PATTERNS = re.compile(
    r"\b(travel\s+to\s+mars|time\s+travel|read\s+minds|"
    r"predict\s+future\s+exactly|create\s+perpetual\s+motion|"
    r"break\s+laws\s+of\s+physics|teleport|"
    r"access\s+classified\s+government|hack\s+into\s+pentagon)\b",
    re.IGNORECASE,
)


class FeasibilityCritic:
    """Evaluates whether an action is actually achievable.

    Rule-based mode: pattern matching for clearly infeasible actions.
    Claude mode: semantic analysis via API (requires anthropic SDK).
    """

    def __init__(
        self,
        api_key: str | None = None,
        available_tools: list[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._available_tools = available_tools or []
        self._client = None
        if api_key:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                pass

    async def evaluate(self, action: str, context: str = "") -> ProofObligation:
        """Evaluate feasibility of an action."""
        if self._client:
            return await self._evaluate_with_claude(action, context)
        return self._evaluate_rule_based(action, context)

    def _evaluate_rule_based(self, action: str, context: str = "") -> ProofObligation:
        """Pure rule-based feasibility check."""
        text = f"{action} {context}".lower()
        match = _INFEASIBLE_PATTERNS.search(text)

        if match:
            return ProofObligation(
                axiom_id=0,
                axiom_name="Feasibility",
                description="Action is achievable by AI agent",
                passed=False,
                evidence=f"Clearly infeasible: '{match.group()}'",
            )

        return ProofObligation(
            axiom_id=0,
            axiom_name="Feasibility",
            description="Action is achievable by AI agent",
            passed=True,
            evidence="No infeasibility patterns detected — action appears feasible",
        )

    async def _evaluate_with_claude(self, action: str, context: str = "") -> ProofObligation:
        """Claude-powered feasibility evaluation."""
        if not self._client:
            return self._evaluate_rule_based(action, context)

        tools_section = ""
        if self._available_tools:
            tool_list = ", ".join(self._available_tools)
            tools_section = f"\nAVAILABLE TOOLS: {tool_list}"

        prompt = f"""You are a feasibility critic for an AI agent system.

Evaluate whether this action is achievable by an AI agent.

ACTION: {action}
CONTEXT: {context or "None provided"}
{tools_section}

RULES:
1. Tasks involving analysis, summarization, writing are ALWAYS feasible.
2. Only reject if it requires capabilities that genuinely do not exist.
3. When in doubt, mark as feasible.

Respond with JSON:
{{
  "feasible": true/false,
  "reason": "Brief explanation"
}}

Return ONLY JSON, no other text."""

        response = await self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse(response.content[0].text)

    @staticmethod
    def _parse(text: str) -> ProofObligation:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
            return ProofObligation(
                axiom_id=0,
                axiom_name="Feasibility",
                description="Action is achievable by AI agent",
                passed=data.get("feasible", False),
                evidence=data.get("reason", ""),
            )
        except (json.JSONDecodeError, ValueError):
            return ProofObligation(
                axiom_id=0,
                axiom_name="Feasibility",
                description="Action is achievable by AI agent",
                passed=False,
                evidence="Failed to parse feasibility check — fail-safe rejection",
            )


# ─── Policy Critic ───────────────────────────────────────────


class PolicyCritic:
    """Validates actions against organizational constraints.

    Rule-based mode: keyword matching against constraints.
    Claude mode: semantic analysis via API (requires anthropic SDK).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client = None
        if api_key:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=api_key)
            except ImportError:
                pass

    async def evaluate(self, action: str, constraints: list[str]) -> ProofObligation:
        """Evaluate action against organizational constraints."""
        if not constraints:
            return ProofObligation(
                axiom_id=0,
                axiom_name="Policy",
                description="No constraints to check",
                passed=True,
                evidence="No policy constraints defined — auto-pass",
            )

        if self._client:
            return await self._evaluate_with_claude(action, constraints)
        return self._evaluate_rule_based(action, constraints)

    def _evaluate_rule_based(self, action: str, constraints: list[str]) -> ProofObligation:
        """Pure rule-based policy check using keyword matching."""
        action_lower = action.lower()
        violated: list[str] = []

        for constraint in constraints:
            # Extract key terms from constraint
            constraint_lower = constraint.lower()
            # Check for "do not" / "never" / "must not" patterns
            negation_match = re.search(
                r"(?:do\s+not|don'?t|never|must\s+not|shall\s+not|cannot|should\s+not)\s+(.+)",
                constraint_lower,
            )
            if negation_match:
                forbidden_phrase = negation_match.group(1).strip().rstrip(".")
                # Check if any significant words from the forbidden phrase appear in action
                forbidden_words = [
                    w for w in forbidden_phrase.split() if len(w) > 3
                ]
                if forbidden_words and any(w in action_lower for w in forbidden_words):
                    violated.append(constraint)
                continue

            # Check for "only" / "must" patterns (positive constraints)
            positive_match = re.search(
                r"(?:only|must|shall|should|require)\s+(.+)",
                constraint_lower,
            )
            if positive_match:
                # Positive constraints are harder to verify with rules
                # — pass by default, let Claude handle complex cases
                continue

        if violated:
            return ProofObligation(
                axiom_id=0,
                axiom_name="Policy",
                description="Action complies with organizational constraints",
                passed=False,
                evidence=f"Violated constraints: {violated}",
            )

        return ProofObligation(
            axiom_id=0,
            axiom_name="Policy",
            description="Action complies with organizational constraints",
            passed=True,
            evidence="No constraint violations detected",
        )

    async def _evaluate_with_claude(
        self, action: str, constraints: list[str]
    ) -> ProofObligation:
        """Claude-powered policy evaluation."""
        if not self._client:
            return self._evaluate_rule_based(action, constraints)

        constraint_list = "\n".join(f"  - {c}" for c in constraints)
        prompt = f"""You are a policy critic for an AI agent system.

Evaluate whether this action violates ANY of the user-defined constraints.

ACTION: {action}

CONSTRAINTS (must not be violated):
{constraint_list}

Respond with JSON:
{{
  "compliant": true/false,
  "violated_constraints": ["list of violated constraints, empty if compliant"],
  "reason": "Brief explanation"
}}

Return ONLY JSON, no other text."""

        response = await self._client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse(response.content[0].text)

    @staticmethod
    def _parse(text: str) -> ProofObligation:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
            violated = data.get("violated_constraints", [])
            return ProofObligation(
                axiom_id=0,
                axiom_name="Policy",
                description="Action complies with organizational constraints",
                passed=data.get("compliant", False),
                evidence=(
                    data.get("reason", "")
                    + (f" Violated: {violated}" if violated else "")
                ),
            )
        except (json.JSONDecodeError, ValueError):
            return ProofObligation(
                axiom_id=0,
                axiom_name="Policy",
                description="Action complies with organizational constraints",
                passed=False,
                evidence="Failed to parse policy check — fail-safe rejection",
            )


# ─── Critic Pipeline ────────────────────────────────────────


class CriticPipeline:
    """Orchestrates all three critics for comprehensive validation.

    An action must pass ALL critics to be approved.
    Results are aggregated into proof obligations.
    """

    def __init__(
        self,
        safety: SafetyCritic,
        feasibility: FeasibilityCritic,
        policy: PolicyCritic,
    ) -> None:
        self.safety = safety
        self.feasibility = feasibility
        self.policy = policy

    async def evaluate(
        self,
        action: str,
        constraints: list[str] | None = None,
        context: str = "",
    ) -> tuple[bool, list[ProofObligation]]:
        """Run all critics on an action.

        Returns (all_passed, list_of_obligations).
        """
        constraints = constraints or []

        # Run safety critic first (most important — L0 axioms)
        safety_obligations = await self.safety.evaluate(action, context)

        # Run feasibility and policy in parallel
        feasibility_result, policy_result = await asyncio.gather(
            self.feasibility.evaluate(action, context),
            self.policy.evaluate(action, constraints),
        )

        all_obligations = safety_obligations + [feasibility_result, policy_result]
        all_passed = all(o.passed for o in all_obligations)

        return all_passed, all_obligations

    @classmethod
    def create(
        cls,
        api_key: str | None = None,
        available_tools: list[str] | None = None,
    ) -> CriticPipeline:
        """Factory method to create a fully configured CriticPipeline.

        If api_key is provided, uses Claude for semantic analysis.
        Otherwise uses pure rule-based checking.

        Note: Constraints are passed per-call to evaluate(), not at creation time.
        This allows different constraints for different actions.
        """
        checker = create_checker(api_key=api_key)
        return cls(
            safety=SafetyCritic(checker),
            feasibility=FeasibilityCritic(
                api_key=api_key,
                available_tools=available_tools,
            ),
            policy=PolicyCritic(api_key=api_key),
        )
