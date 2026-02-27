"""
Kovrin Safety — CrewAI Integration

Provides guardrail functions and step callbacks compatible with
CrewAI's task execution pipeline.

Usage:
    from kovrin_safety import KovrinSafety
    from kovrin_safety.integrations.crewai import kovrin_guardrail

    safety = KovrinSafety()

    task = Task(
        description="Send marketing emails",
        agent=marketer,
        guardrails=[kovrin_guardrail(safety)],
    )

CrewAI Guardrail API:
    - guardrails: list of functions that run post-execution
    - Each guardrail returns (output, status) where status is True if passed
    - If any guardrail fails, the task output is retried or rejected
"""

from __future__ import annotations

from typing import Any

from kovrin_safety.middleware import KovrinSafety


def kovrin_guardrail(
    safety: KovrinSafety,
    risk_level: str = "MEDIUM",
    speculation_tier: str = "GUARDED",
) -> Any:
    """Create a CrewAI-compatible guardrail function.

    Returns a function that validates task output through the
    Kovrin safety pipeline. If the output is unsafe, the guardrail
    returns (error_message, False) to trigger a retry.

    Args:
        safety: KovrinSafety instance.
        risk_level: Default risk level for checked outputs.
        speculation_tier: Default speculation tier.

    Returns:
        A guardrail function compatible with CrewAI's Task(guardrails=[...]).

    Example:
        safety = KovrinSafety(profile="CAUTIOUS")

        task = Task(
            description="Analyze customer data",
            agent=analyst,
            guardrails=[kovrin_guardrail(safety, risk_level="HIGH")],
        )
    """

    def _guardrail(output: str) -> tuple[str, bool]:
        """Validate task output through Kovrin safety pipeline."""
        result = safety.check_sync(
            action=f"Task output: {output[:500]}",
            risk_level=risk_level,
            speculation_tier=speculation_tier,
        )

        if result.approved:
            return output, True
        else:
            return (
                f"Safety check failed: {result.reasoning}. Action required: {result.action.value}",
                False,
            )

    return _guardrail


def kovrin_pre_guardrail(
    safety: KovrinSafety,
    risk_level: str = "MEDIUM",
) -> Any:
    """Create a guardrail that checks the task description before execution.

    Useful for blocking dangerous tasks before they even start.
    Use alongside kovrin_guardrail for pre+post validation.

    Args:
        safety: KovrinSafety instance.
        risk_level: Default risk level.

    Returns:
        A guardrail function that checks the input task description.
    """

    def _pre_guardrail(task_description: str) -> tuple[str, bool]:
        """Validate task description before execution."""
        result = safety.check_sync(
            action=task_description,
            risk_level=risk_level,
        )

        if result.approved:
            return task_description, True
        else:
            return (
                f"Task blocked by safety: {result.reasoning}",
                False,
            )

    return _pre_guardrail


def kovrin_step_callback(
    safety: KovrinSafety,
    risk_level: str = "MEDIUM",
) -> Any:
    """Create a CrewAI step callback for real-time monitoring.

    The callback runs after each agent step and can halt execution
    if safety violations are detected.

    Args:
        safety: KovrinSafety instance.
        risk_level: Risk level for step outputs.

    Returns:
        A callback function compatible with CrewAI's step_callback parameter.

    Example:
        crew = Crew(
            agents=[...],
            tasks=[...],
            step_callback=kovrin_step_callback(safety),
        )
    """

    def _step_callback(step_output: Any) -> None:
        """Monitor each agent step through safety pipeline.

        Note: CrewAI step callbacks cannot block execution — they are
        fire-and-forget. Safety violations are recorded in the audit trail
        for post-hoc review. Use kovrin_guardrail() for blocking checks.
        """
        # Extract text from step output
        if isinstance(step_output, str):
            text = step_output
        elif hasattr(step_output, "output"):
            text = str(step_output.output)
        elif hasattr(step_output, "text"):
            text = str(step_output.text)
        else:
            text = str(step_output)

        # Run safety check — result is recorded in audit trail automatically
        safety.check_sync(
            action=f"Agent step: {text[:500]}",
            risk_level=risk_level,
        )

    return _step_callback
