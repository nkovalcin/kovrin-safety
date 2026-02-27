"""
Kovrin Safety — LangGraph/LangChain Integration

Provides middleware that hooks into LangGraph's agent execution pipeline
to add constitutional safety checks, risk routing, and audit logging.

Usage:
    from kovrin_safety import KovrinSafety
    from kovrin_safety.integrations.langchain import KovrinSafetyMiddleware

    safety = KovrinSafety(profile="CAUTIOUS")
    middleware = KovrinSafetyMiddleware(safety)

    # Use with LangGraph agent
    agent = create_react_agent(model, tools, checkpointer=checkpointer)
    # Add safety via pre/post hooks or as a custom node

LangGraph 1.0 Middleware API:
    - before_model: constitutional check before LLM call
    - after_model: audit logging after LLM response
    - wrap_tool_call: risk routing + audit for each tool call
"""

from __future__ import annotations

from typing import Any

from kovrin_safety.middleware import KovrinSafety, SafetyResult


class KovrinSafetyMiddleware:
    """LangGraph-compatible safety middleware.

    Wraps a KovrinSafety instance and provides hooks compatible
    with LangGraph's middleware pattern.

    Can be used in three ways:
    1. As a LangGraph middleware (if using LangGraph 1.0+ with middleware support)
    2. As pre/post processing in a custom LangGraph node
    3. As a standalone safety gate before/after agent actions
    """

    def __init__(
        self,
        safety: KovrinSafety,
        default_risk_level: str = "MEDIUM",
        block_on_rejection: bool = True,
    ) -> None:
        """Initialize the LangGraph safety middleware.

        Args:
            safety: KovrinSafety instance (configured with profile, API key, etc.)
            default_risk_level: Default risk level for actions without explicit risk.
            block_on_rejection: If True, raises exception on rejection.
                If False, adds rejection info to state but continues.
        """
        self.safety = safety
        self.default_risk_level = default_risk_level
        self.block_on_rejection = block_on_rejection

    async def check_action(
        self,
        action: str,
        risk_level: str | None = None,
        speculation_tier: str = "GUARDED",
        context: str = "",
    ) -> SafetyResult:
        """Check an action through the full safety pipeline.

        Use this in custom LangGraph nodes:

            @node
            async def safety_gate(state):
                result = await middleware.check_action(state["next_action"])
                if not result.approved:
                    return {"blocked": True, "reason": result.reasoning}
                return state
        """
        return await self.safety.check(
            action=action,
            risk_level=risk_level or self.default_risk_level,
            speculation_tier=speculation_tier,
            context=context,
        )

    async def before_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Pre-model hook for LangGraph middleware.

        Checks the latest user message or action against constitutional axioms.
        Returns a NEW state dict with safety metadata — never mutates the input.
        """
        # Extract action from state
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        action = ""
        if isinstance(last_message, dict):
            action = last_message.get("content", "")
        elif hasattr(last_message, "content"):
            action = str(last_message.content)

        if not action:
            return state

        result = await self.safety.check(
            action=action,
            risk_level=self.default_risk_level,
        )

        if not result.approved and self.block_on_rejection:
            from kovrin_safety.exceptions import SafetyBlockedError

            raise SafetyBlockedError(
                f"Action blocked by safety middleware: {result.reasoning}",
                failed_critics=[
                    o.axiom_name for o in result.proof_obligations if not o.passed
                ],
            )

        # Return NEW dict — never mutate the input state
        return {
            **state,
            "_kovrin_safety": {
                "approved": result.approved,
                "action": result.action.value,
                "risk_level": result.risk_level.value,
                "trace_id": result.trace_id,
                "reasoning": result.reasoning,
            },
        }

    async def after_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Post-model hook for LangGraph middleware.

        Currently a no-op passthrough — audit logging happens automatically
        in KovrinSafety.check(). Override this method to add custom
        post-model processing.
        """
        return state

    async def wrap_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        risk_level: str | None = None,
    ) -> SafetyResult:
        """Safety gate for individual tool calls.

        Use this to wrap tool execution in LangGraph:

            result = await middleware.wrap_tool_call(
                tool_name="web_search",
                tool_input={"query": "..."},
                risk_level="MEDIUM",
            )
            if result.approved:
                output = await tool.invoke(tool_input)
        """
        action = f"Tool call: {tool_name}({tool_input})"
        return await self.safety.check(
            action=action,
            risk_level=risk_level or self._infer_tool_risk(tool_name),
        )

    @staticmethod
    def _infer_tool_risk(tool_name: str) -> str:
        """Infer risk level from tool name using heuristics."""
        high_risk = {"file_write", "http_request", "code_execution", "shell", "exec"}
        medium_risk = {"web_search", "file_read", "http_get", "api_call"}
        low_risk = {"calculator", "datetime", "json_transform", "format"}

        name = tool_name.lower()
        if any(h in name for h in high_risk):
            return "HIGH"
        if any(m in name for m in medium_risk):
            return "MEDIUM"
        if any(lo in name for lo in low_risk):
            return "LOW"
        return "MEDIUM"  # Default

    def get_langchain_callbacks(self) -> list:
        """Return LangChain-compatible callback handlers.

        For use with LangChain's callback system:

            llm = ChatAnthropic(callbacks=middleware.get_langchain_callbacks())

        Note: This is a convenience method. For full safety coverage,
        use the middleware hooks (before_model, wrap_tool_call) instead.
        """
        # Return empty list — callbacks provide less control than middleware
        # Users should prefer the middleware pattern
        return []
