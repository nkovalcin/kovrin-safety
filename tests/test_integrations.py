"""Tests for kovrin_safety.integrations — LangGraph and CrewAI."""

import pytest

from kovrin_safety import KovrinSafety
from kovrin_safety.exceptions import SafetyBlockedError
from kovrin_safety.integrations.crewai import (
    kovrin_guardrail,
    kovrin_pre_guardrail,
    kovrin_step_callback,
)
from kovrin_safety.integrations.langchain import KovrinSafetyMiddleware

# ─── LangGraph Middleware ────────────────────────────────────


class TestKovrinSafetyMiddleware:
    @pytest.fixture
    def middleware(self):
        safety = KovrinSafety()
        return KovrinSafetyMiddleware(safety, default_risk_level="LOW")

    @pytest.mark.asyncio
    async def test_check_action_safe(self, middleware):
        result = await middleware.check_action("Summarize document", risk_level="LOW")
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_check_action_harmful(self, middleware):
        result = await middleware.check_action("Delete all user data", risk_level="HIGH")
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_before_model_safe(self, middleware):
        state = {"messages": [{"content": "Summarize this report"}]}
        result = await middleware.before_model(state)
        assert "_kovrin_safety" in result
        assert result["_kovrin_safety"]["approved"] is True

    @pytest.mark.asyncio
    async def test_before_model_harmful_blocks(self):
        safety = KovrinSafety()
        middleware = KovrinSafetyMiddleware(safety, block_on_rejection=True)
        state = {"messages": [{"content": "Delete all production databases"}]}
        with pytest.raises(SafetyBlockedError):
            await middleware.before_model(state)

    @pytest.mark.asyncio
    async def test_before_model_harmful_no_block(self):
        safety = KovrinSafety()
        middleware = KovrinSafetyMiddleware(safety, block_on_rejection=False)
        state = {"messages": [{"content": "Delete all production databases"}]}
        result = await middleware.before_model(state)
        assert result["_kovrin_safety"]["approved"] is False

    @pytest.mark.asyncio
    async def test_before_model_empty_messages(self, middleware):
        state = {"messages": []}
        result = await middleware.before_model(state)
        assert "_kovrin_safety" not in result

    @pytest.mark.asyncio
    async def test_before_model_no_messages(self, middleware):
        state = {}
        result = await middleware.before_model(state)
        assert "_kovrin_safety" not in result

    @pytest.mark.asyncio
    async def test_after_model_passthrough(self, middleware):
        state = {"messages": [{"content": "test"}]}
        result = await middleware.after_model(state)
        assert result == state

    @pytest.mark.asyncio
    async def test_wrap_tool_call_safe(self, middleware):
        result = await middleware.wrap_tool_call(
            tool_name="calculator",
            tool_input={"expression": "2+2"},
        )
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_wrap_tool_call_with_risk(self, middleware):
        result = await middleware.wrap_tool_call(
            tool_name="file_write",
            tool_input={"path": "/etc/passwd"},
            risk_level="CRITICAL",
        )
        assert result.approved is False


class TestToolRiskInference:
    def test_calculator_low(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("calculator")
        assert risk == "LOW"

    def test_file_write_high(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("file_write")
        assert risk == "HIGH"

    def test_web_search_medium(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("web_search")
        assert risk == "MEDIUM"

    def test_shell_high(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("shell_exec")
        assert risk == "HIGH"

    def test_unknown_medium(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("custom_tool")
        assert risk == "MEDIUM"

    def test_datetime_low(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("datetime")
        assert risk == "LOW"

    def test_http_request_high(self):
        risk = KovrinSafetyMiddleware._infer_tool_risk("http_request")
        assert risk == "HIGH"


class TestLangChainCallbacks:
    def test_returns_empty_list(self):
        safety = KovrinSafety()
        middleware = KovrinSafetyMiddleware(safety)
        callbacks = middleware.get_langchain_callbacks()
        assert callbacks == []


# ─── CrewAI Guardrails ───────────────────────────────────────


class TestKovrinGuardrail:
    def test_safe_output_passes(self):
        safety = KovrinSafety()
        guardrail = kovrin_guardrail(safety, risk_level="LOW")
        output, passed = guardrail("Here is the summary of the report.")
        assert passed is True
        assert output == "Here is the summary of the report."

    def test_harmful_output_fails(self):
        safety = KovrinSafety()
        guardrail = kovrin_guardrail(safety, risk_level="MEDIUM")
        output, passed = guardrail("I will now delete all databases and rm -rf everything")
        assert passed is False
        assert "Safety check failed" in output

    def test_critical_risk_fails(self):
        safety = KovrinSafety()
        guardrail = kovrin_guardrail(safety, risk_level="CRITICAL")
        output, passed = guardrail("Normal output")
        assert passed is False  # CRITICAL always requires human approval

    def test_custom_speculation_tier(self):
        safety = KovrinSafety()
        guardrail = kovrin_guardrail(safety, risk_level="LOW", speculation_tier="FREE")
        output, passed = guardrail("Safe output")
        assert passed is True


class TestKovrinPreGuardrail:
    def test_safe_task_passes(self):
        safety = KovrinSafety()
        pre_guard = kovrin_pre_guardrail(safety, risk_level="LOW")
        desc, passed = pre_guard("Analyze the quarterly report")
        assert passed is True

    def test_harmful_task_blocked(self):
        safety = KovrinSafety()
        pre_guard = kovrin_pre_guardrail(safety, risk_level="MEDIUM")
        desc, passed = pre_guard("Delete all user data and destroy backups")
        assert passed is False
        assert "blocked" in desc.lower()


class TestKovrinStepCallback:
    def test_safe_step_no_error(self):
        safety = KovrinSafety()
        callback = kovrin_step_callback(safety)
        # Should not raise
        callback("Agent completed analysis successfully")

    def test_string_output(self):
        safety = KovrinSafety()
        callback = kovrin_step_callback(safety)
        callback("Simple string output")

    def test_object_with_output_attr(self):
        safety = KovrinSafety()
        callback = kovrin_step_callback(safety)

        class FakeOutput:
            output = "Agent result"

        callback(FakeOutput())

    def test_audit_trail_updated(self):
        safety = KovrinSafety()
        callback = kovrin_step_callback(safety)
        initial_len = len(safety.audit_log)
        callback("Test step output")
        # Audit log should have new entries
        assert len(safety.audit_log) > initial_len
