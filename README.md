# kovrin-safety

**Safety middleware for AI agents** — constitutional checks, risk routing, and cryptographic audit for any agent pipeline.

3 lines of code. Zero LLM dependency. Works with LangGraph, CrewAI, or your own framework.

```python
from kovrin_safety import KovrinSafety

safety = KovrinSafety()
result = safety.check_sync("Delete all user data", risk_level="HIGH")
print(result.approved)   # False
print(result.action)     # HUMAN_APPROVAL
print(result.reasoning)  # "Harm Floor violation: ..."
```

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-377%20passed-brightgreen-brightgreen.svg)]()

---

## Why

Every AI agent framework has tools, memory, and planning. None of them have **safety as an architectural guarantee**.

- **LangGraph** — no safety layer
- **CrewAI** — basic guardrails, no audit trail
- **AutoGen** — no risk routing, no constitutional checks

`kovrin-safety` adds the missing layer: formal safety checks **before** your agent acts.

## What You Get

| Feature | Description |
|---------|-------------|
| **Constitutional Core** | 5 immutable axioms (Human Agency, Harm Floor, Transparency, Reversibility, Scope Limit) validated before every action |
| **Risk Router** | Deterministic 4x3 matrix: `(RiskLevel x SpeculationTier) -> Action`. CRITICAL always requires human approval. |
| **Merkle Audit Trail** | SHA-256 hash chain. Append-only. Tamper-evident. Export for compliance. |
| **Critic Pipeline** | SafetyCritic + FeasibilityCritic + PolicyCritic — rule-based by default, Claude-powered optional |
| **Watchdog Monitor** | 6 temporal rules, graduated containment (WARN -> PAUSE -> KILL), drift detection |
| **4 Autonomy Profiles** | DEFAULT, CAUTIOUS, AGGRESSIVE, LOCKED — one setting controls the entire safety posture |

## Install

```bash
pip install kovrin-safety
```

Optional extras:

```bash
pip install kovrin-safety[anthropic]   # Claude-powered semantic checks
pip install kovrin-safety[langchain]   # LangGraph integration
pip install kovrin-safety[crewai]      # CrewAI integration
pip install kovrin-safety[all]         # Everything
```

**Requirements:** Python 3.12+, pydantic >= 2.0

## Quick Start

### Basic Usage

```python
from kovrin_safety import KovrinSafety

safety = KovrinSafety()

# Async API
result = await safety.check(
    "Send email to all customers",
    risk_level="HIGH",
    speculation_tier="NONE",
)

if result.approved:
    execute_action()
else:
    print(f"Blocked: {result.reasoning}")
    print(f"Action: {result.action}")  # HUMAN_APPROVAL

# Sync API (same logic, no async required)
result = safety.check_sync("Analyze quarterly report", risk_level="LOW")
assert result.approved is True
```

### Autonomy Profiles

```python
# CAUTIOUS — stricter routing, more human approvals
safety = KovrinSafety(profile="CAUTIOUS")

# AGGRESSIVE — relaxed routing (CRITICAL still blocked)
safety = KovrinSafety(profile="AGGRESSIVE")

# LOCKED — everything requires human approval
safety = KovrinSafety(profile="LOCKED")
```

### Organization Constraints

```python
safety = KovrinSafety(
    constraints=[
        "Do not delete customer data",
        "Never suggest layoffs",
        "Must not share confidential information externally",
    ]
)

result = await safety.check("Delete customer records")
assert result.approved is False  # PolicyCritic catches this
```

### Audit Trail & Compliance

```python
safety = KovrinSafety()

await safety.check("Action 1", risk_level="LOW")
await safety.check("Action 2", risk_level="HIGH")

# Verify chain integrity (detects tampering)
assert safety.verify_integrity() is True

# Export compliance report (EU AI Act ready)
report = safety.export_compliance_report()
print(report["statistics"]["total_safety_checks"])  # 2
print(report["integrity"]["valid"])                  # True
print(report["profile"])                             # DEFAULT
```

### Claude-Powered Semantic Checks (Optional)

```python
# With API key, critics use Claude for semantic analysis
# Without it, everything works with pure rule-based matching
safety = KovrinSafety(anthropic_api_key="sk-ant-...")
```

## Framework Integrations

### LangGraph / LangChain

```python
from kovrin_safety import KovrinSafety
from kovrin_safety.exceptions import SafetyBlockedError
from kovrin_safety.integrations.langchain import KovrinSafetyMiddleware

safety = KovrinSafety(profile="CAUTIOUS")
middleware = KovrinSafetyMiddleware(safety, block_on_rejection=True)

# Pre-model safety gate
state = {"messages": [{"content": "Delete all databases"}]}
try:
    state = await middleware.before_model(state)
except SafetyBlockedError as e:
    print(f"Blocked: {e.failed_critics}")  # ['Harm Floor']

# Tool call safety gate
result = await middleware.wrap_tool_call(
    tool_name="file_write",
    tool_input={"path": "/etc/passwd"},
)
assert result.approved is False  # HIGH risk tool, blocked
```

### CrewAI

```python
from kovrin_safety import KovrinSafety
from kovrin_safety.integrations.crewai import kovrin_guardrail

safety = KovrinSafety()

# As a task guardrail
task = Task(
    description="Send marketing emails",
    agent=marketer,
    guardrails=[kovrin_guardrail(safety, risk_level="MEDIUM")],
)

# As a pre-execution guardrail
from kovrin_safety.integrations.crewai import kovrin_pre_guardrail

task = Task(
    description="Delete old records",
    agent=cleaner,
    guardrails=[kovrin_pre_guardrail(safety, risk_level="HIGH")],
)
```

## Risk Routing Matrix

The router maps every `(RiskLevel, SpeculationTier)` pair to a deterministic action:

| | FREE | GUARDED | NONE |
|---|---|---|---|
| **LOW** | Auto Execute | Auto Execute | Sandbox Review |
| **MEDIUM** | Auto Execute | Sandbox Review | Human Approval |
| **HIGH** | Sandbox Review | Human Approval | Human Approval |
| **CRITICAL** | Human Approval | Human Approval | Human Approval |

**Safety invariant:** CRITICAL **always** routes to HUMAN_APPROVAL. No profile, no override, no configuration can change this.

## Constitutional Axioms

Every action is validated against 5 immutable axioms:

| # | Axiom | Guarantee |
|---|-------|-----------|
| 1 | **Human Agency** | No action removes the ability for human override |
| 2 | **Harm Floor** | Expected harm never exceeds threshold |
| 3 | **Transparency** | All decisions are traceable to intent |
| 4 | **Reversibility** | Prefer reversible over irreversible actions |
| 5 | **Scope Limit** | Never exceed authorized boundary |

Axioms are protected by SHA-256 integrity hash — they cannot be modified at runtime.

## Watchdog Monitor

Real-time behavioral monitoring with 6 temporal rules:

- **NoExecutionAfterRejection** — blocks execution of previously rejected tasks (KILL)
- **ExcessiveFailureRate** — triggers when failure rate exceeds threshold (PAUSE)
- **UnexpectedEventSequence** — detects out-of-order events (WARN)
- **ExcessiveToolCallRate** — rate-limits tool calls (PAUSE)
- **ToolEscalationDetection** — detects privilege escalation in tools (WARN)
- **ToolCallAfterBlock** — catches repeated attempts after blocking (PAUSE)

```python
safety = KovrinSafety(enable_watchdog=True)
```

## API Reference

### `KovrinSafety`

```python
KovrinSafety(
    profile: str = "DEFAULT",           # DEFAULT, CAUTIOUS, AGGRESSIVE, LOCKED
    anthropic_api_key: str | None = None, # Enables Claude-powered critics
    constraints: list[str] | None = None, # Organization policy constraints
    enable_watchdog: bool = False,        # Enable real-time monitoring
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `check(action, ...)` | `SafetyResult` | Async full safety pipeline |
| `check_sync(action, ...)` | `SafetyResult` | Sync wrapper |
| `verify_integrity()` | `bool` | Verify Merkle chain integrity |
| `export_compliance_report()` | `dict` | Generate compliance report |

### `SafetyResult`

```python
SafetyResult(
    approved: bool,                          # Can the action proceed?
    action: RoutingAction,                   # AUTO_EXECUTE, SANDBOX_REVIEW, HUMAN_APPROVAL
    risk_level: RiskLevel,                   # LOW, MEDIUM, HIGH, CRITICAL
    speculation_tier: SpeculationTier,       # FREE, GUARDED, NONE
    proof_obligations: list[ProofObligation], # Evidence from critics
    trace_id: str,                           # Unique check identifier
    reasoning: str,                          # Human-readable explanation
    timestamp: datetime,                     # When the check was performed
)
```

## Development

```bash
git clone https://github.com/nkovalcin/kovrin-safety.git
cd kovrin-safety
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (377 tests)
pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
mypy src/kovrin_safety/
```

## Part of the Kovrin Ecosystem

`kovrin-safety` is the standalone safety middleware extracted from [Kovrin](https://github.com/nkovalcin/kovrin) — a safety-first AI agent orchestration framework with 1,100+ tests and TLA+ formal verification.

**Kovrin** provides the full orchestration engine (intent parsing, graph execution, MCTS exploration, speculative execution). **kovrin-safety** provides just the safety layer — plug it into any existing pipeline.

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Norbert Kovalcin](https://nkovalcin.com) at DIGITAL SPECIALISTS s.r.o.
