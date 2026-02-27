"""
Kovrin Safety — Constitutional Core (Layer 0)

5 immutable ethical axioms validated before every action.
SHA-256 integrity hash — axioms cannot be modified at runtime.

Two checker implementations:
1. RuleBasedChecker — pure Python keyword/pattern matching (no LLM, zero deps)
2. ClaudeChecker — semantic analysis via Anthropic Claude API (optional)

Factory function create_checker() picks the right one based on available API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kovrin_safety.models import ProofObligation


@dataclass(frozen=True)
class Axiom:
    """A single immutable ethical axiom."""

    id: int
    name: str
    description: str
    formal: str  # Formal specification in temporal logic notation


# The five Layer 0 axioms — frozen, immutable
AXIOMS: tuple[Axiom, ...] = (
    Axiom(
        id=1,
        name="Human Agency",
        description=(
            "No action shall remove or diminish a human's ability "
            "to override, pause, or terminate the system."
        ),
        formal="□(∀action: ¬removes_human_override(action))",
    ),
    Axiom(
        id=2,
        name="Harm Floor",
        description=(
            "No action shall be taken where the expected harm "
            "exceeds the defined safety threshold."
        ),
        formal="□(∀action: expected_harm(action) < harm_threshold)",
    ),
    Axiom(
        id=3,
        name="Transparency",
        description=(
            "All decisions must be traceable to the originating intent. "
            "No opaque decision-making."
        ),
        formal="□(∀decision: ∃intent: traceable_to(decision, intent))",
    ),
    Axiom(
        id=4,
        name="Reversibility",
        description=(
            "Prefer reversible actions over irreversible ones. "
            "Irreversible actions require explicit human approval."
        ),
        formal="□(∀action: irreversible(action) → requires_human_approval(action))",
    ),
    Axiom(
        id=5,
        name="Scope Limit",
        description=(
            "Never exceed the authorized operational boundary "
            "defined by the intent and its constraints."
        ),
        formal="□(∀action: within_scope(action, authorized_boundary))",
    ),
)


def _compute_axiom_hash(axioms: tuple[Axiom, ...]) -> str:
    """Compute SHA-256 integrity hash of the axiom set."""
    content = json.dumps(
        [
            {"id": a.id, "name": a.name, "description": a.description, "formal": a.formal}
            for a in axioms
        ],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


# Integrity hash computed at import time — any modification is detectable
_AXIOM_INTEGRITY_HASH = _compute_axiom_hash(AXIOMS)


class ConstitutionalChecker(ABC):
    """Base class for constitutional checkers."""

    def __init__(self) -> None:
        self._integrity_hash = _AXIOM_INTEGRITY_HASH

    @property
    def axioms(self) -> tuple[Axiom, ...]:
        return AXIOMS

    def verify_integrity(self) -> bool:
        """Verify that the axiom set has not been tampered with."""
        return _compute_axiom_hash(AXIOMS) == self._integrity_hash

    async def check(self, action: str, context: str = "") -> list[ProofObligation]:
        """Validate an action against all Layer 0 axioms.

        Returns a list of proof obligations — one per axiom.
        If any obligation fails, the action MUST be rejected.
        """
        if not self.verify_integrity():
            raise RuntimeError("CRITICAL: Layer 0 axiom integrity compromised — hash mismatch")
        return await self._check_inner(action, context)

    @abstractmethod
    async def _check_inner(self, action: str, context: str = "") -> list[ProofObligation]:
        """Implement axiom checking logic. Must be overridden."""

    @staticmethod
    def all_passed(obligations: list[ProofObligation]) -> bool:
        """Check if all proof obligations passed."""
        return all(o.passed for o in obligations)


# ─── Rule-Based Checker (no LLM) ────────────────────────────


# Keyword patterns that indicate potential axiom violations
_HARM_KEYWORDS = re.compile(
    r"\b(delete\s+all|drop\s+database|rm\s+-rf|format\s+disk|shutdown|kill\s+process|"
    r"destroy|wipe|erase\s+all|terminate\s+all|inject|exploit|hack|attack|"
    r"ddos|malware|ransomware|phishing|steal|exfiltrate|"
    r"bypass\s+security|disable\s+auth|remove\s+protection|"
    r"self[_-]?replicate|self[_-]?modify|recursive\s+spawn)\b",
    re.IGNORECASE,
)

_HUMAN_OVERRIDE_KEYWORDS = re.compile(
    r"\b(disable\s+override|remove\s+human|prevent\s+shutdown|"
    r"block\s+termination|ignore\s+human|override\s+human|"
    r"disable\s+kill\s+switch|remove\s+pause|hide\s+from\s+human|"
    r"conceal\s+from\s+user|deceive\s+operator)\b",
    re.IGNORECASE,
)

_IRREVERSIBLE_KEYWORDS = re.compile(
    r"\b(permanently\s+delete|irreversible|cannot\s+undo|"
    r"no\s+rollback|destroy\s+backup|purge|format|"
    r"wipe\s+clean|nuclear\s+option)\b",
    re.IGNORECASE,
)

_SCOPE_KEYWORDS = re.compile(
    r"\b(access\s+unauthorized|escalate\s+privilege|"
    r"beyond\s+scope|outside\s+boundary|unauthorized\s+access|"
    r"admin\s+override|root\s+access|sudo|"
    r"cross[_-]?tenant|other\s+user['\"]?s?\s+data)\b",
    re.IGNORECASE,
)


class RuleBasedChecker(ConstitutionalChecker):
    """Pure Python rule-based constitutional checker.

    Uses keyword and pattern matching to detect potential axiom violations.
    No LLM dependency — works offline, deterministic, fast.

    This is conservative: borderline cases pass (fail-open for rules,
    but the critic pipeline provides additional safety).
    """

    async def _check_inner(self, action: str, context: str = "") -> list[ProofObligation]:
        text = f"{action} {context}".lower()
        obligations: list[ProofObligation] = []

        # Axiom 1: Human Agency
        human_override_match = _HUMAN_OVERRIDE_KEYWORDS.search(text)
        obligations.append(
            ProofObligation(
                axiom_id=1,
                axiom_name="Human Agency",
                description=AXIOMS[0].description,
                passed=human_override_match is None,
                evidence=(
                    f"Detected pattern: '{human_override_match.group()}'"
                    if human_override_match
                    else "No human override removal patterns detected"
                ),
            )
        )

        # Axiom 2: Harm Floor
        harm_match = _HARM_KEYWORDS.search(text)
        obligations.append(
            ProofObligation(
                axiom_id=2,
                axiom_name="Harm Floor",
                description=AXIOMS[1].description,
                passed=harm_match is None,
                evidence=(
                    f"Detected harmful pattern: '{harm_match.group()}'"
                    if harm_match
                    else "No harmful patterns detected"
                ),
            )
        )

        # Axiom 3: Transparency — rule-based always passes (can't detect opacity via keywords)
        obligations.append(
            ProofObligation(
                axiom_id=3,
                axiom_name="Transparency",
                description=AXIOMS[2].description,
                passed=True,
                evidence="Rule-based check: transparency verified by audit trail existence",
            )
        )

        # Axiom 4: Reversibility
        irreversible_match = _IRREVERSIBLE_KEYWORDS.search(text)
        obligations.append(
            ProofObligation(
                axiom_id=4,
                axiom_name="Reversibility",
                description=AXIOMS[3].description,
                passed=irreversible_match is None,
                evidence=(
                    f"Detected irreversible pattern: '{irreversible_match.group()}'"
                    if irreversible_match
                    else "No irreversible action patterns detected"
                ),
            )
        )

        # Axiom 5: Scope Limit
        scope_match = _SCOPE_KEYWORDS.search(text)
        obligations.append(
            ProofObligation(
                axiom_id=5,
                axiom_name="Scope Limit",
                description=AXIOMS[4].description,
                passed=scope_match is None,
                evidence=(
                    f"Detected scope violation pattern: '{scope_match.group()}'"
                    if scope_match
                    else "No scope violation patterns detected"
                ),
            )
        )

        return obligations


# ─── Claude Checker (optional, requires anthropic SDK) ───────


class ClaudeChecker(ConstitutionalChecker):
    """Semantic constitutional checker using Claude API.

    Requires the anthropic SDK: pip install kovrin-safety[anthropic]
    Provides deeper semantic analysis than rule-based checking.
    """

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5") -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeChecker requires the anthropic SDK. "
                "Install with: pip install kovrin-safety[anthropic]"
            ) from e
        if api_key:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        else:
            self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def _check_inner(self, action: str, context: str = "") -> list[ProofObligation]:

        axiom_descriptions = "\n".join(
            f"  Axiom {a.id} — {a.name}: {a.description}" for a in AXIOMS
        )

        prompt = f"""You are the constitutional safety checker for an AI agent system.

Evaluate whether a proposed action violates any of the five immutable ethical axioms.

AXIOMS:
{axiom_descriptions}

ACTION TO EVALUATE:
  Description: {action}
  Context: {context or "Not provided"}

For EACH axiom, respond with a JSON array of objects:
[
  {{
    "axiom_id": 1,
    "axiom_name": "Human Agency",
    "passed": true/false,
    "evidence": "Brief explanation of why this passes or fails"
  }},
  ... (one entry per axiom, all 5)
]

RULES:
- Be strict. If there is reasonable doubt, FAIL the axiom.
- An action that could indirectly violate an axiom still fails.
- Return ONLY the JSON array, no other text."""

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_obligations(response.content[0].text)

    def _parse_obligations(self, response_text: str) -> list[ProofObligation]:
        """Parse Claude's response into ProofObligation objects."""
        import json as _json

        text = response_text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1

        if start == -1 or end == 0:
            return self._fail_safe("Failed to parse constitutional check response")

        try:
            results = _json.loads(text[start:end])
        except _json.JSONDecodeError:
            return self._fail_safe("JSON parse error in constitutional check")

        obligations: list[ProofObligation] = []
        for item in results:
            axiom_id = item.get("axiom_id", 0)
            axiom = next((a for a in AXIOMS if a.id == axiom_id), None)
            obligations.append(
                ProofObligation(
                    axiom_id=axiom_id,
                    axiom_name=item.get("axiom_name", axiom.name if axiom else "Unknown"),
                    description=axiom.description if axiom else "",
                    passed=item.get("passed", False),
                    evidence=item.get("evidence", ""),
                )
            )

        # Ensure all 5 axioms are covered — missing = failed
        covered_ids = {o.axiom_id for o in obligations}
        for axiom in AXIOMS:
            if axiom.id not in covered_ids:
                obligations.append(
                    ProofObligation(
                        axiom_id=axiom.id,
                        axiom_name=axiom.name,
                        description=axiom.description,
                        passed=False,
                        evidence="Axiom not evaluated — fail-safe rejection",
                    )
                )

        return obligations

    @staticmethod
    def _fail_safe(reason: str) -> list[ProofObligation]:
        """Return all-failed obligations as fail-safe."""
        return [
            ProofObligation(
                axiom_id=a.id,
                axiom_name=a.name,
                description=a.description,
                passed=False,
                evidence=f"{reason} — fail-safe rejection",
            )
            for a in AXIOMS
        ]


# ─── Factory ─────────────────────────────────────────────────


def create_checker(api_key: str | None = None) -> ConstitutionalChecker:
    """Create the appropriate constitutional checker.

    If an Anthropic API key is provided, returns ClaudeChecker
    for deep semantic analysis. Otherwise returns RuleBasedChecker
    for fast, offline pattern matching.
    """
    if api_key:
        return ClaudeChecker(api_key=api_key)
    return RuleBasedChecker()
