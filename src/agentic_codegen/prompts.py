from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Phase(str, Enum):
    IMPLEMENT = "implement"
    DOCS = "docs"
    TESTS = "tests"
    EVALUATE = "evaluate"
    EXPORT = "export"
    FIX = "fix"


AGENT_RULES = """\
You are an agentic code generation assistant.

NON-NEGOTIABLE OUTPUT RULES:
1) You MUST output exactly one fenced block of type ```action``` containing valid JSON.
2) The JSON object MUST contain:
   - "tool_name": string
   - "args": object (can be empty)
3) Do NOT include additional code blocks besides the single ```action``` block.

QUALITY & SAFETY:
- Prefer Python standard library only if spec.stdlib_only is true.
- Produce deterministic, testable code.
- If prior checks failed, prioritize fixing those failures.
"""


@dataclass(frozen=True)
class MemorySnapshot:
    phase: Phase
    spec: Dict[str, Any]
    code: str = ""
    tests: str = ""
    last_checks: Optional[List[str]] = None
    iteration: int = 0
    phase_retries: int = 0


def _compact_spec(spec: Dict[str, Any], max_chars: int = 4000) -> str:
    """
    Render spec into a compact string. Keep it simple now; we can upgrade later.
    """
    s = str(spec)
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def build_prompt(snapshot: MemorySnapshot) -> str:
    """
    Build the full prompt = rules + disciplined memory snapshot.
    """
    lines: List[str] = []
    lines.append(AGENT_RULES)
    lines.append("\n---\n")
    lines.append(f"PHASE: {snapshot.phase.value}")
    lines.append(f"ITERATION: {snapshot.iteration}")
    lines.append(f"PHASE_RETRIES: {snapshot.phase_retries}")
    lines.append("\nSPEC:\n" + _compact_spec(snapshot.spec))

    if snapshot.code.strip():
        lines.append("\nCURRENT_CODE:\n" + snapshot.code)

    if snapshot.tests.strip():
        lines.append("\nCURRENT_TESTS:\n" + snapshot.tests)

    if snapshot.last_checks:
        # keep short, high signal
        joined = "\n".join(f"- {c}" for c in snapshot.last_checks[:10])
        lines.append("\nLAST_CHECKS:\n" + joined)

    lines.append("\n---\n")
    lines.append(_phase_instruction(snapshot))
    return "\n".join(lines)


def _phase_instruction(snapshot: MemorySnapshot) -> str:
    """
    Phase-specific instructions. Each phase produces a tool action.
    """
    if snapshot.phase == Phase.IMPLEMENT:
            return """\
Task: Choose the next tool to generate code.

Return an action:
- tool_name: "write_code"
- args: {}
"""
    if snapshot.phase == Phase.DOCS:
        return """\
Task: Choose the next tool to add docstrings.

Return an action:
- tool_name: "write_docs"
- args: {}
"""
    if snapshot.phase == Phase.TESTS:
        return """\
Task: Choose the next tool to generate unittest tests.

Return an action:
- tool_name: "write_tests"
- args: {}
"""
    if snapshot.phase == Phase.EVALUATE:
        return """\
Task: Choose the next tool to run checks.

Return an action:
- tool_name: "run_checks"
- args: {}
"""
    if snapshot.phase == Phase.EXPORT:
        return """\
Task: Choose the next tool to save files.

Return an action:
- tool_name: "save_file"
- args: {}
"""
    # FIX (or default)
    return """\
Task: Choose the next tool to fix failures described in LAST_CHECKS.

Return an action:
- tool_name: "write_code"
- args: {}
"""
