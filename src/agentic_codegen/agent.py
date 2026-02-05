from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from agentic_codegen.parsing import Action, safe_parse_action
from agentic_codegen.prompts import MemorySnapshot, Phase, build_prompt


class LLM(Protocol):
    """
    Minimal interface to allow swapping OpenAI / mocks in tests.
    """
    def generate(self, prompt: str) -> str:  # pragma: no cover
        ...


@dataclass
class AgentConfig:
    max_iterations: int = 8
    max_phase_retries: int = 2
    strict: bool = True
    out_dir: str = "generated"


@dataclass
class AgentResult:
    ok: bool
    summary: str
    file_paths: Optional[Dict[str, str]] = None
    iterations: int = 0


class Agent:
    """
    Orchestrates a phase-driven agent loop.

    The agent constructs prompt = rules + memory snapshot each iteration (prompt is state),
    then expects a structured action to decide next tool invocation.
    """
    def __init__(self, llm: LLM, actions: Any, config: Optional[AgentConfig] = None) -> None:
        self.llm = llm
        self.actions = actions
        self.config = config or AgentConfig()

    def run(self, spec: Dict[str, Any]) -> AgentResult:
        phase = Phase.IMPLEMENT
        iteration = 0
        phase_retries = 0

        code = ""
        tests = ""
        last_checks: list[str] = []

        while iteration < self.config.max_iterations:
            snapshot = MemorySnapshot(
                phase=phase,
                spec=spec,
                code=code,
                tests=tests,
                last_checks=last_checks if last_checks else None,
                iteration=iteration,
                phase_retries=phase_retries,
            )
            prompt = build_prompt(snapshot)

            model_text = self.llm.generate(prompt)
            action, err = safe_parse_action(model_text)
            if action is None:
                # Parsing failure is treated like a phase retry; ask again same phase.
                last_checks = [f"ActionParseError: {err}"]
                phase_retries += 1
                if phase_retries > self.config.max_phase_retries:
                    return AgentResult(
                        ok=False,
                        summary=f"Failed to parse action after retries. Last error: {err}",
                        iterations=iteration + 1,
                    )
                iteration += 1
                continue

            # Execute tool
            tool_name = action.tool_name
            try:
                result = self._dispatch(tool_name, action.args, spec, code, tests, last_checks)
            except Exception as e:
                last_checks = [f"ToolExecutionError in {tool_name}: {type(e).__name__}: {e}"]
                phase_retries += 1
                if phase_retries > self.config.max_phase_retries:
                    return AgentResult(
                        ok=False,
                        summary="Tool execution failed after retries.\n" + "\n".join(last_checks),
                        iterations=iteration + 1,
                    )
                iteration += 1
                continue

            # Apply results and advance phase machine
            if tool_name == "write_code":
                code = result["code"]
                last_checks = []
                phase = Phase.DOCS
                phase_retries = 0

            elif tool_name == "write_docs":
                code = result["code"]
                last_checks = []
                phase = Phase.TESTS
                phase_retries = 0

            elif tool_name == "write_tests":
                tests = result["tests"]
                last_checks = []
                phase = Phase.EVALUATE
                phase_retries = 0

            elif tool_name == "run_checks":
                ok = bool(result["ok"])
                last_checks = list(result.get("checks", []))
                if ok:
                    phase = Phase.EXPORT
                    phase_retries = 0
                else:
                    # Go to FIX; we keep code/tests, and provide last_checks
                    phase = Phase.FIX
                    phase_retries = 0

            elif tool_name == "save_file":
                paths = result["file_paths"]
                return AgentResult(
                    ok=True,
                    summary="Generated code and tests successfully.",
                    file_paths=paths,
                    iterations=iteration + 1,
                )

            elif tool_name == "terminate":
                return AgentResult(
                    ok=bool(result.get("ok", False)),
                    summary=str(result.get("summary", "")),
                    iterations=iteration + 1,
                )

            else:
                # Unknown tool_name => fail-fast (counts as retry)
                last_checks = [f"Unknown tool_name: {tool_name}"]
                phase_retries += 1
                if phase_retries > self.config.max_phase_retries:
                    return AgentResult(
                        ok=False,
                        summary="Unknown tool repeated. " + "\n".join(last_checks),
                        iterations=iteration + 1,
                    )

            iteration += 1

        return AgentResult(
            ok=False,
            summary=f"Exceeded max_iterations={self.config.max_iterations}.",
            iterations=iteration,
        )

    def _dispatch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        spec: Dict[str, Any],
        code: str,
        tests: str,
        last_checks: list[str],
    ) -> Dict[str, Any]:
        """
        Calls the action handlers. Action args can be empty; we enrich with current state.
        """
        if tool_name == "write_code":
            return self.actions.write_code(
                spec=spec,
                previous_code=code,
                failed_checks=last_checks,
                user_args=args,
            )
        if tool_name == "write_docs":
            return self.actions.write_docs(
                spec=spec,
                code=code,
                user_args=args,
            )
        if tool_name == "write_tests":
            return self.actions.write_tests(
                spec=spec,
                code=code,
                user_args=args,
            )
        if tool_name == "run_checks":
            return self.actions.run_checks(
                spec=spec,
                code=code,
                tests=tests,
                strict=self.config.strict,
                user_args=args,
            )
        if tool_name == "save_file":
            return self.actions.save_file(
                spec=spec,
                code=code,
                tests=tests,
                out_dir=self.config.out_dir,
                user_args=args,
            )
        if tool_name == "terminate":
            # passthrough; let model terminate with summary if needed
            return {"ok": True, "summary": args.get("summary", "Terminated.")}
        raise ValueError(f"Unknown tool_name: {tool_name}")
