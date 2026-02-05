import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agentic_codegen.actions import Actions
from agentic_codegen.agent import Agent, AgentConfig


class FakeLLM:
    """
    Always returns a single action block choosing the correct tool for each phase.

    This makes the agent loop deterministic and tests the integration between:
    prompts -> parsing -> dispatch -> actions -> checks -> save_file.
    """
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        p = prompt.lower()

        # phase is included in prompt by MemorySnapshot
        if "phase: implement" in p:
            return """```action
{"tool_name":"write_code","args":{}}
```"""
        if "phase: docs" in p:
            return """```action
{"tool_name":"write_docs","args":{}}
```"""
        if "phase: tests" in p:
            return """```action
{"tool_name":"write_tests","args":{}}
```"""
        if "phase: evaluate" in p:
            return """```action
{"tool_name":"run_checks","args":{}}
```"""
        if "phase: export" in p:
            return """```action
{"tool_name":"save_file","args":{}}
```"""

        # fallback
        return """```action
{"tool_name":"terminate","args":{"summary":"Unknown phase"}}
```"""


class DeterministicActions(Actions):
    """
    Override the LLM-backed tools so we don't need OpenAI in tests.
    """

    def write_code(self, spec, previous_code, failed_checks, user_args):
        fn = spec["function"]["name"]
        # simple, correct function based on spec behavior
        # We'll implement a tiny example: swap_dict with ValueError on duplicates.
        if fn == "swap_dict":
            code = (
                "def swap_dict(d: dict[str, int]) -> dict[int, str]:\n"
                "    \"\"\"Swap keys and values.\n\n"
                "    Raises:\n"
                "        ValueError: If values are not unique.\n"
                "    \"\"\"\n"
                "    out: dict[int, str] = {}\n"
                "    for k, v in d.items():\n"
                "        if v in out:\n"
                "            raise ValueError('Values must be unique')\n"
                "        out[v] = k\n"
                "    return out\n"
            )
            return {"code": code}

        # generic fallback
        code = (
            f"def {fn}():\n"
            "    \"\"\"Auto-generated function.\"\"\"\n"
            "    return None\n"
        )
        return {"code": code}

    def write_docs(self, spec, code, user_args):
        # already has docstring in deterministic code; return as-is
        return {"code": code}

    def write_tests(self, spec, code, user_args):
        fn = spec["function"]["name"]
        module = fn  # sanitize matches for 'swap_dict' in our implementation
        if fn == "swap_dict":
            tests = (
                "import unittest\n"
                f"from {module} import {fn}\n\n"
                "class TestSwapDict(unittest.TestCase):\n"
                "    def test_basic(self):\n"
                "        self.assertEqual(swap_dict({'a': 1, 'b': 2}), {1: 'a', 2: 'b'})\n\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(swap_dict({}), {})\n\n"
                "    def test_duplicate_values_raises(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            swap_dict({'a': 1, 'b': 1})\n\n"
                "    def test_does_not_mutate(self):\n"
                "        d = {'x': 9}\n"
                "        _ = swap_dict(d)\n"
                "        self.assertEqual(d, {'x': 9})\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            return {"tests": tests}

        tests = (
            "import unittest\n"
            f"from {module} import {fn}\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_smoke(self):\n"
            f"        self.assertIsNone({fn}())\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return {"tests": tests}


class TestEndToEnd(unittest.TestCase):
    def test_agent_end_to_end_generates_files(self):
        fake_llm = FakeLLM()
        actions = DeterministicActions(llm=fake_llm)

        spec = {
            "id": "swap_dict_keys_values_v1",
            "language": "python",
            "stdlib_only": True,
            "function": {
                "name": "swap_dict",
                "args": [{"name": "d", "type": "dict[str, int]"}],
                "returns": "dict[int, str]",
            },
            "behavior": {
                "description": "Return a new dict swapping keys and values.",
                "constraints": [
                    "Raise ValueError if values are not unique.",
                    "Do not mutate input.",
                ],
            },
            "quality": {"min_tests": 4},
        }

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "generated"
            agent = Agent(
                llm=fake_llm,
                actions=actions,
                config=AgentConfig(out_dir=str(out_dir), max_iterations=8, max_phase_retries=2, strict=True),
            )
            result = agent.run(spec)
            self.assertTrue(result.ok, msg=result.summary)
            self.assertIsNotNone(result.file_paths)
            self.assertTrue(Path(result.file_paths["module"]).exists())
            self.assertTrue(Path(result.file_paths["tests"]).exists())

            # Sanity: run checks against generated artifacts
            module_code = Path(result.file_paths["module"]).read_text(encoding="utf-8")
            tests_code = Path(result.file_paths["tests"]).read_text(encoding="utf-8")
            check = actions.run_checks(spec=spec, code=module_code, tests=tests_code, strict=True, user_args={})
            self.assertTrue(check["ok"], msg=str(check["checks"]))


if __name__ == "__main__":
    unittest.main()
