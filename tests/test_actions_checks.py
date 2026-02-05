import unittest

from agentic_codegen.actions import Actions


class DummyLLM:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("DummyLLM should not be called in this test")


class TestActionsChecks(unittest.TestCase):
    def test_run_checks_detects_missing_function(self):
        actions = Actions(llm=DummyLLM())
        spec = {"id": "x", "stdlib_only": True, "function": {"name": "foo", "args": [], "returns": "int"}}
        code = "def bar():\n    return 1\n"
        tests = "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"
        result = actions.run_checks(spec=spec, code=code, tests=tests, strict=True, user_args={})
        self.assertFalse(result["ok"])
        self.assertTrue(any("Function definition not found" in c for c in result["checks"]))

    def test_run_checks_passes_simple_valid_case(self):
        actions = Actions(llm=DummyLLM())
        spec = {"id": "x", "stdlib_only": True, "function": {"name": "foo", "args": [], "returns": "int"},
                "quality": {"min_tests": 1}}
        code = "def foo():\n    return 1\n"
        tests = (
            "import unittest\n"
            "from foo import foo\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_foo(self):\n"
            "        self.assertEqual(foo(), 1)\n"
        )
        result = actions.run_checks(spec=spec, code=code, tests=tests, strict=True, user_args={})
        self.assertTrue(result["ok"], msg=str(result["checks"]))


if __name__ == '__main__':
    unittest.main()
