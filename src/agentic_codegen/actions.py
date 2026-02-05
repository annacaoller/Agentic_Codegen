from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------
# LLM protocol
# ---------------------------------------------------------------------
class LLM(Protocol):
    def generate(self, prompt: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_STD_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)


def _spec_get_function_name(spec: Dict[str, Any]) -> str:
    fn = spec.get("function", {}).get("name")
    if isinstance(fn, str) and fn.strip():
        return fn.strip()
    return str(spec.get("id", "generated_function"))


def _sanitize_module_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return "generated_module"
    if name[0].isdigit():
        name = "m_" + name
    return name


def _render_spec_compact(spec: Dict[str, Any], limit: int = 3500) -> str:
    s = str(spec)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _detect_nonstdlib_imports(
    code: str,
    local_modules: Optional[set[str]] = None,
) -> List[str]:
    local_modules = local_modules or set()

    stdlib_allow = {
        "typing", "types", "dataclasses", "enum",
        "re", "math", "json", "statistics", "random",
        "collections", "itertools", "functools", "operator",
        "datetime", "time",
        "pathlib", "os", "sys", "subprocess", "tempfile",
        "logging", "traceback",
        "unittest",
        "hashlib", "hmac", "secrets",
        "urllib", "http",
        "csv", "sqlite3",
        "decimal", "fractions",
    }

    bad: List[str] = []
    for mod in _STD_IMPORT_RE.findall(code):
        top = mod.split(".", 1)[0]
        if top in local_modules:
            continue
        if top in stdlib_allow:
            continue
        bad.append(mod)

    return bad


# ---------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------
@dataclass
class Actions:
    llm: LLM

    # ---------------------- LLM-backed tools ----------------------
    def write_code(
        self,
        spec: Dict[str, Any],
        previous_code: str,
        failed_checks: List[str],
        user_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""\
You are generating Python module code.

SPEC:
{_render_spec_compact(spec)}

CURRENT_CODE (may be empty):
{previous_code}

FAILED_CHECKS (if any):
{failed_checks}

Requirements:
- Output ONLY the Python module content. No markdown. No explanations.
- Implement exactly one primary function as described in SPEC.function.
- Keep code deterministic and testable.
- If FAILED_CHECKS mention errors, fix them.

Now output the full updated module content:
"""
        code = _strip_code_fences(self.llm.generate(prompt))
        return {"code": code}

    def write_docs(
        self,
        spec: Dict[str, Any],
        code: str,
        user_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        style = (spec.get("quality", {}) or {}).get("docstring_style", "google")
        prompt = f"""\
You are adding a docstring to an existing Python function.

DOCSTRING_STYLE: {style}

SPEC:
{_render_spec_compact(spec)}

CURRENT_CODE:
{code}

Requirements:
- Output ONLY the full updated Python module content. No markdown. No explanations.
- Add or improve a docstring for the main function.
"""
        updated = _strip_code_fences(self.llm.generate(prompt))
        return {"code": updated}

    def write_tests(
        self,
        spec: Dict[str, Any],
        code: str,
        user_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        fn_name = _spec_get_function_name(spec)
        module_name = _sanitize_module_name(fn_name)

        min_tests = int((spec.get("quality", {}) or {}).get("min_tests", 3))

        prompt = f"""\
You are generating unittest tests for a Python function.

MODULE_NAME: {module_name}
FUNCTION_NAME: {fn_name}
MIN_TESTS: {min_tests}

SPEC:
{_render_spec_compact(spec)}

CURRENT_CODE:
{code}

Requirements:
- Output ONLY the test file content. No markdown. No explanations.
- Use Python unittest.
- Import as: from {module_name} import {fn_name}
"""
        tests = _strip_code_fences(self.llm.generate(prompt))
        return {"tests": tests}

    # ---------------------- Eval / Environment tools ----------------------
    def run_checks(
        self,
        spec: Dict[str, Any],
        code: str,
        tests: str,
        strict: bool,
        user_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        checks: List[str] = []

        fn_name = _spec_get_function_name(spec)

        if f"def {fn_name}(" not in code:
            checks.append(f"Function definition not found: def {fn_name}(...).")

        if bool(spec.get("stdlib_only", False)):
            local_mods = {_sanitize_module_name(fn_name)}
            bad = _detect_nonstdlib_imports(code + "\n" + tests, local_modules=local_mods)
            if bad:
                checks.append(f"Non-stdlib imports detected: {sorted(set(bad))}")

        quality = spec.get("quality", {}) or {}
        min_tests = int((quality.get("min_tests", 3)) or 3)
        test_methods = len(re.findall(r"def\s+test_", tests))
        if test_methods < min_tests:
            checks.append(f"Not enough tests: found {test_methods}, expected >= {min_tests}.")

        module_name = _sanitize_module_name(fn_name)

        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                mod_path = td_path / f"{module_name}.py"
                test_path = td_path / "test_generated.py"

                mod_path.write_text(code, encoding="utf-8")
                test_path.write_text(tests, encoding="utf-8")

                proc_compile = subprocess.run(
                    ["python", "-m", "py_compile", str(mod_path)],
                    capture_output=True,
                    text=True,
                )
                if proc_compile.returncode != 0:
                    err = (proc_compile.stderr or proc_compile.stdout or "").strip()
                    checks.append("py_compile failed: " + (err or "unknown compile error"))

                proc_test = subprocess.run(
                    ["python", "-m", "unittest", "-q"],
                    cwd=str(td_path),
                    capture_output=True,
                    text=True,
                )
                if proc_test.returncode != 0:
                    msg = ((proc_test.stdout or "") + "\n" + (proc_test.stderr or "")).strip()
                    checks.append("unittest failed: " + (msg or "unknown unittest error"))
        except Exception as e:
            checks.append(f"Exception during checks: {type(e).__name__}: {e}")

        ok = len(checks) == 0
        return {"ok": ok, "checks": checks}

    def save_file(
        self,
        spec: Dict[str, Any],
        code: str,
        tests: str,
        out_dir: str,
        user_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        fn_name = _spec_get_function_name(spec)
        module_name = _sanitize_module_name(fn_name)

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        module_file = out_path / f"{module_name}.py"
        tests_file = out_path / f"test_{module_name}.py"

        module_file.write_text(code, encoding="utf-8")
        tests_file.write_text(tests, encoding="utf-8")

        return {
            "file_paths": {
                "module": str(module_file),
                "tests": str(tests_file),
            }
        }