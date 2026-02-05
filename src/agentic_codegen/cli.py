from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_codegen.actions import Actions
from agentic_codegen.agent import Agent, AgentConfig
from agentic_codegen.llm_openai import OpenAILLM


def _load_spec(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _spec_from_prompt(prompt: str, name: str, stdlib_only: bool = True) -> Dict[str, Any]:
    # Minimal spec; enough to drive the pipeline.
    return {
        "id": f"{name}_v1",
        "language": "python",
        "stdlib_only": stdlib_only,
        "function": {
            "name": name,
            "args": [],
            "returns": "Any",
        },
        "behavior": {
            "description": prompt,
            "constraints": [],
        },
        "examples": [],
        "edge_cases": [],
        "quality": {
            "docstring_style": "google",
            "min_tests": 4,
            "must_include_edge_case_test": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic_codegen", description="Agentic code generator (portfolio project).")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the agent to generate code+docs+tests from a spec or prompt.")
    run.add_argument("--spec", type=str, default=None, help="Path to spec.json")
    run.add_argument("--prompt", type=str, default=None, help="Quick prompt (creates a minimal spec)")
    run.add_argument("--name", type=str, default="generated_function", help="Function name for --prompt mode")
    run.add_argument("--no-stdlib-only", action="store_true", help="Allow non-stdlib imports in --prompt mode")

    run.add_argument("--out", type=str, default="generated", help="Output directory for generated files")
    run.add_argument("--max-iters", type=int, default=8, help="Max agent iterations")
    run.add_argument("--max-phase-retries", type=int, default=2, help="Max retries per phase")
    run.add_argument("--strict", action="store_true", help="Enable strict checks")
    run.add_argument("--no-strict", action="store_true", help="Disable strict checks")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        if bool(args.spec) == bool(args.prompt):
            print("Error: provide exactly one of --spec or --prompt.", file=sys.stderr)
            return 2

        if args.spec:
            spec = _load_spec(args.spec)
        else:
            stdlib_only = not args.no_stdlib_only
            spec = _spec_from_prompt(args.prompt, args.name, stdlib_only=stdlib_only)

        strict = True
        if args.no_strict:
            strict = False
        elif args.strict:
            strict = True

        llm = OpenAILLM()
        actions = Actions(llm=llm)
        agent = Agent(
            llm=llm,
            actions=actions,
            config=AgentConfig(
                max_iterations=args.max_iters,
                max_phase_retries=args.max_phase_retries,
                strict=strict,
                out_dir=args.out,
            ),
        )

        result = agent.run(spec)
        if result.ok:
            print("OK:", result.summary)
            if result.file_paths:
                print("Files:")
                for k, v in result.file_paths.items():
                    print(f"  {k}: {v}")
            return 0

        print("FAILED:", result.summary, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
