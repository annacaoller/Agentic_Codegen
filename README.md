# Agentic CodeGen

Agentic CodeGen is a deterministic, phase-driven agent that generates Python code, documentation, and tests from either a structured specification or a short natural language prompt. The project is designed as a portfolio-grade example of Agentic AI engineering, emphasizing reliability, explicit control flow, and testability rather than raw prompt output.

The core idea behind this project is that large language models should not be treated as autonomous systems. Instead, they act as decision-makers inside a strictly controlled environment. All state is external, all actions are explicit, and all side effects are validated. The agent never relies on implicit memory or hidden reasoning across turns.

The agent operates through an explicit loop where the prompt is reconstructed on every iteration from a disciplined memory snapshot. This snapshot contains the current phase, the specification, the current code artifact, the current test artifact, and the results of the last validation step. The language model receives this reconstructed state and is only responsible for selecting the next action to perform, always via a structured JSON response.

The generation process is intentionally split into multiple phases. Code implementation, documentation, test generation, validation, and export are treated as separate steps. This design reduces hallucination, improves consistency, and makes failures observable and correctable. Each phase has clear entry conditions, exit conditions, and retry limits.

The system is built around a tool-based architecture. The language model never writes files, executes code, or runs tests directly. Instead, it selects a tool by name, and the environment executes that tool deterministically. Tools include code generation, documentation enrichment, test generation, validation, and file export. This separation ensures that reasoning and execution are clearly decoupled.

Validation is treated as a first-class concern. Generated code is compiled using the Python interpreter, tests are executed with the standard unittest framework, and heuristic quality checks are applied. These checks include minimum test counts, edge case coverage, and restrictions on external dependencies. When validation fails, the agent re-enters the loop with explicit feedback rather than silently continuing.

The entire agent loop is fully testable without network access. A deterministic FakeLLM is used in end-to-end tests to simulate the model’s decision-making. This allows the full agent pipeline to be validated in continuous integration environments without calling external APIs, ensuring reproducibility and reliability.

The project structure follows standard Python packaging conventions, with source code under a src directory, a clean separation of concerns across modules, and a dedicated tests directory. Unit tests cover parsing, validation, and guardrails, while an end-to-end test verifies the complete agent lifecycle from specification to generated artifacts.

To run the project locally, create and activate a virtual environment, then install the package in editable mode. The agent can be executed either by providing a full JSON specification or by using a short prompt with an explicit function name. Generated files are written to a configurable output directory.

All tests can be executed using the Python standard library test runner with explicit test discovery. No third-party testing frameworks are required.

This project demonstrates how to build production-oriented agentic systems by combining explicit state management, structured tool invocation, validation-driven retries, and deterministic testing. It is intended as a reference implementation for reliable Agentic AI design rather than a minimal example.
