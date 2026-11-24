# MCP Server LLM CLI Runner

MCP server providing unified access to multiple LLM providers (agy, llama, claude; see providers/) through a consistent interface.

## Project Structure

```
src/mcp_server_llm_cli_runner/
├── core/           # Core models and base classes
├── providers/      # LLM provider implementations (agy, llama, claude)
├── lean_mcp_interface.py  # Lean FastMCP interface (entrypoint via __main__.py)
├── cache/          # Response caching backends
└── utils/          # Configuration, logging, helpers
tests/              # Test suite
```

## Development Commands

```bash
# Environment management
pixi run -e quality lint          # Run linting (ruff)
pixi run -e quality test          # Run full test suite
pixi run -e quality test-unit     # Run unit tests only
pixi run -e quality quality       # Full quality check (lint + types + tests)
pixi run -e quality pyright src/  # Type checking

# Server execution
pixi run serve                    # Start MCP server
```

## Llama Setup

`llama-cpp-python` comes from conda-forge (pixi default feature; no PyPI build). Set `LLAMA_MODEL_PATH` to a GGUF file and verify with `pixi run check-llama`.

## Quality Standards

- **Linting**: Ruff with critical rules (F, E9) as blocking
- **Type checking**: Pyright in strict mode
- **Testing**: pytest with coverage requirements
- **Pre-commit**: Hooks configured in `.pre-commit-config.yaml`

## Architecture Notes

- Providers implement a common interface for LLM completions
- `claude` (alias `anthropic`, `ProviderType.ANTHROPIC`) wraps the Claude Code CLI in isolated print mode using the user's subscription login (no API key). It is explicit-only: never auto-routed or used as a fallback
- Configuration via `utils/config.py` with environment variable support
- Structured logging via `utils/logging.py` with correlation IDs
- MCP tools exposed via the lean FastMCP interface in `lean_mcp_interface.py` (`create_lean_interface`)

## Session Intelligence

This project uses session-intelligence MCP server for:
- Session lifecycle tracking
- Decision logging
- Learning capture

Avoid storing session-specific or perishable information in this file.
