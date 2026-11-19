# MCP Server LLM CLI Runner

MCP server providing unified access to multiple LLM providers (OpenAI, Gemini, Llama, etc.) through a consistent interface.

## Project Structure

```
src/mcp_server_llm_cli_runner/
├── core/           # Core models and base classes
├── providers/      # LLM provider implementations (openai, gemini, llama)
├── server/         # MCP server handlers and protocol
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
pixi run server                   # Start MCP server
```

## Quality Standards

- **Linting**: Ruff with critical rules (F, E9) as blocking
- **Type checking**: Pyright in strict mode
- **Testing**: pytest with coverage requirements
- **Pre-commit**: Hooks configured in `.pre-commit-config.yaml`

## Architecture Notes

- Providers implement a common interface for LLM completions
- Configuration via `utils/config.py` with environment variable support
- Structured logging via `utils/logging.py` with correlation IDs
- MCP protocol handlers in `server/handlers.py`

## Session Intelligence

This project uses session-intelligence MCP server for:
- Session lifecycle tracking
- Decision logging
- Learning capture

Avoid storing session-specific or perishable information in this file.
