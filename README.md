# MCP Server LLM CLI Runner

A Model Context Protocol server providing access to multi-provider Large Language Model providers including Antigravity CLI (`agy`) and LLaMA. Codex is not a provider here; use the `codex` MCP server (`codex mcp-server`) directly.

## 🚀 Features

- **Claude provider (explicit-only)**: `claude` (alias `anthropic`) wraps the Claude Code CLI (`claude -p`) using your existing Claude subscription login (OAuth, no API key). It runs isolated (no hooks, CLAUDE.md, MCP servers or tools; empty temp cwd) and is billed to your Claude plan, so it is never auto-selected or used as a fallback; only `provider="claude"` reaches it. Set `CLAUDE_CLI_PATH` if `claude` is not on the service `PATH`; default model `sonnet`.
- **Multiple Provider Support**: Antigravity CLI (`agy`, default) and local LLaMA models. The `gemini` provider is a deprecated alias for `agy` (see [Gemini deprecation](#gemini-deprecation))
- **Cost-Effective**: Optimized for efficient LLM access
- **MCP Protocol**: Full Model Context Protocol compliance
- **Enterprise Security**: Comprehensive security scanning and validation
- **Zero-Tolerance Quality**: 100% test coverage with strict quality gates

## 📋 Requirements

- Python ≥ 3.12
- [Pixi](https://pixi.sh/) for environment management
- Optional: `agy` CLI (Antigravity) on `PATH` or `AGY_CLI_PATH`

## 🛠️ Installation

### Using Pixi (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd llm-cli-runner/development

# Install with pixi
pixi install
pixi run install-editable

# Set up pre-commit hooks
pixi run install-pre-commit
```

### Provider Setup

Set up environment variables for your chosen providers:

```bash
# For agy (Antigravity CLI): install the `agy` binary, or point to it
export AGY_CLI_PATH="/path/to/agy"   # optional if `agy` is on PATH

# LLaMA runs locally (no API key needed); llama-cpp-python comes from conda-forge via pixi
export LLAMA_MODEL_PATH="/path/to/model.gguf"
# verify the native library loads: pixi run check-llama
```

## 🧪 Testing

The project follows strict Test-Driven Development with zero-tolerance quality gates:

```bash
# Run all tests
pixi run test

# Run with coverage
pixi run test-cov

# Run specific test categories
pixi run test-unit        # Unit tests
pixi run test-integration # Integration tests
pixi run test-e2e        # End-to-end tests
pixi run test-property   # Property-based tests
```

## 🔒 Quality Assurance

### Zero-Tolerance Quality Gates

All development must pass these mandatory checks:

```bash
# Format checking
pixi run format-check

# Critical linting (F,E9 violations block commits)
pixi run lint

# Type checking
pixi run typecheck

# Security scanning
pixi run -e dev bandit -r src/ --severity-level high
pixi run -e audit audit

# Complete quality validation
pixi run quality
```

### Pre-commit Hooks

Quality gates are enforced automatically:

```bash
# Install hooks
pixi run install-pre-commit

# Run manually
pixi run pre-commit
```

## 🚀 Usage

### Starting the MCP Server

```bash
# Start server (stdio mode for MCP)
pixi run serve

# Debug mode with verbose logging
pixi run serve-debug
```

### Configuration

Create a configuration file or use environment variables:

```toml
# config.toml
# The default provider is llama (configurable via default_provider or
# MCP_DEFAULT_PROVIDER); agy is the recommended CLI provider and must be enabled
# below to be used as the default.
default_provider = "agy"
max_concurrent_requests = 10

[[providers]]
name = "agy"
provider_type = "agy"
enabled = true
model_name = "gemini-3.8-flash-medium"

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
model_name = "llama-2-7b-chat"
```

Available `agy` models: `gemini-3.8-flash-high`, `gemini-3.8-flash-medium`, `gemini-3.8-flash-low`, `claude-sonnet-4-6`, `claude-opus-4-6-thinking`, `gpt-oss-120b-medium`. `agy` returns a single chunk (no true streaming) and ignores `system_prompt`.

### Gemini deprecation

The `gemini` provider is deprecated. The GeminiProvider implementation (which drove the Gemini CLI) has been removed; `gemini` remains an alias for `agy`. `llm_complete(provider="gemini", ...)` is a permanent alias served by the `agy` provider, and responses report `provider: "agy"` with `deprecated_alias: "gemini->agy"`.

Legacy Gemini model names on redirected calls are mapped by tier:

| Legacy model name | Mapped to |
|-------------------|-----------|
| `gemini-*flash-lite*` | `gemini-3.8-flash-low` |
| other `gemini-*flash*` | `gemini-3.8-flash-medium` |
| `gemini-*pro*` | `gemini-3.8-flash-high` |
| unknown `gemini-*` | dropped (agy default) |

If `agy` is not installed, `gemini` calls fail. There is no fallback to the old Gemini CLI.

## 🏗️ Architecture

The project follows atomic design principles:

```
src/mcp_server_llm_cli_runner/
├── core/           # Atoms & Molecules (data models)
├── services/       # Organisms (provider implementations)
├── server/         # Templates (MCP server logic)
├── utils/          # Atoms (utilities, config, logging)
└── __main__.py     # Pages (entry point)
```

## 🔧 Development

### Project Structure

- **100% Test Coverage**: Every component has comprehensive tests
- **Atomic Design**: Clear hierarchy from simple to complex components
- **Type Safety**: Complete type hints throughout
- **Security First**: Enterprise-grade security scanning
- **Documentation**: Comprehensive docstrings with examples

### Quality Standards

- **ZERO FAILURES**: Any test failure stops development
- **ZERO F,E9 VIOLATIONS**: Critical lint issues block commits
- **100% COVERAGE**: All new code must have complete test coverage
- **TDD REQUIRED**: Tests written before implementation

## 🤝 Contributing

1. **Fork and clone** the repository
2. **Create a feature branch** following our naming conventions
3. **Write tests first** (TDD methodology)
4. **Implement functionality** to make tests pass
5. **Run quality checks** and ensure all pass
6. **Submit a pull request** with comprehensive description

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/llm-cli-runner/issues)
- **Documentation**: See `docs/` directory
- **Security**: Report security issues privately

---

**State-of-the-Art**: This project implements enterprise-grade development practices with comprehensive quality validation, security scanning, and zero-tolerance quality gates.
