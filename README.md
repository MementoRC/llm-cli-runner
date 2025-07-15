# MCP Server Cheap LLM

A Model Context Protocol server providing access to cost-effective Large Language Model providers including Gemini CLI, Codex, and LLaMA.

## 🚀 Features

- **Multiple Provider Support**: Gemini CLI, OpenAI Codex, and local LLaMA models
- **Cost-Effective**: Optimized for affordable LLM access
- **MCP Protocol**: Full Model Context Protocol compliance
- **Enterprise Security**: Comprehensive security scanning and validation
- **Zero-Tolerance Quality**: 100% test coverage with strict quality gates

## 📋 Requirements

- Python ≥ 3.12
- [Pixi](https://pixi.sh/) for environment management
- Optional: API keys for Gemini/Codex providers

## 🛠️ Installation

### Using Pixi (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd cheap-llm/development

# Install with pixi
pixi install
pixi run install-editable

# Set up pre-commit hooks
pixi run install-pre-commit
```

### Provider Setup

Set up environment variables for your chosen providers:

```bash
# For Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# For OpenAI Codex
export OPENAI_API_KEY="your-openai-api-key"

# LLaMA runs locally (no API key needed)
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
pixi run -e dev safety check

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
default_provider = "gemini"
max_concurrent_requests = 10

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
model_name = "gemini-pro"
api_key = "${GEMINI_API_KEY}"

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
model_name = "llama-2-7b-chat"
```

## 🏗️ Architecture

The project follows atomic design principles:

```
src/mcp_server_cheap_llm/
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

- **Issues**: [GitHub Issues](https://github.com/anthropics/mcp-server-cheap-llm/issues)
- **Documentation**: See `docs/` directory
- **Security**: Report security issues privately

---

**State-of-the-Art**: This project implements enterprise-grade development practices with comprehensive quality validation, security scanning, and zero-tolerance quality gates.
