# Gemini AI Context for MCP Server Cheap LLM

This file provides context for Gemini AI analysis of the `mcp-server-cheap-llm` project.

## Project Overview

**MCP Server for Cheap LLM Alternatives** - A Model Context Protocol server providing access to cost-effective LLM providers including Gemini CLI, OpenAI, and LLaMA models.

### Technology Stack
- **Python 3.12+** with async/await patterns
- **aiohttp** for async HTTP server
- **pydantic** for data validation and settings
- **structlog** for structured logging
- **pixi** for dependency management (conda-forge ecosystem)
- **MCP Protocol** for standardized AI tool integration

### Architecture Patterns

#### Core Components
- **`src/core/`** - Core models, request processing, error handling
- **`src/cache/`** - Multi-tier caching system with Redis/Memory/File backends
- **`src/providers/`** - LLM provider implementations with circuit breakers
- **`src/server/`** - MCP server handlers and middleware
- **`src/services/`** - Batch processing and similarity matching
- **`src/utils/`** - Configuration, logging, and utility functions

#### Key Design Patterns
1. **Circuit Breaker Pattern** - Provider fault tolerance (`providers/circuit_breaker.py`)
2. **Strategy Pattern** - Multiple cache backends (`cache/backends.py`)
3. **Factory Pattern** - Provider registry and creation (`providers/registry.py`)
4. **Async Patterns** - Non-blocking I/O throughout
5. **Pydantic Models** - Type-safe configuration and data validation

### Quality Standards

#### Zero-Tolerance Policy
- **F,E9 Violations**: Syntax and import errors cause immediate CI failure
- **Type Checking**: pyright with strict configuration
- **Linting**: ruff with comprehensive rule set
- **Testing**: pytest with async support and comprehensive coverage

#### Performance Requirements
- **Async-First**: All I/O operations must be non-blocking
- **Caching**: Implement multi-tier caching for expensive operations
- **Circuit Breakers**: Fail-fast for unreliable external services
- **Batch Processing**: Optimize similar requests through batching

### Security Considerations

#### Critical Security Areas
1. **API Key Management** - Secure handling of provider API keys
2. **Input Validation** - Strict validation of all user inputs
3. **Rate Limiting** - Prevent abuse and manage costs
4. **Error Handling** - Avoid leaking sensitive information in errors

#### Security Tools
- **bandit** for AST-based security analysis
- **safety** for dependency vulnerability scanning
- **detect-secrets** for preventing secret commits

### Testing Strategy

#### Test Structure
- **Unit Tests** (`tests/unit/`) - Individual component testing
- **Integration Tests** (`tests/integration/`) - Component interaction testing
- **End-to-End Tests** (`tests/e2e/`) - Full workflow testing

#### Testing Markers
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration test marker
- `@pytest.mark.requires_gemini` - Tests requiring Gemini API access
- `@pytest.mark.property` - Property-based testing

### Performance Benchmarks

#### Benchmark Categories
- **MCP Operations** - Protocol handler performance
- **Provider Operations** - LLM provider response times
- **Cache Operations** - Cache hit/miss performance
- **Configuration** - Config loading and validation speed

#### Performance Thresholds
- **Response Time**: < 100ms for cached requests
- **Cold Start**: < 500ms for uncached requests
- **Memory Usage**: < 256MB baseline memory footprint
- **Cache Hit Rate**: > 80% for repeated requests

### Common Issues & Solutions

#### Dependency Management
- **pixi Platform Issues**: Project uses linux-64 only for dev speed
- **Conda-forge Priority**: All dependencies from conda-forge when possible
- **Python Version**: Strict 3.12+ requirement due to modern syntax usage

#### Async Patterns
- **Event Loop**: Ensure proper async context management
- **Timeouts**: All external calls must have reasonable timeouts
- **Resource Cleanup**: Proper cleanup of async resources and connections

#### MCP Protocol
- **Tool Registration**: Correct tool metadata and parameter validation
- **Error Responses**: Proper MCP error format compliance
- **Streaming**: Support for streaming responses where applicable

### Code Review Focus Areas

When reviewing code changes, pay special attention to:

1. **Async Correctness** - Proper use of async/await, no blocking calls
2. **Type Safety** - Complete type hints, pydantic model usage
3. **Error Handling** - Comprehensive error catching and proper error responses
4. **Resource Management** - Context managers, proper cleanup
5. **Performance Impact** - Cache usage, unnecessary computations
6. **Security Implications** - Input validation, secret handling
7. **Testing Coverage** - Adequate test coverage for new functionality
8. **Documentation** - Docstrings following Google style

### Development Workflow

#### Quality Checks (Required)
```bash
pixi run -e quality quality     # Full quality pipeline
pixi run -e quality lint         # Critical F,E9 check
pixi run -e quality typecheck    # Type validation
pixi run -e quality test-fast    # Fast test suite
```

#### Security Validation
```bash
pixi run -e quality-extended security-scan  # Security analysis
pixi run -e quality-extended static-analysis # Full static analysis
```

#### Performance Testing
```bash
pixi run -e quality-extended test-benchmark  # Performance benchmarks
```

### CI/CD Pipeline

The project uses the **CI Framework** with intelligent optimization:

1. **Change Detection** - Smart CI job skipping based on change patterns
2. **Quality Gates** - Zero-tolerance quality enforcement
3. **Performance Monitoring** - Regression detection with benchmarks
4. **Security Scanning** - Multi-layer vulnerability analysis
5. **Cross-Platform Validation** - Native pixi dependency testing
6. **AI Analysis** - Gemini-powered code review and insights

### Gemini AI Guidelines

When analyzing this project:

1. **Focus on Python Best Practices** - Modern Python patterns, type hints
2. **Understand Async Context** - Recognize async/await requirements
3. **Consider MCP Protocol** - Ensure compliance with MCP specifications
4. **Evaluate Performance** - Look for optimization opportunities
5. **Security First** - Identify potential security vulnerabilities
6. **Testing Quality** - Assess test coverage and effectiveness

This context should help you provide more accurate and relevant analysis of the codebase.
