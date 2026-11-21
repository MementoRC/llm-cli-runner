# MCP Server LLM CLI Runner - API Guide

## Overview

This document describes the API for MCP Server LLM CLI Runner, a Model Context Protocol server providing access to multi-provider Large Language Model providers.

## Protocol

The server implements the **Model Context Protocol (MCP)** using JSON-RPC 2.0 over stdio transport.

### Transport Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **stdio** | Standard input/output | Claude Desktop, CLI tools |
| **HTTP** | HTTP transport | Web applications, testing |

## Quick Start

### 1. Initialize Session

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "my-client",
      "version": "1.0.0"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": {},
      "resources": {},
      "logging": {}
    },
    "serverInfo": {
      "name": "llm-cli-runner-server",
      "version": "0.1.0"
    }
  }
}
```

### 2. List Available Tools

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "gemini_generate",
        "description": "Generate text via agy (Antigravity CLI); deprecated alias, the Gemini CLI provider has been removed",
        "inputSchema": {
          "type": "object",
          "properties": {
            "prompt": {"type": "string"},
            "model": {"type": "string", "default": "gemini-2.5-flash-lite"}
          },
          "required": ["prompt"]
        }
      },
      {
        "name": "llama_generate",
        "description": "Generate text using local LLaMA model",
        "inputSchema": {...}
      }
    ]
  }
}
```

### 3. Call a Tool

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "gemini_generate",
    "arguments": {
      "prompt": "Write a haiku about programming",
      "model": "gemini-2.5-flash-lite"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Lines of code flow down\nBugs hide in the shadows deep\nDebugger awaits"
      }
    ]
  }
}
```

## Available Tools

### llm_complete with provider="gemini" (deprecated alias for agy)

There is no `gemini_generate` tool. The server registers a single `llm_complete` tool; `provider="gemini"` is a deprecated alias of `provider="agy"` (Antigravity CLI). The GeminiProvider implementation (Gemini CLI) has been removed, so the alias needs an installed, authenticated `agy` CLI, and legacy `gemini-*` model names are tier-mapped to agy models. Prefer `provider="agy"` in new code.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The prompt to generate from |
| `provider` | string | No | server default | `agy` (or the deprecated alias `gemini`) |
| `model` | string | No | agy default | Model to use |
| `temperature` | number | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | 1000 | Maximum tokens to generate |

**Example:**
```json
{
  "name": "llm_complete",
  "arguments": {
    "prompt": "Explain REST APIs in simple terms",
    "provider": "gemini",
    "max_tokens": 500
  }
}
```

> Note: the other `*_generate` sections of this guide are outdated; a broader documentation cleanup is pending.

### llama_generate

Generate text using local LLaMA model via llama-cpp-python.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The prompt to generate from |
| `max_tokens` | integer | No | 256 | Maximum tokens to generate |
| `temperature` | number | No | 0.7 | Sampling temperature |
| `top_p` | number | No | 0.9 | Top-p (nucleus) sampling |
| `top_k` | integer | No | 40 | Top-k sampling |

**Example:**
```json
{
  "name": "llama_generate",
  "arguments": {
    "prompt": "Write a short story about a robot",
    "max_tokens": 200,
    "temperature": 0.8
  }
}
```

### codex_generate

Generate code using OpenAI Codex models.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The prompt describing code to generate |
| `language` | string | No | `python` | Target programming language |
| `max_tokens` | integer | No | 512 | Maximum tokens to generate |

**Example:**
```json
{
  "name": "codex_generate",
  "arguments": {
    "prompt": "Write a function to merge two sorted arrays",
    "language": "python",
    "max_tokens": 300
  }
}
```

## Error Handling

### JSON-RPC Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32700 | Parse error | Invalid JSON received |
| -32600 | Invalid Request | Invalid JSON-RPC structure |
| -32601 | Method not found | Unknown method name |
| -32602 | Invalid params | Invalid method parameters |
| -32603 | Internal error | Server-side error |

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "details": "Missing required parameter: prompt"
    }
  }
}
```

### Tool Execution Errors

When a tool fails, the response includes `isError: true`:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Error: API rate limit exceeded"
      }
    ],
    "isError": true
  }
}
```

## Resource Operations

### List Resources

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/list",
  "params": {}
}
```

### Read Resource

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/read",
  "params": {
    "uri": "config://providers"
  }
}
```

## Configuration

> **Deprecation:** the `gemini` provider is a deprecated alias for `agy` (Antigravity CLI). The provider example below uses `agy`.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Not used; legacy only (deprecated alias for agy) |
| `OPENAI_API_KEY` | OpenAI API key | For Codex provider |
| `LLAMA_MODEL_PATH` | Path to LLaMA model file | For LLaMA provider |

### Provider Configuration

```toml
# config.toml
# The default provider is llama (configurable via default_provider or
# MCP_DEFAULT_PROVIDER); agy is the recommended CLI provider.
default_provider = "agy"
max_concurrent_requests = 10

[[providers]]
name = "agy"
provider_type = "agy"
enabled = true
model_name = "gemini-3.8-flash-low"

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
model_path = "${LLAMA_MODEL_PATH}"
context_size = 4096

[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
model_name = "code-davinci-002"
api_key = "${OPENAI_API_KEY}"
```

## Rate Limiting

The server implements rate limiting per provider:

| Provider | Default Limit | Burst |
|----------|---------------|-------|
| Gemini (deprecated alias for agy; historical Gemini CLI limit) | 60 req/min | 10 |
| Codex | 20 req/min | 5 |
| LLaMA | No limit | - |

## Streaming Responses

For long-running generations, streaming is supported:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "gemini_generate",
    "arguments": {
      "prompt": "Write a long essay about AI",
      "stream": true
    }
  }
}
```

Streaming responses are delivered as multiple JSON-RPC notifications.

## Security Considerations

1. **API Key Protection**: Never expose API keys in logs or responses
2. **Input Validation**: All inputs are validated against JSON schemas
3. **Rate Limiting**: Prevents abuse and API quota exhaustion
4. **Message Size Limits**: Maximum 1MB per message
5. **Nesting Depth Limits**: Maximum 100 levels to prevent DoS

## Client Libraries

### Python

```python
import asyncio
import json

async def call_tool(prompt: str) -> str:
    # Send request via stdio
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "gemini_generate",
            "arguments": {"prompt": prompt}
        }
    }
    # ... handle stdio communication
```

### TypeScript

```typescript
interface MCPClient {
  initialize(): Promise<InitializeResult>;
  listTools(): Promise<Tool[]>;
  callTool(name: string, args: Record<string, unknown>): Promise<ToolResult>;
}
```

## OpenAPI Specification

The full OpenAPI 3.1 specification is available at:
- `docs/api/openapi.yaml`

Generate client libraries using:
```bash
# Python client
openapi-generator generate -i docs/api/openapi.yaml -g python -o clients/python

# TypeScript client
openapi-generator generate -i docs/api/openapi.yaml -g typescript-fetch -o clients/typescript
```
