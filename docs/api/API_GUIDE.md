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
        "name": "llm_complete",
        "description": "Execute LLM completion with provider selection and streaming support",
        "inputSchema": {...}
      },
      {
        "name": "list_providers",
        "description": "List available LLM providers and their status",
        "inputSchema": {...}
      },
      {
        "name": "provider_info",
        "description": "Get detailed information about a specific provider",
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
    "name": "llm_complete",
    "arguments": {
      "prompt": "Write a haiku about programming",
      "provider": "agy"
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

The server registers three tools. Providers are `agy` (Antigravity CLI; `gemini` is a deprecated alias), `llama`, and `claude` (alias `anthropic`; Claude Code CLI via your subscription login, explicit-only: never auto-selected or used as a fallback). Codex is not a provider here; use the `codex` MCP server (`codex mcp-server`) directly.

### llm_complete

Execute an LLM completion with provider selection and optional streaming.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | The prompt to send to the LLM |
| `provider` | string | No | server default | `agy` (alias `gemini`), `llama`, or `claude` (alias `anthropic`, explicit-only). If omitted, the configured default is used and auto-routing applies if it is unavailable; a named provider fails with `PROVIDER_UNAVAILABLE` if unavailable |
| `model` | string | No | provider default | Provider-specific model name |
| `temperature` | number | No | 0.7 | Sampling temperature (0.0-2.0) |
| `max_tokens` | integer | No | provider default | Maximum tokens to generate |
| `stream` | boolean | No | false | Enable streaming response |

**Example:**
```json
{
  "name": "llm_complete",
  "arguments": {
    "prompt": "Explain quantum computing",
    "provider": "llama",
    "temperature": 0.3,
    "max_tokens": 500
  }
}
```

### list_providers

List available providers and their status. Takes no parameters.

### provider_info

Get detailed information about one provider.

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `provider` | string | Yes | `agy` (alias `gemini`), `llama`, or `claude` (alias `anthropic`) |

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
```

## Rate Limiting

The server implements rate limiting per provider:

| Provider | Default Limit | Burst |
|----------|---------------|-------|
| Gemini (deprecated alias for agy; historical Gemini CLI limit) | 60 req/min | 10 |
| LLaMA | No limit | - |

## Streaming Responses

For long-running generations, streaming is supported:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "llm_complete",
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
            "name": "llm_complete",
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
