# Troubleshooting Guide

## Common Issues

### Connection Issues

#### Server Not Starting

**Symptoms:**
- Server fails to start
- No response from MCP client

**Solutions:**

1. Check Python version:
   ```bash
   python --version  # Must be >= 3.12
   ```

2. Verify pixi installation:
   ```bash
   pixi --version
   pixi install  # Reinstall dependencies
   ```

3. Check for port conflicts:
   ```bash
   lsof -i :8080  # Check if port is in use
   ```

4. Review logs:
   ```bash
   pixi run serve-debug  # Enable verbose logging
   ```

#### Connection Refused

**Symptoms:**
- `Connection refused` errors
- Timeout when connecting

**Solutions:**

1. Verify server is running:
   ```bash
   ps aux | grep mcp_server_llm_cli_runner
   ```

2. Check network configuration:
   ```bash
   netstat -tlnp | grep 8080
   ```

3. For Docker deployments:
   ```bash
   docker logs llm-cli-runner
   docker inspect llm-cli-runner | grep IPAddress
   ```

### Provider Issues

#### Gemini (deprecated alias for agy) Errors

> **Deprecated:** the `gemini` provider is a deprecated alias for `agy`. The GeminiProvider (Gemini CLI) implementation has been removed, so `GEMINI_API_KEY` and direct Google API calls no longer apply. If `agy` is not installed, `gemini` calls fail (there is no fallback to the old Gemini CLI).

**Symptoms:**
- `gemini` requests fail while `agy` requests work
- Rate limit exceeded

**Solutions:**

1. Verify the `agy` binary is available:
   ```bash
   command -v agy || echo "$AGY_CLI_PATH"   # agy must be on PATH or AGY_CLI_PATH must point to it
   ```

2. Switch the request or config to `provider: "agy"` to remove the alias from the path.

3. Check quota for the Antigravity account used by `agy`.

#### LLaMA Model Errors

**Symptoms:**
- `Model file not found`
- `Out of memory` errors
- Slow generation

**Solutions:**

1. Verify model path:
   ```bash
   ls -la $LLAMA_MODEL_PATH
   ```

2. Check file integrity:
   ```bash
   md5sum $LLAMA_MODEL_PATH  # Compare with expected hash
   ```

3. For memory issues:
   ```bash
   # Reduce context size in config
   context_size = 2048  # Lower value

   # Or use quantized model
   model_path = "model-q4_0.gguf"  # 4-bit quantized
   ```

4. Enable memory mapping:
   ```python
   # In config
   use_mmap = true
   use_mlock = false
   ```

#### Codex / OpenAI

Codex is not a provider here. Use the `codex` MCP server (`codex mcp-server`) directly; this server only provides `agy` (alias `gemini`) and `llama`.

#### Unknown provider in configuration

**Symptoms:**
- `Unknown provider '<name>' in configuration. Valid providers: agy, gemini, google, llama`

**Solution:** Rename the entry under `[providers]` to one of the listed names.

### Protocol Issues

#### JSON-RPC Errors

**Error: Parse error (-32700)**

Cause: Invalid JSON in request

Solution:
```python
# Validate JSON before sending
import json
try:
    json.loads(request_string)
except json.JSONDecodeError as e:
    print(f"Invalid JSON at position {e.pos}: {e.msg}")
```

**Error: Invalid Request (-32600)**

Cause: Missing required JSON-RPC fields

Solution:
```json
{
  "jsonrpc": "2.0",  // Required
  "id": 1,           // Required for requests
  "method": "...",   // Required
  "params": {}       // Optional
}
```

**Error: Method not found (-32601)**

Cause: Unknown method name

Valid methods:
- `initialize`
- `tools/list`
- `tools/call`
- `resources/list`
- `resources/read`

**Error: Invalid params (-32602)**

Cause: Missing or invalid parameters

Solution:
```json
{
  "method": "tools/call",
  "params": {
    "name": "gemini_generate",  // Required
    "arguments": {
      "prompt": "..."            // Required for most tools
    }
  }
}
```

### Performance Issues

#### Slow Response Times

**Diagnosis:**

1. Check metrics:
   ```bash
   curl http://localhost:8080/metrics | grep latency
   ```

2. Profile requests:
   ```python
   import time
   start = time.time()
   # ... make request ...
   print(f"Request took {time.time() - start:.2f}s")
   ```

**Solutions:**

1. Enable caching:
   ```toml
   [cache]
   enabled = true
   ttl = 3600
   max_size = 1000
   ```

2. Reduce max_tokens:
   ```json
   {"arguments": {"max_tokens": 256}}  // Instead of 1024
   ```

3. Use faster models:
   ```json
   {"arguments": {"model": "gemini-3.8-flash-low"}}  // Instead of gemini-3.8-flash-high (agy models)
   ```

#### High Memory Usage

**Diagnosis:**

```bash
# Check process memory
ps aux | grep mcp_server
top -p $(pgrep -f mcp_server_llm_cli_runner)
```

**Solutions:**

1. For LLaMA:
   - Use smaller/quantized model
   - Reduce context size
   - Enable memory mapping

2. Limit concurrent requests:
   ```toml
   max_concurrent_requests = 5
   ```

3. Enable garbage collection:
   ```python
   import gc
   gc.collect()
   ```

### Configuration Issues

#### Config File Not Found

**Solutions:**

1. Check default locations:
   ```bash
   ls -la config.toml
   ls -la ~/.config/llm-cli-runner/config.toml
   ```

2. Specify config path:
   ```bash
   pixi run serve --config /path/to/config.toml
   ```

3. Use environment variables instead:
   ```bash
   export MCP_DEFAULT_PROVIDER=llama  # default; set to agy to prefer the CLI provider
   export LLM_CLI_RUNNER_MAX_CONCURRENT=10
   ```

#### Invalid Configuration

**Validation:**

```bash
# Validate TOML syntax
pixi run python -c "import toml; toml.load('config.toml')"

# Validate configuration schema
pixi run validate-config config.toml
```

### Debugging

#### Enable Debug Logging

```bash
# Environment variable
export LOG_LEVEL=DEBUG

# Or in config
[logging]
level = "DEBUG"
format = "detailed"
```

#### Request/Response Logging

```python
# Add to configuration
[logging]
log_requests = true
log_responses = true
```

#### Interactive Debugging

```bash
# Start with debugger
pixi run python -m pdb -m mcp_server_llm_cli_runner
```

## FAQ

### General

**Q: Which provider is most cost-effective?**

A: LLaMA (local) is free after initial model download. Cloud provider costs depend on the provider you use.

**Q: Can I use multiple providers simultaneously?**

A: Yes, configure multiple providers and specify which to use per request.

**Q: What's the maximum request size?**

A: 1MB by default. Configure with `max_message_size` setting.

### Technical

**Q: Does it support streaming?**

A: Yes, use `stream: true` in tool arguments for supported providers.

**Q: Can I run multiple instances?**

A: Yes, use different ports or containerized deployments with load balancing.

**Q: Is there a rate limit?**

A: Server-side rate limiting is configurable. Provider limits depend on your API tier.

### Troubleshooting

**Q: Why are my requests timing out?**

A: Check provider connectivity, reduce max_tokens, or increase timeout settings.

**Q: How do I reset the cache?**

A: Restart the server or call the cache clear endpoint (if enabled).

**Q: Where are logs stored?**

A: By default, stdout. Configure `log_output` for file logging.

## Getting Help

1. Check logs: `pixi run serve-debug`
2. Review configuration: `pixi run validate-config`
3. Test providers: `pixi run test-providers`
4. Open issue: Include logs, config (redacted), and reproduction steps
