# Configuration Examples

## Overview

MCP Server LLM CLI Runner supports configuration via:
1. Configuration files (TOML)
2. Environment variables
3. Command-line arguments

## Basic Configuration

### Minimal Setup

```toml
# config.toml
default_provider = "gemini"

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
```

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
LLAMA_MODEL_PATH=/path/to/model.gguf

# Server configuration
LLM_CLI_RUNNER_DEFAULT_PROVIDER=gemini
LLM_CLI_RUNNER_LOG_LEVEL=INFO
LLM_CLI_RUNNER_MAX_CONCURRENT=10
```

## Provider Configurations

### Gemini Provider

```toml
[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
model_name = "gemini-2.5-flash-lite"
api_key = "${GEMINI_API_KEY}"

# Optional settings
max_tokens = 1024
temperature = 0.7
timeout = 30
max_retries = 3

# Rate limiting
rate_limit = 60        # requests per minute
rate_limit_burst = 10  # burst allowance
```

### LLaMA Provider (Local)

```toml
[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
model_path = "${LLAMA_MODEL_PATH}"

# Model settings
context_size = 4096
n_gpu_layers = 35     # GPU acceleration layers
n_threads = 8         # CPU threads

# Generation defaults
max_tokens = 256
temperature = 0.7
top_p = 0.9
top_k = 40
repeat_penalty = 1.1

# Memory optimization
use_mmap = true
use_mlock = false
```

### OpenAI/Codex Provider

```toml
[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
api_key = "${OPENAI_API_KEY}"
model_name = "gpt-3.5-turbo"

# Optional: Organization ID
organization = "${OPENAI_ORG_ID}"

# API settings
base_url = "https://api.openai.com/v1"
timeout = 60
max_retries = 3

# Generation defaults
max_tokens = 512
temperature = 0.7
```

## Server Configurations

### Development

```toml
# config.dev.toml
[server]
host = "localhost"
port = 8080
debug = true
reload = true

[logging]
level = "DEBUG"
format = "detailed"
log_requests = true
log_responses = true

[cache]
enabled = false

default_provider = "gemini"

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
api_key = "${GEMINI_API_KEY}"
```

### Production

```toml
# config.prod.toml
[server]
host = "0.0.0.0"
port = 8080
debug = false
workers = 4

[logging]
level = "INFO"
format = "json"
log_requests = false
log_responses = false
output = "/var/log/llm-cli-runner/server.log"

[security]
max_message_size = 1048576  # 1MB
max_nesting_depth = 100
rate_limit_global = 1000    # requests per minute

[cache]
enabled = true
backend = "redis"
redis_url = "redis://localhost:6379/0"
ttl = 3600
max_size = 10000

default_provider = "gemini"

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
api_key = "${GEMINI_API_KEY}"
model_name = "gemini-2.5-flash-lite"
rate_limit = 100
```

### High Availability

```toml
# config.ha.toml
[server]
host = "0.0.0.0"
port = 8080
workers = 8

[health]
enabled = true
endpoint = "/health"
check_interval = 10

[metrics]
enabled = true
endpoint = "/metrics"
prometheus = true

[failover]
enabled = true
providers = ["gemini", "codex", "llama"]
retry_on_failure = true
max_retries = 3

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
priority = 1  # Primary
api_key = "${GEMINI_API_KEY}"

[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
priority = 2  # Secondary fallback
api_key = "${OPENAI_API_KEY}"

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
priority = 3  # Last resort (local)
model_path = "${LLAMA_MODEL_PATH}"
```

## Use Case Configurations

### Cost-Optimized

```toml
# Minimize API costs
default_provider = "llama"

[cache]
enabled = true
ttl = 86400  # 24 hours
max_size = 50000

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
priority = 1
model_path = "/models/llama-2-7b-q4.gguf"
context_size = 2048

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
priority = 2
api_key = "${GEMINI_API_KEY}"
model_name = "gemini-2.5-flash-lite"  # Cheaper than pro
max_tokens = 256  # Reduce token usage
```

### Performance-Optimized

```toml
# Maximum speed
default_provider = "gemini"

[server]
workers = 8

[cache]
enabled = true
backend = "memory"
ttl = 300
max_size = 10000

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
model_name = "gemini-2.5-flash-lite"  # Fast model
api_key = "${GEMINI_API_KEY}"
timeout = 10
max_concurrent = 20
```

### Code Generation Focus

```toml
# Optimized for code generation
default_provider = "codex"

[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
model_name = "gpt-4-turbo"
api_key = "${OPENAI_API_KEY}"
temperature = 0.2  # More deterministic for code
max_tokens = 2048

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
model_name = "gemini-1.5-pro"
api_key = "${GEMINI_API_KEY}"
temperature = 0.2
```

### Multi-Model Routing

```toml
# Route different tasks to different providers
default_provider = "gemini"

[routing]
enabled = true

[[routing.rules]]
pattern = "code|function|class|implement"
provider = "codex"

[[routing.rules]]
pattern = "translate|language"
provider = "gemini"
model = "gemini-1.5-pro"

[[routing.rules]]
pattern = "creative|story|poem"
provider = "gemini"
temperature = 0.9

[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
api_key = "${GEMINI_API_KEY}"

[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
api_key = "${OPENAI_API_KEY}"
```

## Caching Configurations

### Memory Cache (Development)

```toml
[cache]
enabled = true
backend = "memory"
ttl = 3600
max_size = 1000
```

### Redis Cache (Production)

```toml
[cache]
enabled = true
backend = "redis"
redis_url = "redis://localhost:6379/0"
ttl = 86400
max_size = 100000
key_prefix = "llm-cli-runner:"

[cache.redis]
max_connections = 20
socket_timeout = 5
retry_on_timeout = true
```

### Disk Cache (Large Models)

```toml
[cache]
enabled = true
backend = "disk"
path = "/var/cache/llm-cli-runner"
ttl = 604800  # 7 days
max_size_gb = 10
```

## Logging Configurations

### Development Logging

```toml
[logging]
level = "DEBUG"
format = "detailed"
colorize = true

# Log to console
output = "stdout"

# Additional debug options
log_requests = true
log_responses = true
log_timing = true
```

### Production Logging

```toml
[logging]
level = "INFO"
format = "json"
colorize = false

# Log to file with rotation
output = "/var/log/llm-cli-runner/server.log"
max_size = "100MB"
max_files = 10
compress = true

# Sampling for high-traffic
sample_rate = 0.1  # Log 10% of requests
```

### Structured Logging

```toml
[logging]
level = "INFO"
format = "json"

[logging.fields]
service = "llm-cli-runner"
environment = "${ENVIRONMENT}"
version = "0.1.0"

[logging.exclude_fields]
# Don't log sensitive data
- "api_key"
- "authorization"
- "prompt"  # May contain sensitive info
```

## Security Configurations

### Basic Security

```toml
[security]
# Message limits
max_message_size = 1048576  # 1MB
max_nesting_depth = 100

# Rate limiting
rate_limit_enabled = true
rate_limit_requests = 100
rate_limit_period = 60  # seconds

# API key validation
require_api_key = true
api_key_header = "X-API-Key"
```

### Enhanced Security

```toml
[security]
max_message_size = 524288  # 512KB
max_nesting_depth = 50

# Strict rate limiting
rate_limit_enabled = true
rate_limit_requests = 50
rate_limit_period = 60
rate_limit_burst = 10

# IP-based limits
ip_rate_limit_enabled = true
ip_rate_limit_requests = 200
ip_rate_limit_period = 60

# Request validation
validate_json_schema = true
reject_unknown_methods = true

# Timeout protection
request_timeout = 30
connection_timeout = 10

# Content filtering (optional)
[security.content_filter]
enabled = true
block_patterns = [
    "ignore previous instructions",
    "system prompt",
]
```

## Complete Example

```toml
# config.complete.toml
# Complete production configuration example

#------------------------------------------------------------------------------
# Server Settings
#------------------------------------------------------------------------------
[server]
host = "0.0.0.0"
port = 8080
workers = 4
debug = false

#------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------
[logging]
level = "INFO"
format = "json"
output = "/var/log/llm-cli-runner/server.log"
max_size = "100MB"
max_files = 10

#------------------------------------------------------------------------------
# Security
#------------------------------------------------------------------------------
[security]
max_message_size = 1048576
max_nesting_depth = 100
rate_limit_enabled = true
rate_limit_requests = 100
rate_limit_period = 60

#------------------------------------------------------------------------------
# Cache
#------------------------------------------------------------------------------
[cache]
enabled = true
backend = "redis"
redis_url = "${REDIS_URL}"
ttl = 3600
max_size = 10000

#------------------------------------------------------------------------------
# Health & Metrics
#------------------------------------------------------------------------------
[health]
enabled = true
endpoint = "/health"

[metrics]
enabled = true
endpoint = "/metrics"
prometheus = true

#------------------------------------------------------------------------------
# Default Provider
#------------------------------------------------------------------------------
default_provider = "gemini"
max_concurrent_requests = 20

#------------------------------------------------------------------------------
# Providers
#------------------------------------------------------------------------------
[[providers]]
name = "gemini"
provider_type = "gemini"
enabled = true
priority = 1
api_key = "${GEMINI_API_KEY}"
model_name = "gemini-2.5-flash-lite"
max_tokens = 1024
temperature = 0.7
timeout = 30
rate_limit = 60

[[providers]]
name = "codex"
provider_type = "openai"
enabled = true
priority = 2
api_key = "${OPENAI_API_KEY}"
model_name = "gpt-3.5-turbo"
max_tokens = 512
timeout = 60

[[providers]]
name = "llama"
provider_type = "llama"
enabled = true
priority = 3
model_path = "${LLAMA_MODEL_PATH}"
context_size = 4096
n_gpu_layers = 35
use_mmap = true
```

## Validation

Validate your configuration:

```bash
# Check TOML syntax
pixi run python -c "import toml; toml.load('config.toml')"

# Validate against schema
pixi run validate-config config.toml

# Test configuration
pixi run serve --config config.toml --dry-run
```
