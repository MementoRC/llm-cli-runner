# Deployment Guide

## Overview

This guide covers deploying MCP Server LLM CLI Runner in various environments.

## Deployment Options

| Environment | Transport | Best For |
|-------------|-----------|----------|
| Local Development | stdio | Testing, debugging |
| Claude Desktop | stdio | Personal use |
| Docker | HTTP/stdio | Containerized deployments |
| Kubernetes | HTTP | Production, scaling |
| AWS Lambda | HTTP | Serverless |

## Local Development

### Prerequisites

- Python >= 3.12
- Pixi package manager
- API keys for enabled providers

### Setup

```bash
# Clone repository
git clone <repository-url>
cd llm-cli-runner

# Install dependencies
pixi install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start server
pixi run serve
```

### Environment Variables

```bash
# Required for Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# Required for OpenAI/Codex
export OPENAI_API_KEY="your-openai-api-key"

# Optional for LLaMA
export LLAMA_MODEL_PATH="/path/to/model.gguf"
```

## Claude Desktop Integration

### Configuration

Add to your Claude Desktop configuration (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "llm-cli-runner": {
      "command": "pixi",
      "args": ["run", "-e", "default", "python", "-m", "mcp_server_llm_cli_runner"],
      "cwd": "/path/to/llm-cli-runner",
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

### Verification

1. Restart Claude Desktop
2. Open a new conversation
3. Check that llm-cli-runner tools are available

## Docker Deployment

### Building the Image

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies
RUN pixi install

# Expose HTTP port (if using HTTP transport)
EXPOSE 8080

# Default command
CMD ["pixi", "run", "serve"]
```

Build and run:

```bash
# Build image
docker build -t mcp-llm-cli-runner:latest .

# Run with environment variables
docker run -d \
  --name llm-cli-runner \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -p 8080:8080 \
  mcp-llm-cli-runner:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  llm-cli-runner:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config:ro
      - ./models:/app/models:ro  # For LLaMA models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-cli-runner
  labels:
    app: llm-cli-runner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-cli-runner
  template:
    metadata:
      labels:
        app: llm-cli-runner
    spec:
      containers:
        - name: llm-cli-runner
          image: mcp-llm-cli-runner:latest
          ports:
            - containerPort: 8080
          env:
            - name: GEMINI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: gemini-api-key
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: openai-api-key
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: llm-cli-runner
spec:
  selector:
    app: llm-cli-runner
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
type: Opaque
stringData:
  gemini-api-key: "your-gemini-api-key"
  openai-api-key: "your-openai-api-key"
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-cli-runner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-cli-runner
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## AWS Lambda Deployment

### Handler

```python
# lambda_handler.py
import json
from mcp_server_llm_cli_runner.server.handlers import LLMCliRunnerServer

server = LLMCliRunnerServer()

async def handler(event, context):
    """AWS Lambda handler for MCP requests."""
    await server.initialize()

    body = json.loads(event.get('body', '{}'))
    response = await server.process_request(body)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response)
    }
```

### SAM Template

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  LLMCliRunnerFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_handler.handler
      Runtime: python3.12
      Timeout: 30
      MemorySize: 1024
      Environment:
        Variables:
          GEMINI_API_KEY: !Ref GeminiApiKey
      Events:
        Api:
          Type: Api
          Properties:
            Path: /rpc
            Method: POST

Parameters:
  GeminiApiKey:
    Type: String
    NoEcho: true
```

## Production Considerations

### Scaling

| Component | Consideration |
|-----------|---------------|
| Replicas | Start with 2-3, scale based on load |
| Memory | 512MB minimum, 2GB for LLaMA |
| CPU | 250m minimum, 1 core for LLaMA |

### Monitoring

```yaml
# Prometheus metrics endpoint
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"
```

### Logging

```bash
# Structured logging configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_OUTPUT=stdout
```

### Security

1. **TLS/SSL**: Always use HTTPS in production
2. **API Keys**: Store in secrets management (Vault, AWS Secrets Manager)
3. **Network Policies**: Restrict ingress/egress
4. **Rate Limiting**: Configure per-client limits

### High Availability

```yaml
# Pod disruption budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: llm-cli-runner-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: llm-cli-runner
```

## Health Checks

### Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness check |
| `/ready` | Readiness check |
| `/metrics` | Prometheus metrics |

### Health Check Response

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "providers": {
    "gemini": "available",
    "llama": "available",
    "codex": "unavailable"
  }
}
```

## Rollback Procedures

### Quick Rollback

```bash
# Kubernetes
kubectl rollout undo deployment/llm-cli-runner

# Docker
docker stop llm-cli-runner
docker run -d --name llm-cli-runner mcp-llm-cli-runner:previous-tag
```

### Gradual Rollback

```yaml
# Canary deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```
