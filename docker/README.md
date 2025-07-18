# Docker-Based Cross-Platform Testing

This directory contains Docker environments for testing the MCP server across different Linux distributions and deployment scenarios.

## Strategy

- **Local Development**: Optimized for `linux-64` with pixi
- **CI Testing**: Docker containers simulate different deployment environments
- **Fast Feedback**: Minimal test matrix focusing on common targets

## Available Environments

| Environment | Base Image | Use Case |
|-------------|------------|----------|
| `ubuntu` | ubuntu:22.04 | Most common deployment target |
| `alpine` | alpine:3.18 | Lightweight container deployments |
| `centos` | centos:stream9 | Enterprise/RHEL environments |

## Usage

### Local Testing

```bash
# Quick smoke test
./scripts/test-docker.sh smoke ubuntu

# Full test suite in all environments
./scripts/test-docker.sh test all

# Test specific environment
./scripts/test-docker.sh test alpine

# Cleanup Docker images
./scripts/test-docker.sh clean
```

### Docker Compose

```bash
# Run all tests
cd docker && docker-compose -f docker-compose.test.yml up

# Run specific environment
cd docker && docker-compose -f docker-compose.test.yml up test-ubuntu

# Smoke test only
cd docker && docker-compose -f docker-compose.test.yml up smoke-test
```

### Manual Docker Commands

```bash
# Build test image
docker build -t llm-cli-test-ubuntu -f docker/test-environments/Dockerfile.ubuntu .

# Run tests
docker run --rm -v $(pwd):/workspace -w /workspace \
  llm-cli-test-ubuntu sh -c "pixi install -e quality && pixi run -e quality test"
```

## CI Integration

The GitHub Actions workflow automatically runs Docker-based tests on:
- Ubuntu 22.04 (primary)
- Alpine 3.18 (lightweight)

CentOS testing is available but disabled by default to optimize CI duration.

## Performance Considerations

- **Build Caching**: Docker layer caching reduces build times
- **Parallel Execution**: Test environments run in parallel
- **Selective Testing**: Enable/disable environments based on needs

## Adding New Environments

1. Create `Dockerfile.{name}` in `test-environments/`
2. Add environment to `docker-compose.test.yml`
3. Update CI workflow matrix if needed
4. Test locally before enabling in CI

## Troubleshooting

**Common Issues:**
- **Pixi installation fails**: Check base image has `curl` and `bash`
- **Dependency conflicts**: Verify system packages in Dockerfile
- **Permission errors**: Ensure proper file permissions in container

**Debug Commands:**
```bash
# Interactive container for debugging
docker run -it --rm -v $(pwd):/workspace -w /workspace \
  llm-cli-test-ubuntu bash

# Check pixi installation
docker run --rm llm-cli-test-ubuntu pixi --version
```
