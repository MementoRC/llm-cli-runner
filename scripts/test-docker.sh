#!/bin/bash
# Docker-based cross-platform testing script

set -e

echo "🐳 Docker Cross-Platform Testing"
echo "================================="

# Available test environments
ENVIRONMENTS=("ubuntu" "alpine")

# Function to run tests in a specific environment
test_environment() {
    local env=$1
    echo "🧪 Testing in $env environment..."

    # Build the test image
    docker build -t llm-cli-test-$env \
        -f docker/test-environments/Dockerfile.$env .

    # Run tests
    docker run --rm \
        -v $(pwd):/workspace \
        -w /workspace \
        llm-cli-test-$env \
        sh -c "pixi install -e quality && pixi run -e quality test"

    echo "✅ $env environment tests passed"
}

# Function to run smoke tests
smoke_test() {
    local env=$1
    echo "💨 Smoke testing $env environment..."

    docker build -t llm-cli-test-$env \
        -f docker/test-environments/Dockerfile.$env .

    docker run --rm \
        -v $(pwd):/workspace \
        -w /workspace \
        llm-cli-test-$env \
        sh -c "pixi install -e quality && pixi list && echo 'Smoke test passed'"

    echo "✅ $env smoke test passed"
}

# Parse command line arguments
MODE=${1:-"test"}
ENVIRONMENT=${2:-"all"}

case $MODE in
    "smoke")
        if [ "$ENVIRONMENT" = "all" ]; then
            for env in "${ENVIRONMENTS[@]}"; do
                smoke_test $env
            done
        else
            smoke_test $ENVIRONMENT
        fi
        ;;
    "test")
        if [ "$ENVIRONMENT" = "all" ]; then
            for env in "${ENVIRONMENTS[@]}"; do
                test_environment $env
            done
        else
            test_environment $ENVIRONMENT
        fi
        ;;
    "clean")
        echo "🧹 Cleaning up Docker images..."
        for env in "${ENVIRONMENTS[@]}"; do
            docker rmi -f llm-cli-test-$env 2>/dev/null || true
        done
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 [smoke|test|clean] [ubuntu|alpine|all]"
        echo ""
        echo "Examples:"
        echo "  $0 smoke ubuntu    # Quick smoke test in Ubuntu"
        echo "  $0 test all        # Full test suite in all environments"
        echo "  $0 clean           # Clean up Docker images"
        exit 1
        ;;
esac

echo "🎉 Docker testing complete!"
