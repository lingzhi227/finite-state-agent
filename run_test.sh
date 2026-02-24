#!/bin/bash
# Run integration tests outside of Claude Code session
# Usage: ./run_test.sh

cd "$(dirname "$0")"
unset CLAUDECODE

echo "=== Running unit tests ==="
uv run pytest tests/test_core.py -v

echo ""
echo "=== Running integration tests (real Claude CLI calls) ==="
uv run pytest tests/test_claude.py -v

echo ""
echo "=== Running math agent example ==="
uv run python examples/math_agent.py
