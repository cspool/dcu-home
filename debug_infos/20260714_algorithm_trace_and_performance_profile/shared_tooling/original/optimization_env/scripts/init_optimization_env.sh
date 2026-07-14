#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ENV_DIR/results" "$ENV_DIR/logs" "$ENV_DIR/manifests"

if [ ! -f "$ENV_DIR/.env" ]; then
    cp "$ENV_DIR/env.example" "$ENV_DIR/.env"
    echo "created $ENV_DIR/.env"
else
    echo "kept existing $ENV_DIR/.env"
fi

python3 "$SCRIPT_DIR/check_no_speculative.py"

echo "optimization environment initialized"
echo "no vLLM server or benchmark was started"
