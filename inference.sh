#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: ./inference.sh \"your message\""
  exit 1
fi

python scripts/inference.py --config configs/inference.yaml --message "$1"
