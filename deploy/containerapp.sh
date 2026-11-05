#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m deploy.azure_guard
exec bash deploy/canary.sh promote
