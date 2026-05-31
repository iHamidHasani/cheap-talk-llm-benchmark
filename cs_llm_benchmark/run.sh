#!/usr/bin/env bash
# End-to-end driver for the Crawford-Sobel LLM cheap-talk benchmark.
#
# Usage:
#   ./run.sh                  # full run with real APIs (requires keys)
#   PROVIDER_SET=stub ./run.sh   # offline pipeline check (no API calls)
#
# Environment variables consumed (only those needed by the chosen models):
#   OPENAI_API_KEY     for openai-provider models
#   ANTHROPIC_API_KEY  for anthropic-provider models
#   GOOGLE_API_KEY     for google-provider models
#   TOGETHER_API_KEY   (or any OpenAI-compatible) when using base_url
set -euo pipefail

PROVIDER_SET="${PROVIDER_SET:-default}"
RUN_DIR="${RUN_DIR:-runs/$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/results}"
RETRIES="${RETRIES:-3}"

# Make this script work whether invoked from inside cs_llm_benchmark/ or its
# parent. We always execute python from the parent so `python -m
# cs_llm_benchmark.*` resolves.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$HERE")"
cd "$PARENT"

mkdir -p "$RUN_DIR" "$RESULTS_DIR"
echo "[run.sh] PROVIDER_SET=$PROVIDER_SET  RUN_DIR=$RUN_DIR"

# 1. Install dependencies (idempotent; skip if already satisfied).
pip install -q -r cs_llm_benchmark/requirements.txt || true

# 2. Sender message collection (12,000 calls + 60 comprehension diagnostics).
python -m cs_llm_benchmark.run_experiment \
    --output_dir "$RUN_DIR" \
    --provider_set "$PROVIDER_SET" \
    --retries "$RETRIES"

# 3. Analysis: regenerate every empirical table from the logged messages.
python -m cs_llm_benchmark.analyze_results \
    --input_dir  "$RUN_DIR" \
    --output_dir "$RESULTS_DIR"

# 4. Sanity-print the oracle reference table.
python -m cs_llm_benchmark.oracle | head -n 40

echo "[run.sh] complete. Results in: $RESULTS_DIR"
