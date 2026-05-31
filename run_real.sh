#!/usr/bin/env bash
# Load .env secrets, then run the real experiment + analysis.
#
# Usage:
#   ./run_real.sh path_y          # recommended basket (default here)
#   ./run_real.sh path_x          # OpenAI + Anthropic only
#   ./run_real.sh default         # Arash's original basket
#
# Reads keys from .env (gitignored). Outputs to runs/<basket>-<timestamp>/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BASKET="${1:-path_y}"

# Load .env if present (export every non-comment KEY=VALUE line).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source <(grep -E '^[A-Z_]+=' .env)
  set +a
  echo "[run_real] loaded .env"
else
  echo "[run_real] no .env found — relying on existing environment"
fi

RUN_DIR="runs/${BASKET}-$(date +%Y%m%d-%H%M%S)"
RESULTS_DIR="${RUN_DIR}/results"
mkdir -p "$RUN_DIR" "$RESULTS_DIR"
echo "[run_real] basket=$BASKET  run_dir=$RUN_DIR"

pip install -q -r cs_llm_benchmark/requirements.txt || true

python -m cs_llm_benchmark.run_experiment \
    --output_dir "$RUN_DIR" --provider_set "$BASKET" --retries 4

python -m cs_llm_benchmark.analyze_results \
    --input_dir "$RUN_DIR" --output_dir "$RESULTS_DIR"

echo "[run_real] complete → $RESULTS_DIR"
