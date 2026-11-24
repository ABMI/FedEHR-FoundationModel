#!/usr/bin/env bash
# Model inputs: link base preprocessing, joint vocab and events, sequences, task inputs (CPU only).
# Usage: bash scripts/prepare_joint.sh configs/local_m0.toml
# Env: PYTHON_BIN (default .venv/bin/python)
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
config="${1:?model config TOML required}"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
export PYTHONPATH="$project_root/src"
output_root="$("$python_bin" -c \
  'import sys; from ausom_ehr.common import load_config; print(load_config(sys.argv[1])["paths"]["output_root"])' "$config")"

"$python_bin" -u -m ausom_ehr.tokens link-base --config "$config"
"$python_bin" -u -m ausom_ehr.tokens joint-vocab --config "$config"
"$python_bin" -u -m ausom_ehr.tokens joint-events --config "$config"
if [[ ! -f "$output_root/sequences/summary.json" ]]; then
  "$python_bin" -u -m ausom_ehr.tokens sequences --config "$config"
fi
if [[ ! -f "$output_root/cohorts/inputs_summary.json" ]]; then
  # Audit totals span all buckets, so an interrupted run is rebuilt from scratch.
  rm -rf "$output_root/cohorts/inputs"
  "$python_bin" -u -m ausom_ehr.tasks inputs --config "$config"
fi
echo "Model inputs done: $output_root"
