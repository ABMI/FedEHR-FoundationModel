#!/usr/bin/env bash
# Evaluate non-pretrained count baselines (EHRSHOT-style) on the M0 cohort, labels and inputs (CPU only).
# Usage: bash scripts/run_baselines.sh configs/local_m0.toml [MODEL ...]  (default: age_sex_lr count_lr count_gbm; also count_answers_gbm)
# Env: THREADS (LightGBM threads, default 40), PYTHON_BIN (default .venv/bin/python)
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
config="${1:?model config TOML required}"
shift
if (( $# )); then
  models=("$@")
else
  models=(age_sex_lr count_lr count_gbm)
fi
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
export PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
"$python_bin" -u -m ausom_ehr.evaluate count-baselines --config "$config" \
  --models "${models[@]}" --threads "${THREADS:-40}"
