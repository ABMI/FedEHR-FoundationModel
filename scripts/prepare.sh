#!/usr/bin/env bash
# Base preprocessing: sample, events, demographics, values, units, deciles, tokens, visits, labels (CPU only).
# Usage: bash scripts/prepare.sh configs/local_base.toml
# Env: PYTHON_BIN (default .venv/bin/python)
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
config="${1:?base preprocessing config TOML required}"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
export PYTHONPATH="$project_root/src"

"$python_bin" -u -m ausom_ehr.data sample --config "$config"
"$python_bin" -u -m ausom_ehr.data events --config "$config" --all
"$python_bin" -u -m ausom_ehr.data demographics --config "$config"
"$python_bin" -u -m ausom_ehr.values parse --config "$config"
"$python_bin" -u -m ausom_ehr.values units --config "$config"
"$python_bin" -u -m ausom_ehr.values deciles --config "$config"
"$python_bin" -u -m ausom_ehr.tokens vocab --config "$config"
"$python_bin" -u -m ausom_ehr.tokens events --config "$config"
"$python_bin" -u -m ausom_ehr.data visits --config "$config"
"$python_bin" -u -m ausom_ehr.tasks labels --config "$config"
echo "Base preprocessing done: $config"
