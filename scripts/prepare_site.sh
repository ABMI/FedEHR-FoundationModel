#!/usr/bin/env bash
# Preprocess one simulated site with the M0 rules and vocabulary (no training).
# Usage: bash scripts/prepare_site.sh D0_random  (reads {CONFIG_PREFIX}_{site}.toml and {CONFIG_PREFIX}_{site}_joint.toml)
# Env: SITES_ROOT (default outputs/sites), CONFIG_PREFIX (default configs/local_site), PYTHON_BIN
set -euo pipefail
site="${1:?site name required (e.g. D0_random)}"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
sites_root="${SITES_ROOT:-outputs/sites}"
config_prefix="${CONFIG_PREFIX:-configs/local_site}"
if [[ ! -f "${sites_root}/${site}/base/cohorts/labels_summary.json" ]]; then
  PYTHON_BIN="$python_bin" bash scripts/prepare.sh "${config_prefix}_${site}.toml"
fi
PYTHON_BIN="$python_bin" bash scripts/prepare_joint.sh "${config_prefix}_${site}_joint.toml"
echo "Site preprocessing done: $site"
