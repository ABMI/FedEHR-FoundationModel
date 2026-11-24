#!/usr/bin/env bash
# FedAvg from scratch over 7 disjoint sites (A, D0-D5; 1 round = 1 epoch per site), then evaluate the final model on each site's test set.
# Usage: setsid nohup bash scripts/run_fedavg_disjoint.sh > outputs/sites_disjoint/logs/fedavg_7sites.log 2>&1 &
# Env: ROUNDS=20. A rerun skips finished rounds and sites.
set -uo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
py="$project_root/.venv/bin/python"
export PYTHONPATH="$project_root/src" CUDA_DEVICE_ORDER=PCI_BUS_ID
rounds="${ROUNDS:-20}"
root=outputs/sites_disjoint
run_dir="$root/fedavg_7sites"

echo "FedAvg-7 ${rounds} rounds started: $(date)"
# A goes first: the first client's config (M0 seed) builds the initial weights.
"$py" -u -m ausom_ehr.federated server --run-dir "$run_dir" --rounds "$rounds" \
  --client "A=configs/local_m0.toml@3" \
  --client "D3_cancer=configs/local_disjoint_D3_cancer_joint.toml@2" \
  --client "D0_random=configs/local_site_D0_random_joint.toml@2" \
  --client "D5_elderly=configs/local_disjoint_D5_elderly_joint.toml@3" \
  --client "D2_er=configs/local_disjoint_D2_er_joint.toml@4" \
  --client "D4_female=configs/local_disjoint_D4_female_joint.toml@4" \
  --client "D1_pediatric=configs/local_disjoint_D1_pediatric_joint.toml@4" || exit 1
echo "Training finished: $(date)"

evaluate() {
  local gpu=$1 site population
  shift
  for site in "$@"; do
    if [[ $site == A ]]; then population=outputs/m0; else population=$root/$site/joint; fi
    CUDA_VISIBLE_DEVICES=$gpu "$py" -u -m ausom_ehr.federated eval --model "$run_dir/final/state.pt" \
      --population-root "$population" --output "$root/eval/fedavg7_round${rounds}@${site}" --run-name "epoch${rounds}" \
      > "$root/logs/eval_fedavg7_round${rounds}@${site}.log" 2>&1
    echo "Eval ${site} finished (exit $?): $(date)"
  done
}
evaluate 2 D3_cancer D0_random &
evaluate 3 D5_elderly A &
evaluate 4 D2_er D4_female D1_pediatric &
wait
echo "FedAvg-7 training and eval finished: $(date)"
