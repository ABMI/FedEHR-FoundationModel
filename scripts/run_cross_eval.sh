#!/usr/bin/env bash
# Evaluate per-site FMs (epoch 20) on a target site's tasks (frozen FM + logistic regression); finished runs are skipped.
# Usage: bash scripts/run_cross_eval.sh D5_elderly "2:A,D0_random 3:D1_pediatric,D2_er 4:D3_cancer,D4_female"  (per GPU: GPU:SITE,SITE)
set -uo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
py="$project_root/.venv/bin/python"
export PYTHONPATH="$project_root/src" CUDA_DEVICE_ORDER=PCI_BUS_ID
target="${1:?target site required}"
plan="${2:?per-GPU site list required, e.g. \"2:A,D0_random 3:D1_pediatric\"}"
root=outputs/sites_disjoint

population() { if [[ $1 == A ]]; then echo outputs/m0; else echo "$root/$1/joint"; fi; }
checkpoint() { if [[ $1 == A ]]; then echo "$project_root/outputs/m0/training/continued/epoch-20/state.pt"; else echo "$project_root/$root/$1/joint/training/continued/epoch-20/state.pt"; fi; }

problems=0
target_pop=$(population "$target")
for need in tokens/vocab.parquet cohorts/inputs cohorts/inputs_summary.json; do
  [[ -e $target_pop/$need ]] || { echo "Missing eval input: $target_pop/$need"; problems=1; }
done
for item in $plan; do
  IFS=, read -ra sites <<< "${item#*:}"
  for site in "${sites[@]}"; do
    [[ -f $(checkpoint "$site") ]] || { echo "Missing checkpoint: $(checkpoint "$site")"; problems=1; }
    cmp -s "$(population "$site")/tokens/vocab.parquet" "$target_pop/tokens/vocab.parquet" || { echo "Vocabulary differs: $site"; problems=1; }
  done
done
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
       awk -F', ' '$1 >= 2 && $1 <= 4 && $2 > 1000 {printf "%s ", $1}')
[[ -z $busy ]] || { echo "GPUs already in use: $busy"; problems=1; }
(( problems == 0 )) || { echo "Preflight failed; not starting."; exit 1; }

evaluate() {
  local gpu=$1 site out
  shift
  for site in "$@"; do
    out="$root/eval/${site}_epoch20@${target}"
    if [[ -f $out/evaluation/continued/epoch20/results.json ]]; then
      echo "Eval ${site}@${target} already exists"; continue
    fi
    echo "Eval ${site}@${target} started (GPU $gpu): $(date)"
    CUDA_VISIBLE_DEVICES=$gpu "$py" -u -m ausom_ehr.federated eval --model "$(checkpoint "$site")" \
      --population-root "$target_pop" --output "$out" --run-name epoch20 \
      > "$root/logs/eval_${site}_epoch20@${target}.log" 2>&1
    echo "Eval ${site}@${target} finished (exit $?): $(date)"
  done
}
echo "Cross eval -> ${target}: $plan"
for item in $plan; do
  IFS=, read -ra sites <<< "${item#*:}"
  evaluate "${item%%:*}" "${sites[@]}" &
done
wait
echo "Cross eval -> ${target} all finished: $(date)"
