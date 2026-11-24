#!/usr/bin/env bash
# Leave-one-site-out FedAvg: train on 6 sites from the FedAvg-7 initial weights, evaluate the final model on all 7 test sets.
# Usage: setsid nohup bash scripts/run_fedavg_loso.sh D5_elderly > outputs/sites_disjoint/logs/fedavg6_minus_D5_elderly.log 2>&1 &
# Env: ROUNDS=20, DRY_RUN=0 (1 = checks and printed commands only), GPU_MEMORY_FRACTIONS (e.g. 4:0.5), SKIP_GPU_CHECK=0. A rerun skips finished work.
set -uo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
py="$project_root/.venv/bin/python"
export PYTHONPATH="$project_root/src" CUDA_DEVICE_ORDER=PCI_BUS_ID
exclude="${1:?site to exclude required: A D0_random D1_pediatric D2_er D3_cancer D4_female D5_elderly}"
rounds="${ROUNDS:-20}"
root=outputs/sites_disjoint
run_dir="$root/fedavg6_minus_${exclude}"
tag="fedavg6_minus_${exclude}_round${rounds}"
reference_init="$root/fedavg_7sites/round_00/global.pt"

declare -A config=(
  [A]=configs/local_m0.toml
  [D0_random]=configs/local_site_D0_random_joint.toml
  [D1_pediatric]=configs/local_disjoint_D1_pediatric_joint.toml
  [D2_er]=configs/local_disjoint_D2_er_joint.toml
  [D3_cancer]=configs/local_disjoint_D3_cancer_joint.toml
  [D4_female]=configs/local_disjoint_D4_female_joint.toml
  [D5_elderly]=configs/local_disjoint_D5_elderly_joint.toml
)
case "$exclude" in
  A)            plan="D3_cancer@2 D0_random@3 D5_elderly@4 D2_er@3 D4_female@2 D1_pediatric@4" ;;
  D0_random)    plan="A@2 D3_cancer@3 D5_elderly@4 D2_er@2 D4_female@3 D1_pediatric@4" ;;
  D1_pediatric) plan="A@2 D3_cancer@3 D0_random@3 D5_elderly@4 D2_er@2 D4_female@4" ;;
  D2_er)        plan="A@2 D3_cancer@3 D0_random@2 D5_elderly@4 D4_female@3 D1_pediatric@4" ;;
  D3_cancer)    plan="A@2 D0_random@2 D5_elderly@3 D2_er@4 D4_female@4 D1_pediatric@3" ;;
  D4_female)    plan="A@2 D3_cancer@3 D0_random@3 D5_elderly@4 D2_er@2 D1_pediatric@4" ;;
  D5_elderly)   plan="A@2 D3_cancer@3 D0_random@2 D2_er@4 D4_female@4 D1_pediatric@3" ;;
  *) echo "Unknown site: $exclude" >&2; exit 2 ;;
esac

population() { if [[ $1 == A ]]; then echo outputs/m0; else echo "$root/$1/joint"; fi; }
model="$project_root/$run_dir/final/state.pt"

# Preflight, including eval inputs, so a long run does not fail at evaluation.
problems=0
for site in "${!config[@]}"; do
  [[ -f ${config[$site]} ]] || { echo "Missing config: ${config[$site]}"; problems=1; }
  pop=$(population "$site")
  for need in tokens/vocab.parquet cohorts/inputs cohorts/inputs_summary.json; do
    [[ -e $pop/$need ]] || { echo "Missing eval input: $pop/$need"; problems=1; }
  done
  link="$root/eval/${tag}@${site}/training/continued/best/state.pt"
  if [[ -L $link && $(readlink "$link") != "$model" ]]; then
    echo "Eval folder points to a different model: $link -> $(readlink "$link")"; problems=1
  fi
done
[[ -f $reference_init ]] || { echo "Missing FedAvg-7 initial weights: $reference_init"; problems=1; }
if [[ -f $run_dir/round_00/global.pt ]] && ! cmp -s "$run_dir/round_00/global.pt" "$reference_init"; then
  echo "Initial weights differ from FedAvg-7: $run_dir/round_00/global.pt"; problems=1
fi
if [[ ${SKIP_GPU_CHECK:-0} != 1 ]]; then
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
         awk -F', ' '$1 >= 2 && $1 <= 4 && $2 > 1000 {printf "%s ", $1}')
  [[ -z $busy ]] || { echo "GPUs already in use: $busy"; problems=1; }
fi
(( problems == 0 )) || { echo "Preflight failed; not starting."; exit 1; }

clients=()
for item in $plan; do
  name=${item%@*}; gpu=${item#*@}
  clients+=(--client "$name=${config[$name]}@$gpu")
done
echo "FedAvg-6 (-${exclude}) ${rounds} rounds, placement: $plan, GPU memory limit: ${GPU_MEMORY_FRACTIONS:-none}"
if [[ ${DRY_RUN:-0} == 1 ]]; then
  echo "[DRY_RUN] $py -u -m ausom_ehr.federated server --run-dir $run_dir --rounds $rounds ${clients[*]}"
  echo "[DRY_RUN] eval: GPU2 D3_cancer,D4_female; GPU3 D5_elderly,A; GPU4 D2_er,D0_random,D1_pediatric -> $root/eval/${tag}@{site}"
  exit 0
fi

mkdir -p "$run_dir/round_00"
[[ -f $run_dir/round_00/global.pt ]] || cp "$reference_init" "$run_dir/round_00/global.pt"

echo "Training started: $(date)"
"$py" -u -m ausom_ehr.federated server --run-dir "$run_dir" --rounds "$rounds" "${clients[@]}" || exit 1
echo "Training finished: $(date)"
sleep 60  # let client processes release the GPUs

fraction_for() {
  local item
  for item in ${GPU_MEMORY_FRACTIONS//,/ }; do
    [[ ${item%%:*} == "$1" ]] && { echo "${item#*:}"; return; }
  done
  echo 1
}
evaluate() {
  local gpu=$1 site out
  shift
  export GPU_MEMORY_FRACTION
  GPU_MEMORY_FRACTION=$(fraction_for "$gpu")
  for site in "$@"; do
    out="$root/eval/${tag}@${site}"
    if [[ -f $out/evaluation/continued/epoch${rounds}/results.json ]]; then
      echo "Eval ${site} already exists"; continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu "$py" -u -m ausom_ehr.federated eval --model "$run_dir/final/state.pt" \
      --population-root "$(population "$site")" --output "$out" --run-name "epoch${rounds}" \
      > "$root/logs/eval_${tag}@${site}.log" 2>&1
    echo "Eval ${site} finished (exit $?): $(date)"
  done
}
evaluate 2 D3_cancer D4_female &
evaluate 3 D5_elderly A &
evaluate 4 D2_er D0_random D1_pediatric &
wait
echo "FedAvg-6 (-${exclude}) training and eval finished: $(date)"
