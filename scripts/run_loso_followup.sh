#!/usr/bin/env bash
# Wait for run_fedavg_loso.sh, then evaluate FedAvg-6 (minus SITE) on the 12 EHRSHOT tasks at all 7 sites and pair it with FedAvg-7.
# Usage: setsid nohup bash scripts/run_loso_followup.sh D1_pediatric > outputs/sites_disjoint/logs/followup_fedavg6_minus_D1_pediatric.log 2>&1 &
# Env: GPU_MEMORY_FRACTIONS (e.g. 4:0.5; default none)
set -uo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
py="$project_root/.venv/bin/python"
exclude="${1:?excluded site required}"
root=outputs/sites_disjoint
sites="A D0_random D1_pediatric D2_er D3_cancer D4_female D5_elderly"

echo "$(date) waiting for run_fedavg_loso.sh $exclude to finish"
while pgrep -f "^bash [^ ]*run_fedavg_loso\.sh $exclude" > /dev/null; do sleep 600; done
missing=0
for site in $sites; do
  f="$root/eval/fedavg6_minus_${exclude}_round20@${site}/evaluation/continued/epoch20/results.json"
  [[ -f $f ]] || { echo "Missing 3-task eval: $f"; missing=1; }
done
(( missing == 0 )) || { echo "Training or 3-task eval not finished; stopping."; exit 1; }

echo "$(date) EHRSHOT 12-task eval started (GPU memory limit: ${GPU_MEMORY_FRACTIONS:-none})"
"$py" -u scripts/run_ehrshot_eval.py --loso "$exclude" || { echo "EHRSHOT eval failed"; exit 1; }
for site in $sites; do
  f="$root/eval_ehrshot/fedavg6_minus_${exclude}_round20@${site}/evaluation/continued/epoch20/test_predictions.parquet"
  [[ -f $f ]] || { echo "Missing EHRSHOT result: $f"; exit 1; }
done

echo "$(date) paired comparison started"
"$py" -u scripts/paired_ehrshot15.py --fl6 "$exclude" || { echo "Paired comparison failed"; exit 1; }
echo "$(date) all done: $root/analysis/paired_fl6_minus_${exclude}.json"
