#!/usr/bin/env bash
# Prepare model inputs, pretrain GPT-2 (1 epoch, then fixed-LR continuation), evaluate frozen embeddings with logistic regression.
# Usage: GPU_IDS=0 bash scripts/train_model.sh configs/local_m0.toml  (GPU_IDS=0,1,2 runs torchrun; first GPU saves and embeds)
# Env: GPU_IDS (default: freest RTX 6000 Ada), GPU_WORK_SHARES (e.g. 1,1,0.5), EPOCHS=30, MIN_LR=2e-5, PATIENCE=3, MIN_DELTA=0.001, WARMUP_STEPS (default 5% of one epoch)
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
config="${1:?model config TOML required}"
min_lr="${MIN_LR:-2e-5}"
total_epochs="${EPOCHS:-30}"
patience="${PATIENCE:-3}"
min_delta="${MIN_DELTA:-0.001}"
python_bin="$project_root/.venv/bin/python"
export PYTHONPATH="$project_root/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
if [[ ! "$total_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer." >&2
  exit 2
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  gpu_index="$GPU_IDS"
else
  gpu_index="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits | awk -F, '$2 ~ /RTX 6000 Ada/ {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $3); if ($3 > free) {free=$3; gpu=$1}} END {print gpu}')"
fi
if [[ -z "$gpu_index" ]]; then
  echo "No RTX 6000 Ada GPU found." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu_index"
gpu_count="$(tr ',' '\n' <<< "$gpu_index" | wc -l)"
train_launcher=("$python_bin" -u -m)
if (( gpu_count > 1 )); then
  # Workaround: NCCL P2P hangs between PXB-linked GPUs, so go through host memory.
  export NCCL_P2P_DISABLE=1
  train_launcher=("$python_bin" -u -m torch.distributed.run --standalone --nproc_per_node="$gpu_count" -m)
fi
echo "config=$config GPU=$gpu_index epochs=$total_epochs"

PYTHON_BIN="$python_bin" bash scripts/prepare_joint.sh "$config"
"$python_bin" -u -m ausom_ehr.train plan --config "$config"
read -r output_root step_count < <("$python_bin" - "$config" <<'PYTHON'
import json
import sys
from ausom_ehr.common import load_config
root = load_config(sys.argv[1])["paths"]["output_root"]
plan = json.loads((root / "training" / "plan.json").read_text())
print(root, plan["train"]["optimizer_steps"])
PYTHON
)
warmup_steps="${WARMUP_STEPS:-$("$python_bin" -c "print(round($step_count * 0.05))")}"
echo "warmup steps: $warmup_steps (one epoch = $step_count steps)"
checkpoint="$output_root/training/checkpoints/step-$(printf '%05d' "$step_count")/state.pt"
if [[ ! -f "$checkpoint" ]]; then
  resume_args=()
  if compgen -G "$output_root/training/checkpoints/step-*/state.pt" > /dev/null; then
    resume_args=(--resume)
  fi
  "${train_launcher[@]}" ausom_ehr.train pretrain --config "$config" \
    --warmup-steps "$warmup_steps" --min-lr "$min_lr" "${resume_args[@]}"
fi
model_args=()
if (( total_epochs > 1 )); then
  "${train_launcher[@]}" ausom_ehr.train continue --config "$config" --max-epochs "$total_epochs" \
    --early-stopping-patience "$patience" --early-stopping-min-delta "$min_delta"
  model_args=(--initialization continued --run-name "epoch${total_epochs}")
fi
# Embedding runs on the first GPU only.
export CUDA_VISIBLE_DEVICES="${gpu_index%%,*}"
"$python_bin" -u -m ausom_ehr.evaluate embed --config "$config" "${model_args[@]}"
"$python_bin" -u -m ausom_ehr.evaluate logistic --config "$config" "${model_args[@]}"
echo "Training and evaluation done: $output_root"
