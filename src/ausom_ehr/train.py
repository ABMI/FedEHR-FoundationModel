"""EHR GPT-2 pretraining from scratch (1 epoch), then continued training at a fixed learning rate.

Stages: plan, pretrain, continue
Usage: python -m ausom_ehr.train <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedModel

from ausom_ehr.common import config_parent, load_config, output_file, write_json
from ausom_ehr.tokens import context_limit, special_ids


# ── Batching plan ──

MICROBATCH_PADDED_LIMIT = 16_384
MICROBATCH_PATIENT_LIMIT = 128
STEP_NONPAD_TARGET = 65_536
SHUFFLE_SEED = 42


def training_seed(config: dict) -> int:
    """Seed for initial weights and batch order: training.seed, default 42."""
    return int(config.get("training", {}).get("seed", SHUFFLE_SEED))


def make_microbatches(
    records: list[dict], shuffle: bool, padded_limit: int = MICROBATCH_PADDED_LIMIT, seed: int = SHUFFLE_SEED,
) -> list[list[dict]]:
    """Groups patients of similar length; a microbatch holds <= padded_limit padded tokens and <= 128 patients."""
    ordered = sorted(records, key=lambda row: (row["input_length"], row["person_id"]))
    batches = []
    current = []
    for row in ordered:
        length = int(row["input_length"])
        if length < 3 or length > padded_limit:
            raise ValueError(f"Invalid input length: {length}")
        next_size = len(current) + 1
        if current and (
            next_size * length > padded_limit
            or next_size > MICROBATCH_PATIENT_LIMIT
        ):
            batches.append(current)
            current = []
        current.append(row)
    if current:
        batches.append(current)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def summarize_batches(batches: list[list[dict]], accumulation: bool) -> dict:
    nonpad = 0
    padded = 0
    patients = 0
    optimizer_steps = 0
    step_tokens = 0
    step_token_min = None
    step_token_max = 0
    for batch in batches:
        batch_nonpad = sum(int(row["input_length"]) for row in batch)
        batch_padded = len(batch) * max(int(row["input_length"]) for row in batch)
        nonpad += batch_nonpad
        padded += batch_padded
        patients += len(batch)
        if accumulation:
            step_tokens += batch_nonpad
            if step_tokens >= STEP_NONPAD_TARGET:
                optimizer_steps += 1
                step_token_min = step_tokens if step_token_min is None else min(step_token_min, step_tokens)
                step_token_max = max(step_token_max, step_tokens)
                step_tokens = 0
    if accumulation and step_tokens:
        optimizer_steps += 1
        step_token_min = step_tokens if step_token_min is None else min(step_token_min, step_tokens)
        step_token_max = max(step_token_max, step_tokens)
    return {
        "patients": patients,
        "microbatches": len(batches),
        "nonpad_input_tokens": nonpad,
        "supervised_target_tokens": nonpad - patients,
        "padded_tokens": padded,
        "padding_fraction": (padded - nonpad) / padded if padded else None,
        "optimizer_steps": optimizer_steps if accumulation else None,
        "optimizer_step_min_nonpad": step_token_min,
        "optimizer_step_max_nonpad": step_token_max if accumulation else None,
    }


def run_plan(config_path: str) -> dict:
    config = load_config(config_path)
    sequence_glob = str(config["paths"]["output_root"] / "sequences" / "bucket=*.parquet")
    metadata = (
        pl.scan_parquet(sequence_glob)
        .filter(pl.col("input_length") > 0)
        .select("person_id", "split", "input_length")
        .collect()
    )
    train = metadata.filter(pl.col("split") == "train").to_dicts()
    validation = metadata.filter(pl.col("split") == "validation").to_dicts()
    padded_limit = context_limit(config)
    seed = training_seed(config)
    train_batches = make_microbatches(train, shuffle=True, padded_limit=padded_limit, seed=seed)
    validation_batches = make_microbatches(validation, shuffle=False, padded_limit=padded_limit)
    train_summary = summarize_batches(train_batches, accumulation=True)
    total_steps = train_summary["optimizer_steps"]
    warmup_candidates = sorted({
        max(1, round(total_steps * fraction))
        for fraction in (0.01, 0.03, 0.05)
        if max(1, round(total_steps * fraction)) < total_steps
    })
    report = {
        "gpu_count": 1,
        "microbatch_max_padded_tokens": padded_limit,
        "microbatch_max_patients": MICROBATCH_PATIENT_LIMIT,
        "optimizer_target_nonpad_input_tokens": STEP_NONPAD_TARGET,
        "shuffle_seed": seed,
        "train": train_summary,
        "validation": summarize_batches(validation_batches, accumulation=False),
        "epochs_planned": 1,
        "warmup_steps": None,
        "cosine_min_learning_rate": None,
        "warmup_step_candidates": warmup_candidates,
        "warmup_candidate_reason": "1%, 3%, 5% of total steps as candidates; not selected automatically",
        "cosine_min_learning_rate_candidates": [0.0, 1e-5, 2e-5],
        "cosine_candidate_reason": "minimum learning rate 0 or 5%, 10% of the peak; not selected automatically",
    }
    write_json(output_file(config, "training", "plan.json"), report)
    print(
        f"1 epoch: {report['train']['optimizer_steps']:,} optimizer steps, "
        f"{report['train']['microbatches']:,} microbatches, "
        f"PAD {report['train']['padding_fraction']:.1%}",
        flush=True,
    )
    return report


# ── Pretraining ──

BASE_LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.9, 0.95)
MAX_CONTEXT = 16_384


def load_split(config: dict, split: str, preload_tokens: bool = False) -> list[dict]:
    """preload_tokens=True attaches token slices of one int32 array; otherwise tokens are read per batch."""
    pattern = str(config["paths"]["output_root"] / "sequences" / "bucket=*.parquet")
    query = (
        pl.scan_parquet(pattern, include_file_paths="input_path")
        .filter((pl.col("split") == split) & (pl.col("input_length") > 0))
    )
    if not preload_tokens:
        return query.select("person_id", "input_length", "input_path").collect().to_dicts()
    data = query.select("person_id", "input_length", "input_path", "input_tokens").collect()
    return rows_with_token_arrays(data)


def rows_with_token_arrays(data: pl.DataFrame) -> list[dict]:
    """Packs input_tokens into one int32 array; each row gets a view of its slice."""
    flat = data["input_tokens"].explode(empty_as_null=True).to_numpy().astype(np.int32, copy=False)
    ends = np.cumsum(data["input_length"].to_numpy(), dtype=np.int64)
    if len(flat) != (int(ends[-1]) if len(ends) else 0):
        raise ValueError("Stored token count does not match the sum of input lengths.")
    rows = data.drop("input_tokens").to_dicts()
    start = 0
    for row, end in zip(rows, ends):
        row["input_tokens"] = flat[start:end]
        start = int(end)
    return rows


def read_batch_tokens(batch: list[dict]) -> list[dict]:
    """Reads tokens for this batch only, in the batch's row order."""
    if all("input_tokens" in row for row in batch):
        return batch
    # Task inputs hold several cutoffs of the same patient, so key by (person_id, task).
    key_columns = ["person_id", "task"] if all("task" in row for row in batch) else ["person_id"]
    patients_by_file = {}
    for row in batch:
        key = {name: row[name] for name in key_columns}
        patients_by_file.setdefault(row["input_path"], []).append(key)
    tokens_by_key = {}
    for filename, requested_keys in patients_by_file.items():
        requested = pl.DataFrame(requested_keys).unique()
        rows = (
            pl.scan_parquet(filename)
            .join(requested.lazy(), on=key_columns, how="inner")
            .select(*key_columns, "input_tokens")
            .collect()
            .to_dicts()
        )
        for row in rows:
            tokens_by_key[tuple(row[name] for name in key_columns)] = row["input_tokens"]
    loaded = []
    for row in batch:
        key = tuple(row[name] for name in key_columns)
        if key not in tokens_by_key:
            raise ValueError(f"Missing input for batch row: {key}")
        tokens = tokens_by_key[key]
        if len(tokens) != row["input_length"]:
            raise ValueError(f"input_length does not match stored token count: {key}")
        loaded.append({**row, "input_tokens": tokens})
    return loaded


def group_optimizer_steps(batches: list[list[dict]]) -> list[list[list[dict]]]:
    """Groups microbatches into optimizer steps of about 65,536 non-PAD input tokens."""
    groups = []
    current = []
    tokens = 0
    for batch in batches:
        current.append(batch)
        tokens += sum(row["input_length"] for row in batch)
        if tokens >= STEP_NONPAD_TARGET:
            groups.append(current)
            current = []
            tokens = 0
    if current:
        groups.append(current)
    return groups


def tensor_batch(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads to the longest row; returns (input_ids, attention_mask, labels with -100 at PAD) on device."""
    batch = read_batch_tokens(batch)
    width = max(row["input_length"] for row in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for index, row in enumerate(batch):
        length = row["input_length"]
        ids[index, :length] = torch.tensor(row["input_tokens"], dtype=torch.long)
        mask[index, :length] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    return ids.to(device), mask.to(device), labels.to(device)


def autocast_context():
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )


def setup_gpus() -> tuple[int, int, str]:
    """Returns (rank, world_size, device); one GPU per process under torchrun, else cuda:0."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        torch.cuda.set_device(0)
        return 0, 1, "cuda:0"
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    share = gpu_work_shares(world_size)[local_rank]
    if share < 1.0:
        torch.cuda.set_per_process_memory_fraction(share, device)
    dist.init_process_group("nccl", timeout=timedelta(hours=1), device_id=device)
    return dist.get_rank(), world_size, f"cuda:{local_rank}"


def gpu_work_shares(world_size: int) -> list[float]:
    """Per-GPU work fractions from GPU_WORK_SHARES (e.g. 1,1,0.5 in CUDA_VISIBLE_DEVICES order); default all 1."""
    text = os.environ.get("GPU_WORK_SHARES")
    if not text:
        return [1.0] * world_size
    shares = [float(value) for value in text.split(",")]
    if len(shares) != world_size or not all(0 < share <= 1 for share in shares):
        raise ValueError(f"GPU_WORK_SHARES needs one value in (0, 1] for each of {world_size} GPUs: {text}")
    return shares


def rank_and_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def wait_for_all_gpus() -> None:
    if rank_and_world()[1] > 1:
        dist.barrier()


def finish_gpus() -> None:
    if rank_and_world()[1] > 1:
        dist.destroy_process_group()


def microbatch_cost(batch: list[dict]) -> float:
    """Approximate compute for load balancing; width/8192 models attention (about 2x linear layers at 16,384)."""
    width = max(int(row["input_length"]) for row in batch)
    return len(batch) * width * (1.0 + width / 8_192)


def my_microbatches(microbatches: list[list[dict]]) -> list[list[dict]]:
    """This rank's share of a step; deterministic greedy split, so all ranks agree without communication."""
    rank, world_size = rank_and_world()
    if world_size == 1:
        return microbatches
    work_shares = gpu_work_shares(world_size)
    loads = [0.0] * world_size
    assigned = [[] for _ in range(world_size)]
    for batch in sorted(microbatches, key=microbatch_cost, reverse=True):
        cost = microbatch_cost(batch)
        finish_times = [(loads[gpu] + cost) / work_shares[gpu] for gpu in range(world_size)]
        target = finish_times.index(min(finish_times))
        assigned[target].append(batch)
        loads[target] += cost
    return assigned[rank]


def sum_gradients(model: torch.nn.Module) -> None:
    """Sums (not averages) gradients in one flat all_reduce; losses are already weighted by step-wide targets."""
    if rank_and_world()[1] == 1:
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    flat = torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter in parameters
    ])
    dist.all_reduce(flat)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = flat[offset:offset + count].view_as(parameter)
        offset += count


def sum_across_gpus(*values: float) -> list[float]:
    if rank_and_world()[1] == 1:
        return list(values)
    totals = torch.tensor(values, dtype=torch.float64, device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(totals)
    return totals.tolist()


MODEL_ARCHITECTURES = ("gpt2",)


def model_architecture(config: dict) -> str:
    architecture = config.get("model", {}).get("architecture", "gpt2")
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return architecture


def make_model(
    vocab_size: int, bos_id: int, eos_id: int, pad_id: int, max_context: int = MAX_CONTEXT,
    architecture: str = "gpt2",
) -> PreTrainedModel:
    """Random-init GPT-2: 12 layers, 768 dimensions, 12 heads, learned absolute positions."""
    if architecture != "gpt2":
        raise ValueError(f"Unknown model architecture: {architecture}")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_context,
        n_ctx=max_context,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        use_cache=False,
        attn_implementation="sdpa",
    )
    model = GPT2LMHeadModel(config)
    model.gradient_checkpointing_enable()
    return model


def create_model_optimizer(
    vocab_size: int, ids: dict[str, int], max_context: int, device: str, architecture: str = "gpt2",
) -> tuple[PreTrainedModel, torch.optim.AdamW]:
    model = make_model(
        vocab_size, ids["[BOS]"], ids["[EOS]"], ids["[PAD]"], max_context, architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY,
    )
    return model, optimizer


def preflight_memory(device: str) -> dict:
    """Peak GPU memory in bytes; None on CPU."""
    if not device.startswith("cuda"):
        return {"peak_gpu_allocated_bytes": None, "peak_gpu_reserved_bytes": None}
    return {
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def run_preflight(
    config: dict, train_rows: list[dict], vocab_path: Path, ids: dict[str, int],
    max_context: int, device: str, gpu_name: str,
) -> dict:
    """Memory check: two optimizer steps on the longest train input; writes preflight.json, saves no weights."""
    if not train_rows:
        raise ValueError("No train inputs for the preflight check.")
    longest = max(train_rows, key=lambda row: row["input_length"])
    vocab_size = pl.read_parquet(vocab_path).height
    with vocab_path.open("rb") as file:
        vocab_hash = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "status": "running",
        "gpu": gpu_name,
        "gpu_count_used": int(device.startswith("cuda")),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "bf16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "fp32",
        "max_context": max_context,
        "vocab_size": vocab_size,
        "vocab_sha256": vocab_hash,
        "input_length": int(longest["input_length"]),
        "microbatch_patients": 1,
        "input_provenance": {
            "source": "saved patient sequence from sequences/",
            "split": "train",
            "selection": "longest saved training input",
            "source_file": str(longest.get("input_path", "in-memory test fixture")),
            "synthetic_tokens_generated_by_preflight": False,
        },
        "optimizer": "AdamW",
        "learning_rate": BASE_LEARNING_RATE,
        "adam_betas": list(ADAM_BETAS),
        "weight_decay": WEIGHT_DECAY,
        "optimizer_step_completed": False,
        "optimizer_steps_planned": 2,
        "optimizer_steps_completed": 0,
        "optimizer_state_present_before_second_forward": False,
        "step_with_existing_optimizer_state_completed": False,
        "step_losses": [],
        "model_weights_saved": False,
        "memory_check_limitations": [
            "checks one microbatch only: the single longest train input",
            "does not guarantee memory for every planned microbatch or gradient accumulation",
            "validation input lengths and per-batch memory must be checked separately",
        ],
    }
    sample_summary = output_file(config, "sample", "summary.json")
    event_summary = output_file(config, "tokens", "event_summary.json")
    if sample_summary.exists():
        sample_report = json.loads(sample_summary.read_text(encoding="utf-8"))
        report["input_provenance"]["source_export_fingerprint"] = sample_report.get("source_fingerprint")
    if event_summary.exists():
        event_report = json.loads(event_summary.read_text(encoding="utf-8"))
        report["input_provenance"]["tokenization_mode"] = event_report.get("tokenization_mode", "original")
        report["input_provenance"]["event_summary_path"] = str(event_summary)
    destination = output_file(config, "training", "preflight.json")
    started = time.monotonic()
    phase = "model_and_optimizer"
    try:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            report["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        if not 2 <= longest["input_length"] <= max_context:
            raise ValueError("Longest input length must be between 2 and max_context.")
        model, optimizer = create_model_optimizer(
            vocab_size, ids, max_context, device, model_architecture(config),
        )
        report["architecture"] = model_architecture(config)
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        phase = "read_longest_input"
        input_ids, attention_mask, labels = tensor_batch([longest], ids["[PAD]"], device)
        if input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
            raise ValueError("Input token IDs are outside the current vocabulary.")
        # Step 2 measures activations together with the AdamW state allocated in step 1.
        for step_number in (1, 2):
            report["current_optimizer_step"] = step_number
            phase_suffix = "" if step_number == 1 else "_with_optimizer_state"
            phase = "zero_grad" + phase_suffix
            optimizer.zero_grad(set_to_none=True)
            if step_number == 2:
                report["optimizer_state_present_before_second_forward"] = bool(optimizer.state)
            phase = "forward" + phase_suffix
            with autocast_context():
                loss = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False,
                ).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Preflight step {step_number}: non-finite loss.")
            phase = "backward" + phase_suffix
            loss.backward()
            phase = "optimizer_step" + phase_suffix
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            report["optimizer_steps_completed"] = step_number
            report["optimizer_step_completed"] = True
            report["optimizer_state_bytes"] = sum(
                value.numel() * value.element_size()
                for state in optimizer.state.values() for value in state.values()
                if isinstance(value, torch.Tensor)
            )
            report["loss"] = float(loss.detach().item())
            report["step_losses"].append({"optimizer_step": step_number, "loss": report["loss"]})
            if step_number == 2:
                report["step_with_existing_optimizer_state_completed"] = True
            del loss
        report.update({
            "status": "passed",
            "nonpad_input_tokens": int(attention_mask.sum().item()),
            "supervised_tokens": int((labels[:, 1:] != -100).sum().item()),
        })
    except Exception as error:
        report.update({
            "status": "failed",
            "failed_phase": phase,
            "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
            **preflight_memory(device),
        })
        write_json(destination, report)
        raise
    report.update({"elapsed_seconds": time.monotonic() - started, **preflight_memory(device)})
    write_json(destination, report)
    print(f"Preflight passed: input {report['input_length']:,} tokens, vocab {vocab_size:,}", flush=True)
    return report


def learning_rate_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr: float) -> float:
    """Multiplier of BASE_LEARNING_RATE: linear warmup, then cosine decay to min_lr."""
    floor = min_lr / BASE_LEARNING_RATE
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * progress)) / 2.0


def evaluate(
    model: PreTrainedModel,
    batches: list[list[dict]],
    pad_id: int,
    device: str,
) -> float:
    """Mean next-token cross entropy over all validation targets, summed across GPUs."""
    model.eval()
    total_loss = 0.0
    target_tokens = 0
    with torch.no_grad(), autocast_context():
        for batch in my_microbatches(batches):
            ids, mask, labels = tensor_batch(batch, pad_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
            count = sum(row["input_length"] - 1 for row in batch)
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite validation loss.")
            total_loss += value * count
            target_tokens += count
    total_loss, target_tokens = sum_across_gpus(total_loss, target_tokens)
    if target_tokens == 0:
        raise ValueError("No next-token targets for validation.")
    model.train()
    return total_loss / target_tokens


def save_checkpoint(
    config: dict,
    step: int,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    best_loss: float | None,
    plan: dict,
    kind: str,
) -> Path:
    """Atomically saves resume state together with vocab, decile boundaries and config."""
    root = output_file(config, "training", "checkpoints", "placeholder").parent
    target = root / (f"step-{step:05d}" if kind == "resume" else "best")
    temporary = root / (target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_validation_loss": best_loss,
        "plan": plan,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    torch.save(state, temporary / "state.pt")
    shutil.copy2(output_file(config, "tokens", "vocab.parquet"), temporary / "vocab.parquet")
    shutil.copy2(output_file(config, "deciles", "boundaries.parquet"), temporary / "boundaries.parquet")
    shutil.copy2(config["config_path"], temporary / "config.toml")
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    return target


def load_latest_checkpoint(config: dict) -> Path | None:
    root = config["paths"]["output_root"] / "training" / "checkpoints"
    files = sorted(root.glob("step-*/state.pt")) if root.exists() else []
    return files[-1].parent if files else None


def check_checkpoint_inputs(checkpoint: Path, config: dict) -> None:
    """Raises if vocab, boundaries or config changed since the checkpoint was saved."""
    inputs = {
        "vocab.parquet": output_file(config, "tokens", "vocab.parquet"),
        "boundaries.parquet": output_file(config, "deciles", "boundaries.parquet"),
        "config.toml": Path(config["config_path"]),
    }
    for name, current in inputs.items():
        saved = checkpoint / name
        if not saved.is_file() or not current.is_file():
            raise FileNotFoundError(f"Missing resume input: {name}")
        with saved.open("rb") as file:
            saved_hash = hashlib.file_digest(file, "sha256").digest()
        with current.open("rb") as file:
            current_hash = hashlib.file_digest(file, "sha256").digest()
        if saved_hash != current_hash:
            raise ValueError(f"Input changed since the checkpoint was saved: {name}")


def restore_training_state(state: dict, model, optimizer, scheduler) -> None:
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    # set_rng_state needs a CPU tensor even if map_location moved it.
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())


def run_pretrain(
    config_path: str, warmup_steps: int | None, min_lr: float | None,
    resume: bool, check_only: bool = False,
) -> None:
    if check_only and resume:
        raise ValueError("--check-only and --resume cannot be used together.")
    config = load_config(config_path)
    plan_path = output_file(config, "training", "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    total_steps = int(plan["train"]["optimizer_steps"])
    effective_plan = {**plan, "warmup_steps": warmup_steps, "cosine_min_learning_rate": min_lr}
    if not check_only and (warmup_steps is None or not 0 < warmup_steps < total_steps):
        raise ValueError("warmup steps must be > 0 and < total steps.")
    if not check_only and (min_lr is None or not 0 <= min_lr < BASE_LEARNING_RATE):
        raise ValueError("min learning rate must be >= 0 and < the base learning rate.")
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required.")
    rank, world_size, device = setup_gpus()
    if check_only and world_size > 1:
        raise ValueError("Run --check-only on a single GPU.")
    gpu_name = torch.cuda.get_device_name(device)
    if "RTX 6000 Ada" not in gpu_name:
        raise RuntimeError(f"Expected an RTX 6000 Ada GPU, got: {gpu_name}")
    is_main = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    # Same seed on every rank gives identical initial weights; summed gradients keep them identical.
    seed = training_seed(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    vocab = output_file(config, "tokens", "vocab.parquet")
    ids = special_ids(vocab)
    vocab_size = pl.read_parquet(vocab).height
    sequence_summary = output_file(config, "sequences", "summary.json")
    if sequence_summary.exists():
        summary = json.loads(sequence_summary.read_text(encoding="utf-8"))
        if summary.get("ready_for_training") is False:
            raise ValueError("Sequence audit has unresolved exceptions; see sequences/summary.json.")
    if is_main:
        print("Reading train patient IDs, lengths and file paths", flush=True)
    train_rows = load_split(config, "train", preload_tokens=not check_only)
    max_context = context_limit(config)
    if check_only:
        run_preflight(config, train_rows, vocab, ids, max_context, device, gpu_name)
        return
    validation_rows = load_split(config, "validation", preload_tokens=True)
    train_batches = make_microbatches(train_rows, shuffle=True, padded_limit=max_context, seed=seed)
    validation_batches = make_microbatches(validation_rows, shuffle=False, padded_limit=max_context)
    if summarize_batches(train_batches, True)["optimizer_steps"] != total_steps:
        raise ValueError("Step count of the current data differs from training/plan.json.")
    step_groups = group_optimizer_steps(train_batches)
    if is_main:
        print(f"GPU {gpu_name} x{world_size}; vocab {vocab_size}; train steps {len(step_groups)}", flush=True)

    model, optimizer = create_model_optimizer(vocab_size, ids, max_context, device, model_architecture(config))
    parameter_count = sum(param.numel() for param in model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(step, total_steps, warmup_steps, min_lr),
    )

    start_step = 0
    best_loss = None
    if resume:
        checkpoint = load_latest_checkpoint(config)
        if checkpoint is None:
            raise FileNotFoundError("No checkpoint to resume from.")
        check_checkpoint_inputs(checkpoint, config)
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        saved_plan = state["plan"]
        if saved_plan["train"] != plan["train"]:
            raise ValueError("Checkpoint training plan differs from the current inputs.")
        if (saved_plan.get("warmup_steps") != warmup_steps or
                saved_plan.get("cosine_min_learning_rate") != min_lr):
            raise ValueError("Resume needs the same warmup steps and min learning rate as the original run.")
        restore_training_state(state, model, optimizer, scheduler)
        start_step = int(state["step"])
        best_loss = state["best_validation_loss"]
        del state
        if is_main:
            print(f"Resuming from checkpoint step {start_step}", flush=True)

    validation_steps = {math.ceil(total_steps * fraction / 4) for fraction in range(1, 5)}
    checkpoint_steps = {math.ceil(total_steps * fraction / 10) for fraction in range(1, 11)}
    metrics_path = output_file(config, "training", "metrics.json")
    if metrics_path.exists() and resume:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Steps after the checkpoint are rerun, so drop their records.
        metrics["validation"] = [
            row for row in metrics.get("validation", []) if row["step"] <= start_step
        ]
        metrics["checkpoints"] = [
            row for row in metrics.get("checkpoints", []) if row["step"] <= start_step
        ]
        metrics["completed_steps"] = start_step
        metrics["best_validation_loss"] = best_loss
        if best_loss is None:
            metrics.pop("best_step", None)
    elif metrics_path.exists():
        raise FileExistsError("Training metrics already exist; use --resume.")
    else:
        metrics = {
            "gpu": gpu_name,
            "gpu_count_used": world_size,
            "gradient_sync": (
                "single GPU" if world_size == 1
                else "step microbatches split across GPUs; one summed fp32 all_reduce per optimizer step"
            ),
            "gpu_work_shares": gpu_work_shares(world_size),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp32",
            "parameter_count": parameter_count,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "min_learning_rate": min_lr,
            "base_learning_rate": BASE_LEARNING_RATE,
            "validation": [],
            "checkpoints": [],
        }
    # rank 0 writes metrics only after every rank has read them.
    wait_for_all_gpus()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    cumulative_nonpad = 0
    cumulative_targets = 0
    cumulative_padded = 0
    for index in range(start_step, total_steps):
        microbatches = step_groups[index]
        group_targets = sum(
            row["input_length"] - 1 for batch in microbatches for row in batch
        )
        group_nonpad = sum(
            row["input_length"] for batch in microbatches for row in batch
        )
        group_padded = sum(
            len(batch) * max(row["input_length"] for row in batch)
            for batch in microbatches
        )
        weighted_loss = 0.0
        # Loss weights use the whole step's target count, not this rank's share.
        for batch in my_microbatches(microbatches):
            ids_tensor, mask, labels = tensor_batch(batch, ids["[PAD]"], device)
            batch_targets = sum(row["input_length"] - 1 for row in batch)
            with autocast_context():
                loss = model(
                    input_ids=ids_tensor, attention_mask=mask, labels=labels, use_cache=False
                ).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"step {index + 1}: nonfinite loss")
            weighted_loss += float(loss.detach().item()) * batch_targets / group_targets
            (loss * (batch_targets / group_targets)).backward()
        sum_gradients(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step = index + 1
        cumulative_nonpad += group_nonpad
        cumulative_targets += group_targets
        cumulative_padded += group_padded
        if step % 25 == 0 or step == total_steps:
            [weighted_loss] = sum_across_gpus(weighted_loss)
            elapsed = time.monotonic() - started
            if is_main:
                print(
                    f"step {step}/{total_steps} train_loss={weighted_loss:.4f} "
                    f"nonpad={group_nonpad:,} elapsed={elapsed/3600:.2f}h",
                    flush=True,
                )
        if step in validation_steps:
            val_loss = evaluate(model, validation_batches, ids["[PAD]"], device)
            if is_main:
                print(f"step {step}: validation_loss={val_loss:.5f}", flush=True)
            metrics["validation"].append({"step": step, "loss": val_loss})
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                if is_main:
                    save_checkpoint(config, step, model, optimizer, scheduler, best_loss, effective_plan, "best")
                metrics["best_step"] = step
                metrics["best_validation_loss"] = best_loss
        if step in checkpoint_steps:
            if is_main:
                checkpoint = save_checkpoint(config, step, model, optimizer, scheduler, best_loss, effective_plan, "resume")
                metrics["checkpoints"].append({"step": step, "path": str(checkpoint)})
        if (step in validation_steps or step in checkpoint_steps) and is_main:
            elapsed = time.monotonic() - started
            metrics["completed_steps"] = step
            metrics["elapsed_seconds_this_process"] = elapsed
            metrics["nonpad_input_tokens_this_process"] = cumulative_nonpad
            metrics["supervised_tokens_this_process"] = cumulative_targets
            metrics["padded_tokens_this_process"] = cumulative_padded
            metrics["train_input_tokens_per_second"] = cumulative_nonpad / elapsed
            metrics["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metrics["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_json(metrics_path, metrics)
        if step in validation_steps or step in checkpoint_steps:
            wait_for_all_gpus()
    if is_main:
        print(f"1 epoch done; best validation loss {best_loss}", flush=True)
    finish_gpus()


# ── Continued training ──

CONTINUED_LR = 2e-5
FIRST_EPOCH = 1


def first_checkpoint(config: dict):
    """Returns (state.pt of the last first-epoch checkpoint, its step count)."""
    plan = json.loads(output_file(config, "training", "plan.json").read_text(encoding="utf-8"))
    steps = int(plan["train"]["optimizer_steps"])
    return output_file(config, "training", "checkpoints", f"step-{steps:05d}", "state.pt"), steps


def save_epoch(config: dict, epoch: int, model, optimizer, metrics: dict) -> None:
    target = output_file(config, "training", "continued", f"epoch-{epoch:02d}", "state.pt")
    temporary = target.with_name("state.pt.tmp")
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }
    torch.save(state, temporary)
    temporary.replace(target)


def copy_best(config: dict, epoch: int) -> None:
    """Copies the best epoch's state with vocab, boundaries and config to continued/best/."""
    if epoch == FIRST_EPOCH:
        # Epoch 1's best weights can differ from its last weights.
        source = output_file(config, "training", "checkpoints", "best", "state.pt")
    else:
        source = output_file(config, "training", "continued", f"epoch-{epoch:02d}", "state.pt")
    destination = output_file(config, "training", "continued", "best", "state.pt")
    temporary = destination.with_name("state.pt.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    for filename, source_file in [
        ("vocab.parquet", output_file(config, "tokens", "vocab.parquet")),
        ("boundaries.parquet", output_file(config, "deciles", "boundaries.parquet")),
        ("config.toml", config["config_path"]),
    ]:
        shutil.copy2(source_file, destination.parent / filename)


def restore_rng(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())


def train_epoch(
    model, optimizer, batches: list[list[dict]], pad_id: int, epoch: int,
    device: str = "cuda:0", is_main: bool = True, seed: int = 42, scheduler=None, shuffle: bool = True,
) -> tuple[float, int]:
    """Returns (target-weighted train loss over all GPUs, optimizer steps); scheduler steps per optimizer step."""
    shuffled = list(batches)
    if shuffle:
        random.Random(seed + epoch).shuffle(shuffled)
    step_groups = group_optimizer_steps(shuffled)
    weighted_total = 0.0
    target_total = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, microbatches in enumerate(step_groups, start=1):
        group_targets = sum(
            row["input_length"] - 1 for batch in microbatches for row in batch
        )
        for batch in my_microbatches(microbatches):
            input_ids, mask, labels = tensor_batch(batch, pad_id, device)
            batch_targets = sum(row["input_length"] - 1 for row in batch)
            with autocast_context():
                loss = model(
                    input_ids=input_ids, attention_mask=mask, labels=labels,
                    use_cache=False,
                ).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch}, step {step}: non-finite loss")
            weighted_total += float(loss.detach().item()) * batch_targets
            target_total += batch_targets
            (loss * (batch_targets / group_targets)).backward()
        sum_gradients(model)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if is_main and (step % 100 == 0 or step == len(step_groups)):
            print(f"epoch {epoch}: step {step}/{len(step_groups)}", flush=True)
    weighted_total, target_total = sum_across_gpus(weighted_total, target_total)
    return weighted_total / target_total, len(step_groups)


def extend_epoch_goal(metrics: dict, max_epochs: int) -> tuple[dict, bool]:
    """Raises the epoch goal only after the current goal finished; returns (updated copy, extended)."""
    previous_goal = metrics["max_epochs"]
    if max_epochs < previous_goal:
        raise ValueError(f"Cannot lower the epoch goal below {previous_goal} epochs.")
    if max_epochs == previous_goal:
        return metrics, False
    if not metrics.get("stopped") or metrics["completed_epochs"] != previous_goal:
        raise ValueError("Finish the current goal before raising max epochs.")
    updated = dict(metrics)
    updated["max_epochs"] = max_epochs
    updated["stopped"] = False
    return updated, True


def update_early_stopping(metrics: dict, validation_loss: float) -> bool:
    """Counts epochs without a min_delta gain over the reference loss; True once patience is reached."""
    patience = metrics.get("early_stopping_patience", 0)
    if patience == 0:
        return False
    reference = metrics["early_stopping_reference_loss"]
    min_delta = metrics["early_stopping_min_delta"]
    if reference - validation_loss >= min_delta:
        # The reference moves only here, so small gains accumulate until they reach min_delta.
        metrics["early_stopping_reference_loss"] = validation_loss
        metrics["epochs_without_significant_improvement"] = 0
    else:
        metrics["epochs_without_significant_improvement"] += 1
    return metrics["epochs_without_significant_improvement"] >= patience


def run_continue(config_path: str, max_epochs: int, patience: int = 0, min_delta: float = 0.0) -> None:
    """Trains from the saved state up to max_epochs in total (epoch 1 is the pretrain run)."""
    if max_epochs < 2:
        raise ValueError("max_epochs must be at least 2.")
    if patience < 0 or min_delta < 0 or not math.isfinite(min_delta):
        raise ValueError("patience and min_delta must be finite and >= 0.")
    if patience > 0 and min_delta == 0:
        raise ValueError("Early stopping needs min_delta > 0.")
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required.")
    rank, world_size, device = setup_gpus()
    if "RTX 6000 Ada" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Select an RTX 6000 Ada GPU.")
    is_main = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    vocab_path = output_file(config, "tokens", "vocab.parquet")
    ids = special_ids(vocab_path)
    vocab_size = pl.read_parquet(vocab_path).height
    train_rows = load_split(config, "train", preload_tokens=True)
    validation_rows = load_split(config, "validation", preload_tokens=True)
    max_context = context_limit(config)
    train_batches = make_microbatches(train_rows, shuffle=False, padded_limit=max_context)
    validation_batches = make_microbatches(validation_rows, shuffle=False, padded_limit=max_context)
    if is_main:
        print(f"train {len(train_rows):,}, validation {len(validation_rows):,} patients; {world_size} GPUs", flush=True)

    first_path, first_steps = first_checkpoint(config)
    metrics_path = output_file(config, "training", "continued", "metrics.json")
    fresh_run = not metrics_path.exists()
    if metrics_path.exists():
        previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics, extending = extend_epoch_goal(previous_metrics, max_epochs)
        if metrics.get("stopped"):
            if is_main:
                print("Requested epochs are already complete.", flush=True)
            finish_gpus()
            return
        last_epoch = metrics["completed_epochs"]
        # Epoch 1 is in training/checkpoints; later epochs are in training/continued.
        if last_epoch == FIRST_EPOCH:
            source = first_checkpoint(config)[0]
        else:
            source = output_file(config, "training", "continued", f"epoch-{last_epoch:02d}", "state.pt")
    else:
        extending = False
        last_epoch = FIRST_EPOCH
        source = first_checkpoint(config)[0]
        metrics = {
            "source_checkpoint": str(source),
            "initial_epoch_validation_loss": None,
            "max_epochs": max_epochs,
            "continuation_learning_rate": CONTINUED_LR,
            "learning_rate_rule": "constant at the first epoch cosine endpoint",
            "gpu_count": world_size,
            "completed_epochs": FIRST_EPOCH,
            "best_epoch": FIRST_EPOCH,
            "best_validation_loss": None,
            "epochs": [],
            "stopped": False,
        }
    # Drop keys from the older early-stopping format.
    metrics.pop("patience", None)
    metrics.pop("epochs_without_improvement", None)
    if extending:
        metrics.pop("stopping_reason", None)
        metrics["early_stopping_patience"] = patience
        metrics["early_stopping_min_delta"] = min_delta
        metrics["early_stopping_reference_loss"] = metrics["best_validation_loss"]
        metrics["epochs_without_significant_improvement"] = 0
    elif "early_stopping_patience" in metrics:
        if (metrics["early_stopping_patience"] != patience
                or metrics["early_stopping_min_delta"] != min_delta):
            raise ValueError("Early stopping settings cannot change on restart.")
    elif patience > 0:
        if not fresh_run:
            raise ValueError("Cannot turn on early stopping in the middle of a run.")
        metrics["early_stopping_patience"] = patience
        metrics["early_stopping_min_delta"] = min_delta
        metrics["epochs_without_significant_improvement"] = 0
    metrics["stopping_rule"] = (
        f"max {max_epochs} epochs; patience {patience}; min_delta {min_delta}"
        if patience else "finish all epochs; select lowest validation loss"
    )
    if not source.exists():
        raise FileNotFoundError(f"Restart checkpoint not found: {source}")
    # rank 0 writes metrics only after every rank has read them.
    wait_for_all_gpus()
    if extending and is_main:
        archived_metrics = output_file(
            config, "training", "continued", f"epoch-{last_epoch:02d}",
            "metrics_at_completion.json",
        )
        if not archived_metrics.exists():
            write_json(archived_metrics, previous_metrics)
        print(f"Goal extended to {max_epochs} epochs; kept the {last_epoch}-epoch record", flush=True)
    state = torch.load(source, map_location="cpu", weights_only=False)
    if last_epoch == FIRST_EPOCH:
        if state["step"] != first_steps:
            raise ValueError("Step count of the 1-epoch checkpoint does not match the plan.")
        first_loss = float(state["best_validation_loss"])
        saved_lr = state["optimizer"]["param_groups"][0]["lr"]
        if saved_lr != CONTINUED_LR:
            raise ValueError(f"Unexpected learning rate at the end of epoch 1: {saved_lr}")
        metrics["initial_epoch_validation_loss"] = first_loss
        metrics["best_validation_loss"] = first_loss
        if fresh_run and patience > 0:
            metrics["early_stopping_reference_loss"] = first_loss
    model = make_model(
        vocab_size, ids["[BOS]"], ids["[EOS]"], ids["[PAD]"], max_context, model_architecture(config),
    ).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONTINUED_LR,
        betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY,
    )
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = CONTINUED_LR
    restore_rng(state)
    del state
    best_path = output_file(config, "training", "continued", "best", "state.pt")
    if is_main:
        if not best_path.exists():
            copy_best(config, metrics["best_epoch"])
        write_json(metrics_path, metrics)
        print(f"Resuming at epoch {last_epoch}; best val loss={metrics['best_validation_loss']:.5f}", flush=True)

    for epoch in range(last_epoch + 1, max_epochs + 1):
        started = time.monotonic()
        train_loss, steps = train_epoch(
            model, optimizer, train_batches, ids["[PAD]"], epoch, device, is_main, training_seed(config),
        )
        val_loss = evaluate(model, validation_batches, ids["[PAD]"], device)
        if not math.isfinite(val_loss):
            raise FloatingPointError(f"epoch {epoch}: non-finite validation loss")
        improved = val_loss < metrics["best_validation_loss"]
        if improved:
            metrics["best_epoch"] = epoch
            metrics["best_validation_loss"] = val_loss
        metrics["completed_epochs"] = epoch
        metrics["epochs"].append({
            "epoch": epoch,
            "optimizer_steps": steps,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "elapsed_seconds": time.monotonic() - started,
        })
        early_stop = update_early_stopping(metrics, val_loss)
        metrics["stopped"] = early_stop or epoch == max_epochs
        if metrics["stopped"]:
            metrics["stopping_reason"] = "max_epochs" if epoch == max_epochs else "early_stopping"
        if is_main:
            print(
                f"epoch {epoch}: train_loss={train_loss:.5f}, validation_loss={val_loss:.5f}, "
                f"best_epoch={metrics['best_epoch']}",
                flush=True,
            )
            save_epoch(config, epoch, model, optimizer, metrics)
            if improved:
                copy_best(config, epoch)
            write_json(metrics_path, metrics)
        wait_for_all_gpus()
        if metrics["stopped"]:
            if is_main:
                print(
                    f"Training stopped: epoch={epoch}, max={max_epochs}, "
                    f"reason={metrics['stopping_reason']}, best_epoch={metrics['best_epoch']}",
                    flush=True,
                )
            break
    finish_gpus()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    stages.add_parser("plan", parents=[common_args], help="microbatches and optimizer steps for 1 epoch")
    pretrain = stages.add_parser("pretrain", parents=[common_args], help="train 1 epoch from scratch")
    pretrain.add_argument("--warmup-steps", type=int)
    pretrain.add_argument("--min-lr", type=float)
    pretrain.add_argument("--resume", action="store_true")
    pretrain.add_argument("--check-only", action="store_true", help="memory check on the longest train input only")
    extend = stages.add_parser("continue", parents=[common_args], help="continue from 1 epoch at a fixed 2e-5")
    extend.add_argument("--max-epochs", type=int, required=True)
    extend.add_argument("--early-stopping-patience", type=int, default=0)
    extend.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    args = parser.parse_args()
    if args.stage == "plan":
        run_plan(args.config)
    elif args.stage == "pretrain":
        if args.check_only and args.resume:
            pretrain.error("--check-only and --resume cannot be used together.")
        if not args.check_only and (args.warmup_steps is None or args.min_lr is None):
            pretrain.error("Training needs both --warmup-steps and --min-lr.")
        run_pretrain(args.config, args.warmup_steps, args.min_lr, args.resume, check_only=args.check_only)
    else:
        run_continue(args.config, args.max_epochs, args.early_stopping_patience, args.early_stopping_min_delta)


if __name__ == "__main__":
    main()
