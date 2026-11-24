"""Training masks, causality, resume and epoch-goal rules on synthetic tokens and tiny CPU models."""

import copy
import json
import random

import numpy as np
import polars as pl
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ausom_ehr import train
from ausom_ehr.common import result_folder_parts
from ausom_ehr.train import (
    check_checkpoint_inputs,
    extend_epoch_goal,
    load_split,
    restore_training_state,
    tensor_batch,
    update_early_stopping,
)
from ausom_ehr.train import run_plan as make_training_plan


def test_training_reads_only_batch_tokens_and_masks_padding(tmp_path):
    folder = tmp_path / "sequences"
    folder.mkdir()
    pl.DataFrame({
        "person_id": [1, 2, 3],
        "split": ["train", "validation", "train"],
        "input_length": [5, 3, 3],
        "input_tokens": [[1, 3, 4, 5, 2], [1, 6, 2], [1, 7, 2]],
    }).write_parquet(folder / "bucket=000.parquet")
    metadata = load_split({"paths": {"output_root": tmp_path}}, "train")
    assert [row["person_id"] for row in metadata] == [1, 3]
    assert all("input_tokens" not in row for row in metadata)
    ids, mask, labels = tensor_batch(metadata, pad_id=0, device="cpu")
    assert ids.tolist() == [[1, 3, 4, 5, 2], [1, 7, 2, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert labels.tolist() == [[1, 3, 4, 5, 2], [1, 7, 2, -100, -100]]


def test_causal_prediction_and_shifted_loss_ignore_padding():
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=12, n_positions=8, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, use_cache=False,
    )).eval()
    first = torch.tensor([[1, 3, 4, 5]])
    changed_future = torch.tensor([[1, 3, 4, 7]])
    with torch.no_grad():
        original_logits = model(first).logits
        changed_logits = model(changed_future).logits
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3])

    batch = [
        {"input_length": 5, "input_tokens": [1, 3, 4, 5, 2]},
        {"input_length": 3, "input_tokens": [1, 6, 2]},
    ]
    ids, mask, labels = tensor_batch(batch, pad_id=0, device="cpu")
    with torch.no_grad():
        result = model(input_ids=ids, attention_mask=mask, labels=labels)
    expected_loss = torch.nn.functional.cross_entropy(
        result.logits[:, :-1].reshape(-1, 12),
        labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    torch.testing.assert_close(result.loss, expected_loss)


def test_batch_reader_uses_correct_task_cutoff_for_same_patient(tmp_path):
    filename = tmp_path / "cohort.parquet"
    pl.DataFrame({
        "person_id": [1, 1, 1],
        "task": ["los7", "readmit30", "outpatient90"],
        "input_tokens": [[1, 3, 4], [1, 3, 4, 5], [1, 3, 9]],
    }).write_parquet(filename)
    batch = [{
        "person_id": 1, "task": "los7", "input_length": 3,
        "input_path": str(filename),
    }]
    ids, mask, _labels = tensor_batch(batch, pad_id=0, device="cpu")
    assert ids.tolist() == [[1, 3, 4]]
    assert mask.tolist() == [[1, 1, 1]]


def test_saved_optimizer_scheduler_and_rng_resume_same_next_step(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    def make_training_parts():
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1 / (step + 1),
        )
        return model, optimizer, scheduler

    def take_step(model, optimizer, scheduler):
        values = torch.randn(2, 3) + random.random() + float(np.random.random())
        optimizer.zero_grad(set_to_none=True)
        model(values).square().mean().backward()
        optimizer.step()
        scheduler.step()

    model, optimizer, scheduler = make_training_parts()
    take_step(model, optimizer, scheduler)
    state = copy.deepcopy({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
    })
    checkpoint = tmp_path / "state.pt"
    torch.save(state, checkpoint)
    take_step(model, optimizer, scheduler)
    expected = copy.deepcopy(model.state_dict())

    resumed_model, resumed_optimizer, resumed_scheduler = make_training_parts()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restore_training_state(saved, resumed_model, resumed_optimizer, resumed_scheduler)
    take_step(resumed_model, resumed_optimizer, resumed_scheduler)
    for name, actual in resumed_model.state_dict().items():
        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()


def test_checkpoint_rejects_changed_token_definitions(tmp_path):
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for folder, filename in [("tokens", "vocab.parquet"), ("deciles", "boundaries.parquet")]:
        target = output / folder / filename
        target.parent.mkdir(parents=True)
        target.write_bytes(b"same-definition")
        (checkpoint / filename).write_bytes(b"same-definition")
    config_file = tmp_path / "settings.toml"
    config_file.write_text("# synthetic settings\n", encoding="utf-8")
    (checkpoint / "config.toml").write_bytes(config_file.read_bytes())
    config = {"paths": {"output_root": output}, "config_path": str(config_file)}
    check_checkpoint_inputs(checkpoint, config)
    (output / "tokens" / "vocab.parquet").write_bytes(b"changed-ids")
    with pytest.raises(ValueError, match="vocab.parquet"):
        check_checkpoint_inputs(checkpoint, config)


def test_training_plan_offers_candidates_without_selecting_policy(tmp_path, monkeypatch):
    folder = tmp_path / "sequences"
    folder.mkdir()
    pl.DataFrame({
        "person_id": [1, 2, 3, 4, 5, 6],
        "split": ["train"] * 5 + ["validation"],
        "input_length": [16_384] * 6,
    }).write_parquet(folder / "bucket=000.parquet")
    config = {"paths": {"output_root": tmp_path}}
    monkeypatch.setattr("ausom_ehr.train.load_config", lambda _path: config)
    plan = make_training_plan("synthetic.toml")
    assert plan["train"]["optimizer_steps"] == 2
    assert plan["warmup_steps"] is None
    assert plan["cosine_min_learning_rate"] is None
    assert plan["warmup_step_candidates"] == [1]


@pytest.mark.parametrize("oom_step", [None, 1, 2])
def test_preflight_checks_longest_input_and_optimizer_without_training(tmp_path, monkeypatch, oom_step):
    def tiny_model(vocab_size, bos_id, eos_id, pad_id, max_context, architecture="gpt2"):
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=vocab_size, n_positions=max_context, n_ctx=max_context,
            n_embd=16, n_layer=1, n_head=2, bos_token_id=bos_id,
            eos_token_id=eos_id, pad_token_id=pad_id, use_cache=False,
        ))
        model.gradient_checkpointing_enable()
        return model

    vocabulary = tmp_path / "vocab.parquet"
    pl.DataFrame({"token_id": list(range(8))}).write_parquet(vocabulary)
    filename = tmp_path / "synthetic-sequences.parquet"
    pl.DataFrame({
        "person_id": [1, 2],
        "input_tokens": [[1, 4, 5, 2], [1, 4, 5, 6, 7, 2]],
    }).write_parquet(filename)
    rows = [
        {"person_id": 1, "input_length": 4, "input_path": str(filename)},
        {"person_id": 2, "input_length": 6, "input_path": str(filename)},
    ]
    config = {"paths": {"output_root": tmp_path}}
    ids = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
    monkeypatch.setattr(train, "make_model", tiny_model)
    original_create = train.create_model_optimizer
    states_seen_at_forward = []

    def create_checked_parts(*args):
        model, optimizer = original_create(*args)
        original_step = optimizer.step
        original_forward = model.forward
        steps_attempted = 0

        def checked_step(*step_args, **step_keywords):
            nonlocal steps_attempted
            steps_attempted += 1
            if steps_attempted == oom_step:
                raise torch.cuda.OutOfMemoryError("synthetic optimizer allocation failure")
            return original_step(*step_args, **step_keywords)

        def checked_forward(*forward_args, **forward_keywords):
            states_seen_at_forward.append(bool(optimizer.state))
            return original_forward(*forward_args, **forward_keywords)

        optimizer.step = checked_step
        model.forward = checked_forward
        return model, optimizer

    monkeypatch.setattr(train, "create_model_optimizer", create_checked_parts)
    if oom_step is not None:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            train.run_preflight(config, rows, vocabulary, ids, 32, "cpu", "synthetic CPU test")
    else:
        train.run_preflight(config, rows, vocabulary, ids, 32, "cpu", "synthetic CPU test")
    report = json.loads((tmp_path / "training" / "preflight.json").read_text())
    assert report["input_length"] == 6
    assert report["max_context"] == 32
    assert report["parameter_count"] > 0 and report["vocab_size"] == 8
    assert report["model_weights_saved"] is False
    assert not (tmp_path / "training" / "checkpoints").exists()
    assert not (tmp_path / "training" / "metrics.json").exists()
    assert report["optimizer_steps_planned"] == 2
    assert report["memory_check_limitations"]
    if oom_step is not None:
        assert report["status"] == "failed" and report["out_of_memory"] is True
        assert report["failed_phase"] == ("optimizer_step" if oom_step == 1 else "optimizer_step_with_optimizer_state")
        assert report["optimizer_steps_completed"] == oom_step - 1
        assert report["step_with_existing_optimizer_state_completed"] is False
    else:
        assert report["status"] == "passed" and report["optimizer_step_completed"] is True
        assert report["optimizer_steps_completed"] == 2
        assert report["optimizer_state_present_before_second_forward"] is True
        assert report["step_with_existing_optimizer_state_completed"] is True
        assert states_seen_at_forward == [False, True]
        assert len(report["step_losses"]) == 2
        assert report["optimizer_state_bytes"] > 0
        assert report["nonpad_input_tokens"] == 6 and report["supervised_tokens"] == 5


def test_preflight_rejects_resume_before_loading_config():
    with pytest.raises(ValueError, match="--resume"):
        train.run_pretrain("unused", None, None, True, check_only=True)


def test_multi_gpu_split_keeps_every_microbatch_once_and_respects_half_share(monkeypatch):
    lengths = [16_000, 9_000, 8_000, 4_000, 3_000, 2_000, 1_000, 500]
    step = [[{"person_id": index, "input_length": length}] for index, length in enumerate(lengths)]
    monkeypatch.setenv("GPU_WORK_SHARES", "1,1,0.5")
    shares = []
    for rank in range(3):
        monkeypatch.setattr(train, "rank_and_world", lambda rank=rank: (rank, 3))
        shares.append(train.my_microbatches(step))
    assigned = sorted(batch[0]["person_id"] for share in shares for batch in share)
    assert assigned == list(range(len(lengths)))
    loads = [sum(train.microbatch_cost(batch) for batch in share) for share in shares]
    assert loads[2] < min(loads[0], loads[1])

    monkeypatch.setattr(train, "rank_and_world", lambda: (0, 1))
    assert train.my_microbatches(step) == step


def test_gpu_work_shares_reject_wrong_count(monkeypatch):
    monkeypatch.setenv("GPU_WORK_SHARES", "1,0.5")
    with pytest.raises(ValueError):
        train.gpu_work_shares(3)
    monkeypatch.delenv("GPU_WORK_SHARES")
    assert train.gpu_work_shares(2) == [1.0, 1.0]


def test_model_architecture_defaults_to_gpt2_and_rejects_unknown():
    assert train.model_architecture({}) == "gpt2"
    for unknown in ("llama_rope", "neox_rope", "mamba"):
        with pytest.raises(ValueError):
            train.model_architecture({"model": {"architecture": unknown}})


def test_training_seed_defaults_to_42_and_changes_batch_order():
    from ausom_ehr.train import make_microbatches, training_seed
    assert training_seed({}) == 42
    assert training_seed({"training": {"seed": 43}}) == 43
    rows = [{"person_id": i, "input_length": 10 + i % 40} for i in range(400)]
    same = make_microbatches(rows, shuffle=True, padded_limit=64)
    assert [b[0]["person_id"] for b in same] == [b[0]["person_id"] for b in make_microbatches(rows, True, 64, seed=42)]
    other = make_microbatches(rows, shuffle=True, padded_limit=64, seed=43)
    assert sorted(len(b) for b in same) == sorted(len(b) for b in other)
    assert [b[0]["person_id"] for b in same] != [b[0]["person_id"] for b in other]


def test_completed_ten_epochs_can_extend_without_changing_old_record():
    old = {"max_epochs": 10, "completed_epochs": 10, "stopped": True, "best_epoch": 10}
    new, extended = extend_epoch_goal(old, 20)
    assert extended is True
    assert new["max_epochs"] == 20
    assert new["completed_epochs"] == 10
    assert new["stopped"] is False
    assert old["max_epochs"] == 10
    assert old["stopped"] is True


def test_unfinished_goal_cannot_change_mid_run():
    unfinished = {"max_epochs": 20, "completed_epochs": 11, "stopped": False}
    with pytest.raises(ValueError, match="Finish the current goal"):
        extend_epoch_goal(unfinished, 30)


def test_smaller_goal_is_rejected_and_same_finished_goal_is_unchanged():
    complete = {"max_epochs": 10, "completed_epochs": 10, "stopped": True}
    with pytest.raises(ValueError, match="lower the epoch goal"):
        extend_epoch_goal(complete, 9)
    unchanged, extended = extend_epoch_goal(complete, 10)
    assert unchanged == complete
    assert extended is False


def test_new_epoch_evaluation_has_its_own_folder():
    assert result_folder_parts("embeddings", "continued") == ("embeddings", "continued")
    assert result_folder_parts("embeddings", "continued", "epoch20") == (
        "embeddings", "continued", "epoch20"
    )
    assert result_folder_parts("evaluation", "continued", "epoch20") == (
        "evaluation", "continued", "epoch20"
    )
    with pytest.raises(ValueError):
        result_folder_parts("evaluation", "continued", "../epoch10")
    with pytest.raises(ValueError):
        result_folder_parts("evaluation", "pretrained", "epoch20")


def test_early_stopping_waits_for_three_epochs_without_meaningful_gain():
    metrics = {
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.001,
        "early_stopping_reference_loss": 0.8000,
        "epochs_without_significant_improvement": 0,
    }
    # One small gain does not move the reference; an accumulated 0.001 gain does.
    assert update_early_stopping(metrics, 0.7996) is False
    assert metrics["epochs_without_significant_improvement"] == 1
    assert update_early_stopping(metrics, 0.7989) is False
    assert metrics["epochs_without_significant_improvement"] == 0
    assert metrics["early_stopping_reference_loss"] == 0.7989
    assert update_early_stopping(metrics, 0.7987) is False
    assert update_early_stopping(metrics, 0.7986) is False
    assert update_early_stopping(metrics, 0.7985) is True
