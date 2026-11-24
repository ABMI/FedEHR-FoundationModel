"""Virtual sites, FedAvg training of the GPT-2 EHR model across sites, and per-site evaluation.

Stages: site-profile, site-make, site-disjoint, server, client, eval
Usage: python -m ausom_ehr.federated <stage> [options]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.optimize import linprog

from ausom_ehr.common import RAW_ROOT, load_config, output_file, write_json
from ausom_ehr.tokens import context_limit, special_ids
from ausom_ehr.train import (
    ADAM_BETAS, BASE_LEARNING_RATE, CONTINUED_LR, WEIGHT_DECAY, evaluate, finish_gpus,
    group_optimizer_steps, learning_rate_multiplier, load_split, make_microbatches, make_model,
    model_architecture, setup_gpus, train_epoch, training_seed, wait_for_all_gpus,
)


# ── Virtual sites ──
# Sites never overlap the reference site; site-make sites may overlap each other, site-disjoint sites never do.

SITE_SIZE = 300_000
SPLIT_SHARES = {"train": 0.70, "validation": 0.15, "test": 0.15}
DATA_END_YEAR = 2025  # clinical data ends in 2025; caps filler end years such as 3000 or 9999
FEMALE, MALE = 8532, 8507
INPATIENT_VISITS = [9201, 262, 32037]
ER_VISITS = [9203, 262]
OUTPATIENT_VISIT = 9202
CHECKUP_VISIT = 32693
# Any diagnosis of this SNOMED concept or a descendant puts a patient in the group.
DISEASE_GROUPS = {
    "cancer": 443392, "diabetes": 201820, "hypertension": 316866, "ischemic_heart": 4185932,
    "cerebrovascular": 381591, "heart_failure": 316139, "ckd": 46271022, "copd": 255573,
    "asthma": 317009, "mental": 432586, "dementia": 4182210,
}
# The one shifted axis per site and its target share; None takes every eligible patient.
SITES = {
    "D0_random": None,
    "D1_elderly": ("age_65_plus", 0.80),
    "D2_inpatient": ("ever_inpatient", 0.80),
    "D3_cancer": ("dx_cancer", None),
    "D4_female": ("female", 0.80),
}
# Disjoint design (D0 is reused as is); the SITES axes overlap too much to be drawn disjointly.
DISJOINT_SITES = {
    "D1_pediatric": ("pediatric", 0.80),
    "D2_er": ("ever_er", 0.80),
    "D3_cancer": ("dx_cancer", 0.25),
    "D4_female": ("female", 0.80),
    "D5_elderly": ("age_65_plus", 0.60),
}
# Patient cells for disjoint allocation; every target axis is a function of these columns.
CELL_COLUMNS = ["age_group", "ever_inpatient", "ever_er", "dx_cancer", "female"]


def build_profile(raw_root: Path, reference_path: Path) -> pl.DataFrame:
    """One row per patient: sex, age, visit counts by type, disease-group flags."""
    dbo = raw_root / "dbo"
    person = pl.scan_parquet(dbo / "PERSON" / "**" / "*.parquet").select(
        pl.col("person_id").cast(pl.Int64), pl.col("gender_concept_id").alias("gender"), "year_of_birth",
    )
    period = (
        pl.scan_parquet(dbo / "OBSERVATION_PERIOD" / "**" / "*.parquet")
        .select(
            pl.col("person_id").cast(pl.Int64),
            pl.col("observation_period_start_date").str.slice(0, 4).cast(pl.Int32).alias("start_year"),
            pl.col("observation_period_end_date").str.slice(0, 4).cast(pl.Int32).alias("end_year"),
        )
        .group_by("person_id")
        .agg(pl.col("start_year").min(), pl.col("end_year").max())
    )
    visit = pl.col("visit_concept_id")
    visits = (
        pl.scan_parquet(dbo / "VISIT_OCCURRENCE" / "**" / "*.parquet")
        .select(pl.col("person_id").cast(pl.Int64), "visit_concept_id")
        .group_by("person_id")
        .agg(
            pl.len().alias("visits"),
            visit.is_in(INPATIENT_VISITS).sum().alias("inpatient_visits"),
            visit.is_in(ER_VISITS).sum().alias("er_visits"),
            (visit == OUTPATIENT_VISIT).sum().alias("outpatient_visits"),
            (visit == CHECKUP_VISIT).sum().alias("checkup_visits"),
        )
    )
    descendants = (
        pl.scan_parquet(dbo / "CONCEPT_ANCESTOR" / "**" / "*.parquet")
        .filter(pl.col("ancestor_concept_id").is_in(list(DISEASE_GROUPS.values())))
        .select(
            pl.col("ancestor_concept_id").alias("group_id"),
            pl.col("descendant_concept_id").alias("condition_concept_id"),
        )
    )
    diagnosed = (
        pl.scan_parquet(dbo / "CONDITION_OCCURRENCE" / "**" / "*.parquet")
        .select(pl.col("person_id").cast(pl.Int64), "condition_concept_id")
        .join(descendants, on="condition_concept_id", how="inner")
        .select("person_id", "group_id")
        .unique()
        .collect(engine="streaming")
    )
    disease_flags = pl.DataFrame({"person_id": diagnosed["person_id"].unique().sort()})
    for name, concept in DISEASE_GROUPS.items():
        ids = diagnosed.filter(pl.col("group_id") == concept)["person_id"]
        disease_flags = disease_flags.with_columns(pl.col("person_id").is_in(ids.implode()).alias(f"dx_{name}"))
    reference = pl.scan_parquet(reference_path).select(pl.col("person_id").cast(pl.Int64), pl.lit(True).alias("reference"))
    count_columns = ["visits", "inpatient_visits", "er_visits", "outpatient_visits", "checkup_visits"]
    flag_columns = [f"dx_{name}" for name in DISEASE_GROUPS]
    profile = (
        person.join(period, on="person_id", how="left")
        .join(visits, on="person_id", how="left")
        .join(reference, on="person_id", how="left")
        .collect(engine="streaming")
        .join(disease_flags, on="person_id", how="left")
        .with_columns(
            pl.col("reference").fill_null(False),
            *[pl.col(column).fill_null(0) for column in count_columns],
            *[pl.col(column).fill_null(False) for column in flag_columns],
            (pl.min_horizontal(pl.col("end_year"), pl.lit(DATA_END_YEAR)) - pl.col("year_of_birth")).alias("age_last"),
        )
    )
    return add_axes(profile)


def add_axes(profile: pl.DataFrame) -> pl.DataFrame:
    return profile.with_columns(
        (pl.col("gender") == FEMALE).alias("female"),
        (pl.col("age_last") >= 65).alias("age_65_plus"),
        pl.when(pl.col("age_last") < 19).then(pl.lit("0-18"))
        .when(pl.col("age_last") < 40).then(pl.lit("19-39"))
        .when(pl.col("age_last") < 65).then(pl.lit("40-64"))
        .otherwise(pl.lit("65+")).alias("age_group"),
        (pl.col("inpatient_visits") > 0).alias("ever_inpatient"),
        (pl.col("er_visits") > 0).alias("ever_er"),
        pl.when(pl.col("visits") == 0).then(pl.lit("no_visit"))
        .when(pl.col("inpatient_visits") > 0).then(pl.lit("inpatient"))
        .when(pl.col("er_visits") > 0).then(pl.lit("er_only"))
        .when(pl.col("outpatient_visits") > 0).then(pl.lit("outpatient_only"))
        .otherwise(pl.lit("other")).alias("care_type"),
    )


def draw(pool: pl.DataFrame, size: int, rng: np.random.Generator) -> pl.Series:
    """Draw size patients without replacement; ids are sorted first so the draw is reproducible."""
    if size > pool.height:
        raise ValueError(f"Cannot draw {size:,} patients from {pool.height:,} candidates.")
    ids = pool["person_id"].sort().to_numpy()
    return pl.Series("person_id", np.sort(rng.choice(ids, size=size, replace=False)))


def make_site(pool: pl.DataFrame, rule: tuple[str, float | None] | None, rng: np.random.Generator) -> pl.Series:
    """Draw SITE_SIZE patients at the target share of one axis; other axes follow within each stratum."""
    if rule is None:
        return draw(pool, SITE_SIZE, rng)
    column, share = rule
    focus = pool.filter(pl.col(column))
    other = pool.filter(~pl.col(column))
    focus_size = focus.height if share is None else round(SITE_SIZE * share)
    focus_size = min(focus_size, SITE_SIZE)
    return pl.concat([draw(focus, focus_size, rng), draw(other, SITE_SIZE - focus_size, rng)]).sort()


def assign_splits(person_ids: pl.Series, rng: np.random.Generator) -> pl.DataFrame:
    """Patient-level 70/15/15 train/validation/test split within a site."""
    order = rng.permutation(len(person_ids))
    cut_train = round(len(person_ids) * SPLIT_SHARES["train"])
    cut_validation = cut_train + round(len(person_ids) * SPLIT_SHARES["validation"])
    splits = np.empty(len(person_ids), dtype=object)
    splits[order[:cut_train]] = "train"
    splits[order[cut_train:cut_validation]] = "validation"
    splits[order[cut_validation:]] = "test"
    return pl.DataFrame({"person_id": person_ids, "split": splits.astype(str)})


def describe(group: pl.DataFrame) -> dict:
    """Axis distributions of a patient group, in percent."""
    def shares(column: str) -> dict:
        counts = group[column].value_counts(normalize=True).sort(column)
        return {str(key): round(float(value) * 100, 2) for key, value in counts.iter_rows()}

    disease = {
        name: round(float(group[f"dx_{name}"].mean()) * 100, 2) for name in DISEASE_GROUPS
    }
    return {
        "patients": group.height,
        "female_%": round(float(group["female"].mean()) * 100, 2),
        "age_group_%": shares("age_group"),
        "care_type_%": shares("care_type"),
        "ever_inpatient_%": round(float(group["ever_inpatient"].mean()) * 100, 2),
        "ever_er_%": round(float(group["ever_er"].mean()) * 100, 2),
        "disease_%": disease,
    }


def smd(site: pl.DataFrame, reference: pl.DataFrame) -> dict:
    """Standardized mean differences vs the reference site."""
    columns = ["female", "age_65_plus", "ever_inpatient", "ever_er", *[f"dx_{name}" for name in DISEASE_GROUPS]]
    result = {}
    for column in columns:
        p_site = float(site[column].mean())
        p_reference = float(reference[column].mean())
        pooled = np.sqrt((p_site * (1 - p_site) + p_reference * (1 - p_reference)) / 2)
        result[column] = round((p_site - p_reference) / pooled, 3) if pooled > 0 else 0.0
    age_site = site["age_last"].cast(pl.Float64)
    age_reference = reference["age_last"].cast(pl.Float64)
    pooled_age = np.sqrt((age_site.var() + age_reference.var()) / 2)
    result["age_last"] = round(float((age_site.mean() - age_reference.mean()) / pooled_age), 3)
    return result


def make_sites(profile: pl.DataFrame, output_root: Path, seed: int = 42) -> dict:
    reference = profile.filter(pl.col("reference"))
    pool = profile.filter(~pl.col("reference"))
    rng = np.random.default_rng(seed)
    summary = {
        "seed": seed,
        "site_size": SITE_SIZE,
        "pool": "AUSOM patients not in the reference 300k sample",
        "pool_patients": pool.height,
        "reference": describe(reference),
        "sites": {},
    }
    for name, rule in SITES.items():
        person_ids = make_site(pool, rule, rng)
        if person_ids.n_unique() != SITE_SIZE or person_ids.is_in(reference["person_id"].implode()).any():
            raise ValueError(f"{name}: wrong size or overlaps the reference site.")
        assign_splits(person_ids, rng).write_parquet(output_root / f"{name}.parquet")
        site = pool.filter(pl.col("person_id").is_in(person_ids.implode()))
        summary["sites"][name] = {
            "rule": None if rule is None else {"column": rule[0], "target_share": rule[1] or "all available"},
            **describe(site),
            "smd_vs_reference": smd(site, reference),
        }
        print(f"{name}: {SITE_SIZE:,} patients", flush=True)
    write_json(output_root / "sites_summary.json", summary)
    return summary


def allocate_cells(pool: pl.DataFrame, taken: pl.Series, rules: dict, size: int) -> tuple[pl.DataFrame, np.ndarray, float]:
    """Site x cell counts by LP: minimal L1 shift from the make_site mix, exact site and target-axis sizes.

    Returns (cells, integer site x cell counts, share of patients moved from the natural mix).
    """
    columns = list(dict.fromkeys(CELL_COLUMNS + [column for column, _ in rules.values()]))
    cells = pool.group_by(columns).agg(
        pl.len().alias("full"),
        (~pl.col("person_id").is_in(taken.implode())).sum().alias("left"),
    ).sort(columns)
    full = cells["full"].to_numpy().astype(float)
    left = cells["left"].to_numpy().astype(float)
    k, s = cells.height, len(rules)
    n = k * s
    natural, focus, goals = [], [], []
    for column, share in rules.values():
        in_focus = cells[column].to_numpy().astype(float)
        natural.append(size * (share * full * in_focus / (full * in_focus).sum()
                               + (1 - share) * full * (1 - in_focus) / (full * (1 - in_focus)).sum()))
        focus.append(in_focus)
        goals.append(round(size * share))
    natural = np.concatenate(natural)
    # Variables: site x cell counts x and deviations e; |x - natural| <= e as two inequalities.
    identity = np.eye(n)
    cell_sum = np.hstack([np.tile(np.eye(k), s), np.zeros((k, n))])
    a_ub = np.vstack([np.hstack([identity, -identity]), np.hstack([-identity, -identity]), cell_sum])
    b_ub = np.concatenate([natural, -natural, np.maximum(left - s, 0)])  # one patient per site of rounding slack
    a_eq, b_eq = [], []
    for site in range(s):
        for weights, target in [(np.ones(k), size), (focus[site], goals[site])]:
            row = np.zeros(2 * n)
            row[site * k:(site + 1) * k] = weights
            a_eq.append(row)
            b_eq.append(target)
    result = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=a_ub, b_ub=b_ub, A_eq=np.array(a_eq), b_eq=b_eq,
                     bounds=(0, None), method="highs")
    if result.status != 0:
        raise ValueError(f"No disjoint site allocation exists: {result.message}")
    counts = result.x[:n].reshape(s, k)
    moved = float(result.x[n:].sum() / 2 / (s * size))
    # Floor focus and non-focus cells separately, then add 1 to the largest remainders to hit the targets.
    whole = np.zeros((s, k), dtype=np.int64)
    for site in range(s):
        for part, target in [(focus[site] == 1, goals[site]), (focus[site] == 0, size - goals[site])]:
            candidates = np.flatnonzero(part)
            values = counts[site, candidates]
            floor = np.floor(values + 1e-9).astype(np.int64)
            order = np.argsort(-(values - floor), kind="stable")
            floor[order[:target - floor.sum()]] += 1
            whole[site, candidates] = floor
    if (whole.sum(axis=0) > left).any():
        raise ValueError("Cell counts exceed the remaining patients.")
    return cells, whole, moved


def make_disjoint_sites(profile: pl.DataFrame, first_site: Path, output_root: Path, seed: int = 42) -> dict:
    """Keep the first site (D0) as is; draw DISJOINT_SITES disjoint from the reference, D0 and each other."""
    profile = profile.with_columns((pl.col("age_group") == "0-18").alias("pediatric"))
    reference = profile.filter(pl.col("reference"))
    pool = profile.filter(~pl.col("reference"))
    first = pl.read_parquet(first_site)
    if first["person_id"].is_in(reference["person_id"].implode()).any():
        raise ValueError(f"{first_site}: overlaps the reference site.")
    rng = np.random.default_rng(seed)
    cells, counts, moved = allocate_cells(pool, first["person_id"], DISJOINT_SITES, SITE_SIZE)
    columns = [column for column in cells.columns if column not in ("full", "left")]
    cells = cells.with_row_index("cell")
    # Shuffle remaining patients within each cell; sites take consecutive slices of each cell.
    left = pool.filter(~pl.col("person_id").is_in(first["person_id"].implode())).sort("person_id")
    left = (
        left.with_columns(pl.Series("order", rng.permutation(left.height)))
        .join(cells.select("cell", *columns), on=columns)
        .with_columns((pl.col("order").rank("ordinal").over("cell").cast(pl.Int64) - 1).alias("rank"))
    )
    starts = np.vstack([np.zeros(cells.height, dtype=np.int64), np.cumsum(counts, axis=0)])
    shutil.copyfile(first_site, output_root / first_site.name)
    summary = {
        "seed": seed,
        "site_size": SITE_SIZE,
        "pool": "AUSOM patients not in the reference 300k sample; sites share no patients",
        "pool_patients": pool.height,
        "first_site": str(first_site),
        "cells": columns,
        "moved_from_natural_share": round(moved, 5),
        "reference": describe(reference),
        "sites": {},
    }
    first_frame = pool.filter(pl.col("person_id").is_in(first["person_id"].implode()))
    summary["sites"][first_site.stem] = {
        "rule": None, **describe(first_frame), "smd_vs_reference": smd(first_frame, reference),
    }
    used = [first["person_id"]]
    for index, (name, rule) in enumerate(DISJOINT_SITES.items()):
        bounds = pl.DataFrame({
            "cell": pl.Series(np.arange(cells.height), dtype=pl.UInt32),
            "start": starts[index], "end": starts[index + 1],
        })
        person_ids = (
            left.join(bounds, on="cell")
            .filter((pl.col("rank") >= pl.col("start")) & (pl.col("rank") < pl.col("end")))["person_id"].sort()
        )
        if person_ids.n_unique() != SITE_SIZE:
            raise ValueError(f"{name}: wrong size.")
        assign_splits(person_ids, rng).write_parquet(output_root / f"{name}.parquet")
        used.append(person_ids)
        site = pool.filter(pl.col("person_id").is_in(person_ids.implode()))
        summary["sites"][name] = {
            "rule": {"column": rule[0], "target_share": rule[1]},
            **describe(site),
            "smd_vs_reference": smd(site, reference),
        }
        print(f"{name}: {SITE_SIZE:,} patients", flush=True)
    every = pl.concat(used)
    if every.n_unique() != every.len() or every.is_in(reference["person_id"].implode()).any():
        raise ValueError("Sites share patients with each other or the reference site.")
    write_json(output_root / "sites_summary.json", summary)
    return summary


# ── FedAvg ──
# Only weights reach the server; each client keeps its own AdamW state across rounds.

WARMUP_SHARE = 0.05
MIN_LEARNING_RATE = CONTINUED_LR  # end of the round-1 cosine = later constant rate, as in M0


def vocab_digest(config: dict) -> str:
    return hashlib.sha256(output_file(config, "tokens", "vocab.parquet").read_bytes()).hexdigest()


def load_weights(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["model"] if "model" in state else state


def build_model(config: dict, device: str):
    vocab_path = output_file(config, "tokens", "vocab.parquet")
    ids = special_ids(vocab_path)
    vocab_size = pl.read_parquet(vocab_path).height
    model = make_model(
        vocab_size, ids["[BOS]"], ids["[EOS]"], ids["[PAD]"], context_limit(config), model_architecture(config),
    ).to(device)
    return model, ids


def initial_weights(config_path: str, destination: Path) -> None:
    """Untrained weights from the same seed as M0 pretraining."""
    config = load_config(config_path)
    torch.manual_seed(training_seed(config))
    torch.cuda.manual_seed_all(training_seed(config))
    model, _ = build_model(config, "cpu")
    torch.save({"model": model.state_dict(), "seed": training_seed(config)}, destination)


def run_client(config_path: str, weights: Path, destination: Path, round_number: int, eval_only: bool,
               optimizer_path: Path | None, mode: str = "continued") -> None:
    """Validation loss of the incoming global model; unless eval_only, train 1 epoch and save the weights.

    Under torchrun the epoch is split across GPUs and only rank 0 saves.
    """
    config = load_config(config_path)
    rank, world_size, device = setup_gpus()
    is_main = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    if "RTX 6000 Ada" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Select an RTX 6000 Ada GPU.")
    model, ids = build_model(config, device)
    model.load_state_dict(load_weights(weights))
    started = time.monotonic()
    max_context = context_limit(config)
    validation_rows = load_split(config, "validation", preload_tokens=True)
    validation_batches = make_microbatches(validation_rows, shuffle=False, padded_limit=max_context)
    report = {
        "config": config["config_path"],
        "round": round_number,
        "gpus": world_size,
        "incoming_weights": str(weights),
        "validation_patients": len(validation_rows),
        "incoming_validation_loss": evaluate(model, validation_batches, ids["[PAD]"], device),
    }
    if is_main:
        print(f"round {round_number}: incoming global model validation loss={report['incoming_validation_loss']:.5f}", flush=True)
    if not eval_only:
        seed = training_seed(config)
        train_rows = load_split(config, "train", preload_tokens=True)
        first_round = mode == "scratch" and round_number == 1
        # Scratch round 1 keeps M0's pre-shuffled first-epoch order; other rounds reshuffle.
        train_batches = make_microbatches(train_rows, shuffle=first_round, padded_limit=max_context, seed=seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=BASE_LEARNING_RATE if first_round else CONTINUED_LR,
            betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY,
        )
        scheduler = None
        if first_round:
            total_steps = len(group_optimizer_steps(train_batches))
            warmup_steps = round(total_steps * WARMUP_SHARE)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: learning_rate_multiplier(step, total_steps, warmup_steps, MIN_LEARNING_RATE),
            )
            report["warmup_steps"] = warmup_steps
        elif round_number > 1:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
            for group in optimizer.param_groups:
                group["lr"] = CONTINUED_LR
        torch.manual_seed(seed + round_number)
        train_loss, steps = train_epoch(
            model, optimizer, train_batches, ids["[PAD]"], round_number, device, is_main, seed,
            scheduler=scheduler, shuffle=not first_round,
        )
        report.update({
            "train_patients": len(train_rows),
            "train_target_tokens": int(sum(row["input_length"] - 1 for row in train_rows)),
            "optimizer_steps": steps,
            "train_loss": train_loss,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
        })
        if is_main:
            temporary = destination.with_name(destination.name + ".tmp")
            torch.save({"model": model.state_dict()}, temporary)
            temporary.replace(destination)
            optimizer_temporary = optimizer_path.with_name(optimizer_path.name + ".tmp")
            torch.save(optimizer.state_dict(), optimizer_temporary)
            optimizer_temporary.replace(optimizer_path)
    report["elapsed_seconds"] = time.monotonic() - started
    if is_main:
        write_json(destination.with_suffix(".json"), report)
    wait_for_all_gpus()
    finish_gpus()


def average(states: list[dict], weights: list[float]) -> dict:
    """Weighted average; integer buffers keep the first client's value."""
    total = sum(weights)
    result = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            result[key] = sum(state[key].float() * (weight / total) for state, weight in zip(states, weights)).to(value.dtype)
        else:
            result[key] = value.clone()
    return result


RUN_CAPPED = Path(__file__).resolve().parents[2] / "scripts" / "run_capped.py"


def memory_fractions() -> dict[str, str]:
    """GPU_MEMORY_FRACTIONS="4:0.5" caps clients on GPU 4 at 50% of its memory (shared GPUs)."""
    pairs = [item.split(":") for item in os.environ.get("GPU_MEMORY_FRACTIONS", "").split(",") if item]
    return {gpu.strip(): fraction.strip() for gpu, fraction in pairs}


def client_command(name: str, config_path: str, gpus: str, weights: Path, folder: Path, round_number: int,
                   eval_only: bool, run_dir: Path, mode: str) -> tuple[list[str], dict]:
    environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpus)
    count = len(gpus.split(","))
    launcher = [sys.executable, "-u", "-m"]
    fraction = memory_fractions().get(gpus)
    if fraction is not None and count == 1:
        environment["GPU_MEMORY_FRACTION"] = fraction
        launcher = [sys.executable, "-u", str(RUN_CAPPED)]
    if count > 1:
        # GPU peer-to-peer transfers can hang; go through host memory.
        environment["NCCL_P2P_DISABLE"] = "1"
        environment["MASTER_PORT"] = str(29500 + abs(hash(name)) % 400)
        launcher = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={count}", "-m"]
    command = launcher + [
        "ausom_ehr.federated", "client", "--config", config_path, "--weights", str(weights),
        "--output", str(folder / f"{name}.pt"), "--round", str(round_number),
        "--optimizer-state", str(run_dir / "clients" / f"{name}_optimizer.pt"), "--mode", mode,
    ]
    if eval_only:
        command.append("--eval-only")
    return command, environment


def launch(clients: dict[str, tuple[str, str]], weights: Path, folder: Path, round_number: int, eval_only: bool,
           run_dir: Path, mode: str) -> None:
    """Run every client on its GPUs and wait; clients sharing a GPU string run in turn.

    On restart, clients that already saved this round's results are skipped.
    """
    queues: dict[str, list[str]] = {}
    for name, (_, gpus) in clients.items():
        finished = (folder / f"{name}.json").exists() and (eval_only or (folder / f"{name}.pt").exists())
        if not finished:
            queues.setdefault(gpus, []).append(name)
    running = {}
    failed = []

    def start(gpus: str) -> None:
        name = queues[gpus].pop(0)
        command, environment = client_command(name, clients[name][0], gpus, weights, folder, round_number, eval_only,
                                              run_dir, mode)
        log = open(run_dir / "logs" / f"round_{round_number:02d}_{name}{'_eval' if eval_only else ''}.log", "w")
        running[gpus] = (name, subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT), log)

    for gpus in queues:
        start(gpus)
    while running:
        time.sleep(5)
        for gpus, (name, process, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            del running[gpus]
            if process.returncode != 0:
                failed.append(name)
            if queues[gpus]:
                start(gpus)
    if failed:
        raise RuntimeError(f"round {round_number}: clients failed {failed}. Logs: {run_dir / 'logs'}")


def run_server(run_dir: Path, clients: dict[str, tuple[str, str]], rounds: int, init: Path | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["logs", "clients", "round_00"]:
        (run_dir / name).mkdir(exist_ok=True)
    configs = {name: load_config(config_path) for name, (config_path, _) in clients.items()}
    digests = {name: vocab_digest(config) for name, config in configs.items()}
    if len(set(digests.values())) != 1:
        raise ValueError(f"Sites use different vocabularies: {digests}")
    mode = "scratch" if init is None else "continued"
    if init is None:
        init = run_dir / "round_00" / "global.pt"
        if not init.exists():
            initial_weights(next(iter(clients.values()))[0], init)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {
        "mode": mode,
        "start": "same seeded initial weights as M0 pretraining (from scratch)" if mode == "scratch" else str(init),
        "clients": {name: {"config": config_path, "gpus": gpus} for name, (config_path, gpus) in clients.items()},
        "vocab_sha256": next(iter(digests.values())),
        "local_epochs_per_round": 1,
        "learning_rate": (
            "round 1: warmup 5% + cosine 2e-4 -> 2e-5; later rounds: 2e-5 constant"
            if mode == "scratch" else "2e-5 constant (M0 continuation rate)"
        ),
        "optimizer_state": "kept inside each client across rounds",
        "aggregation": "FedAvg weighted by train target tokens",
        "rounds": [],
    }
    metrics["rounds_planned"] = rounds
    write_json(metrics_path, metrics)
    weights = init
    for round_number in range(1, rounds + 1):
        folder = run_dir / f"round_{round_number:02d}"
        global_path = folder / "global.pt"
        if global_path.exists():
            weights = global_path
            continue
        folder.mkdir(exist_ok=True)
        started = time.monotonic()
        launch(clients, weights, folder, round_number, False, run_dir, mode)
        reports = {name: json.loads((folder / f"{name}.json").read_text()) for name in clients}
        token_weights = [reports[name]["train_target_tokens"] for name in clients]
        states = [load_weights(folder / f"{name}.pt") for name in clients]
        temporary = global_path.with_name("global.pt.tmp")
        torch.save({"model": average(states, token_weights), "round": round_number}, temporary)
        temporary.replace(global_path)
        del states
        metrics["rounds"].append({
            "round": round_number,
            "aggregation_weights": {name: w / sum(token_weights) for name, w in zip(clients, token_weights)},
            "clients": reports,
            "elapsed_seconds": time.monotonic() - started,
        })
        write_json(metrics_path, metrics)
        for name in clients:
            (folder / f"{name}.pt").unlink()
        print(f"round {round_number}/{rounds} done ({(time.monotonic() - started) / 3600:.2f}h)", flush=True)
        weights = global_path
    # Final global model: validation loss on every site, then copy it for evaluation.
    final_folder = run_dir / "final"
    final_folder.mkdir(exist_ok=True)
    final_state = final_folder / "state.pt"
    if final_state.exists() and json.loads(metrics_path.read_text()).get("final_round") == rounds:
        print(f"{rounds}-round result already exists: {final_state}", flush=True)
        return
    launch(clients, weights, final_folder, rounds + 1, True, run_dir, mode)
    metrics["final_round"] = rounds
    metrics["final_validation_loss"] = {
        name: json.loads((final_folder / f"{name}.json").read_text())["incoming_validation_loss"] for name in clients
    }
    shutil.copy2(weights, final_folder / "state.pt.tmp")
    (final_folder / "state.pt.tmp").replace(final_state)
    write_json(metrics_path, metrics)
    print(f"FedAvg {rounds} rounds done: {final_state}", flush=True)


def parse_clients(values: list[str]) -> dict[str, tuple[str, str]]:
    """Parse 'NAME=CONFIG@GPUS', where GPUS is '4' or '4,2'."""
    clients = {}
    for value in values:
        name, rest = value.split("=", 1)
        config_path, gpus = rest.rsplit("@", 1)
        clients[name] = (config_path, gpus)
    return clients


# ── Evaluation ──

def link(target: Path, link_path: Path) -> None:
    """Symlink link_path to target; idempotent, raises if it links elsewhere or is a real file."""
    if not target.exists():
        raise FileNotFoundError(target)
    if link_path.is_symlink():
        if link_path.resolve() != target.resolve():
            raise ValueError(f"linked to a different input: {link_path}")
        return
    if link_path.exists():
        raise FileExistsError(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=target.is_dir())


EVAL_CONFIG = """# Evaluation-only config written by federated eval; not used for training.
[paths]
raw_root = "{raw_root}"
output_root = "{output_root}"
[tokenization]
mode = "concept_value_joint"
demographic_tokens = true
[model]
architecture = "gpt2"
"""


def prepare_eval(model: Path, population_root: Path, output_root: Path, raw_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    link(model.resolve(), output_root / "training" / "continued" / "best" / "state.pt")
    link(population_root / "tokens" / "vocab.parquet", output_root / "tokens" / "vocab.parquet")
    link(population_root / "cohorts" / "inputs", output_root / "cohorts" / "inputs")
    link(population_root / "cohorts" / "inputs_summary.json", output_root / "cohorts" / "inputs_summary.json")
    config_path = output_root / "eval_config.toml"
    config_path.write_text(EVAL_CONFIG.format(raw_root=raw_root, output_root=output_root.resolve()))
    return config_path


def run_sites(command: str, raw_root: Path, reference: Path, output: Path, seed: int) -> None:
    """command 'profile' writes the patient profile; 'make' writes the site patient lists."""
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "patient_profile.parquet"
    if command == "profile":
        build_profile(raw_root, reference).write_parquet(profile_path)
    else:
        make_sites(read_profile(profile_path), output, seed)


def read_profile(path: Path) -> pl.DataFrame:
    """Load a saved profile and rebuild its axis columns with the current definitions."""
    return add_axes(pl.read_parquet(path).select(
        pl.exclude("female", "age_65_plus", "age_group", "ever_inpatient", "ever_er", "care_type")
    ))


def run_eval(model: Path, population_root: Path, output: Path, raw_root: Path, run_name: str) -> None:
    """Build the eval folder (read-only links to weights, vocab, task inputs), then run evaluate embed and logistic."""
    config_path = prepare_eval(model, population_root.resolve(), output, raw_root)
    common = ["--config", str(config_path), "--initialization", "continued", "--run-name", run_name]
    capped = float(os.environ.get("GPU_MEMORY_FRACTION", "1")) < 1
    launcher = [sys.executable, "-u", str(RUN_CAPPED)] if capped else [sys.executable, "-u", "-m"]
    for stage in ["embed", "logistic"]:
        subprocess.run([*launcher, "ausom_ehr.evaluate", stage, *common], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage, help_text in [("site-profile", "profile all patients on the site axes"), ("site-make", "virtual site patient lists")]:
        sub = stages.add_parser(stage, help=help_text)
        sub.add_argument("--raw-root", type=Path, default=RAW_ROOT)
        sub.add_argument("--reference", type=Path, default=Path("outputs/base/sample/selected_patients.parquet"))
        sub.add_argument("--output", type=Path, default=Path("outputs/sites"))
        sub.add_argument("--seed", type=int, default=42)
    disjoint = stages.add_parser("site-disjoint", help="virtual site patient lists that share no patients")
    disjoint.add_argument("--profile", type=Path, default=Path("outputs/sites/patient_profile.parquet"))
    disjoint.add_argument("--first-site", type=Path, default=Path("outputs/sites/D0_random.parquet"))
    disjoint.add_argument("--output", type=Path, default=Path("outputs/sites_disjoint"))
    disjoint.add_argument("--seed", type=int, default=42)
    server = stages.add_parser("server", help="FedAvg server")
    server.add_argument("--run-dir", type=Path, required=True)
    server.add_argument("--client", action="append", default=[], help="NAME=CONFIG@GPUS, repeatable")
    server.add_argument("--init", type=Path, help="start weights (M0 state.pt); omit to train from scratch")
    server.add_argument("--rounds", type=int, default=5)
    client = stages.add_parser("client", help="one site's training round (launched by server)")
    client.add_argument("--config", required=True)
    client.add_argument("--weights", type=Path, required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--round", type=int, required=True)
    client.add_argument("--optimizer-state", type=Path)
    client.add_argument("--eval-only", action="store_true")
    client.add_argument("--mode", choices=["continued", "scratch"], default="continued")
    evaluation = stages.add_parser("eval", help="evaluate an FL model on one site's tasks")
    evaluation.add_argument("--model", required=True, type=Path)
    evaluation.add_argument("--population-root", required=True, type=Path, help="joint-token folder with the task inputs")
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    evaluation.add_argument("--run-name", default="epoch10", help="result folder name, e.g. epoch10")
    args = parser.parse_args()
    if args.stage in ("site-profile", "site-make"):
        run_sites(args.stage.removeprefix("site-"), args.raw_root, args.reference, args.output, args.seed)
    elif args.stage == "site-disjoint":
        args.output.mkdir(parents=True, exist_ok=True)
        make_disjoint_sites(read_profile(args.profile), args.first_site, args.output, args.seed)
    elif args.stage == "server":
        run_server(args.run_dir, parse_clients(args.client), args.rounds, args.init)
    elif args.stage == "client":
        run_client(args.config, args.weights, args.output, args.round, args.eval_only, args.optimizer_state, args.mode)
    else:
        run_eval(args.model, args.population_root, args.output, args.raw_root, args.run_name)


if __name__ == "__main__":
    main()
