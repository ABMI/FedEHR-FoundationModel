"""Frozen GPT-2 embeddings and baseline models on the downstream tasks (EHRSHOT protocol).

Stages: embed, logistic, count-baselines
Usage: python -m ausom_ehr.evaluate <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import joblib
import lightgbm
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from ausom_ehr.common import (
    config_parent, load_config, output_file, result_folder_parts, source_parts, write_json,
)
from ausom_ehr.tasks import TASKS
from ausom_ehr.tokens import context_limit, special_ids
from ausom_ehr.train import (
    autocast_context, make_microbatches, make_model, model_architecture, rows_with_token_arrays,
    tensor_batch,
)


# ── Embeddings ──

def embed_task(config: dict, task: str, model, pad_id: int, initialization: str = "pretrained", run_name: str | None = None, max_context: int = 16_384) -> dict:
    """Last non-PAD hidden state (768-d) of every index sample of one task."""
    folder = result_folder_parts("embeddings", initialization, run_name)
    destination = output_file(config, *folder, f"{task}.parquet")
    if destination.exists():
        existing = pl.scan_parquet(destination).select(pl.len()).collect().item()
        print(f"{task}: reusing {existing:,} existing embeddings", flush=True)
        return {"task": task, "patients": existing, "microbatches": None, "elapsed_seconds": 0.0, "reused": True}
    pattern = str(config["paths"]["output_root"] / "cohorts" / "inputs" / "bucket=*.parquet")
    # Load all input tokens once; per-microbatch Parquet reads leave the GPU idle.
    rows = rows_with_token_arrays(
        pl.scan_parquet(pattern, include_file_paths="input_path")
        .filter(pl.col("task") == task)
        .select("person_id", "task", "split", "label", "index_visit_id", "cutoff_date",
                "input_length", "input_path", "input_tokens")
        .collect()
    )
    batches = make_microbatches(rows, shuffle=False, padded_limit=max_context)
    schema = pa.schema([
        ("person_id", pa.int64()), ("split", pa.string()), ("label", pa.int8()),
        ("index_visit_id", pa.int64()), ("cutoff_date", pa.date32()),
        ("embedding", pa.list_(pa.float32(), 768)),
    ])
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    buffer = []
    started = time.monotonic()
    for number, batch in enumerate(batches, start=1):
        input_ids, mask, _labels = tensor_batch(batch, pad_id, "cuda:0")
        with torch.no_grad(), autocast_context():
            hidden = model.base_model(
                input_ids=input_ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
        last_position = mask.sum(dim=1) - 1
        last_hidden = hidden[
            torch.arange(len(batch), device="cuda:0"), last_position
        ].float().cpu().numpy()
        for row, vector in zip(batch, last_hidden):
            buffer.append({
                "person_id": row["person_id"],
                "split": row["split"],
                "label": row["label"],
                "index_visit_id": row["index_visit_id"],
                "cutoff_date": row["cutoff_date"],
                "embedding": vector.tolist(),
            })
        if len(buffer) >= 512:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            buffer = []
        if number % 100 == 0:
            print(f"{task}: {number}/{len(batches)} microbatches", flush=True)
    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    writer.close()
    temporary.replace(destination)
    elapsed = time.monotonic() - started
    print(f"{task}: {len(rows):,} embeddings done ({elapsed/3600:.2f}h)", flush=True)
    return {
        "task": task,
        "patients": len(rows),
        "microbatches": len(batches),
        "elapsed_seconds": elapsed,
    }


def run_embed(config_path: str, tasks: list[str] | None = None, initialization: str = "pretrained", run_name: str | None = None) -> None:
    folder = result_folder_parts("embeddings", initialization, run_name)
    config = load_config(config_path)
    input_summary = output_file(config, "cohorts", "inputs_summary.json")
    if input_summary.exists():
        summary = json.loads(input_summary.read_text(encoding="utf-8"))
        if summary.get("ready_for_embedding") is False:
            raise ValueError("task inputs have unresolved exceptions; check inputs_summary.json.")
    if initialization == "continued":
        checkpoint = output_file(config, "training", "continued", "best", "state.pt")
    else:
        checkpoint = output_file(config, "training", "checkpoints", "best", "state.pt")
    if initialization != "random_init" and not checkpoint.exists():
        raise FileNotFoundError("best validation checkpoint not found")
    if not torch.cuda.is_available() or "RTX 6000 Ada" not in torch.cuda.get_device_name(0):
        raise RuntimeError("select one RTX 6000 Ada GPU")
    ids = special_ids(output_file(config, "tokens", "vocab.parquet"))
    vocab_size = pl.read_parquet(output_file(config, "tokens", "vocab.parquet")).height
    if initialization == "random_init":
        # Same seed as pretraining, to reproduce the initial weights.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    max_context = context_limit(config)
    model = make_model(
        vocab_size, ids["[BOS]"], ids["[EOS]"], ids["[PAD]"], max_context, model_architecture(config),
    ).to("cuda:0")
    if initialization != "random_init":
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        del state
    model.gradient_checkpointing_disable()
    model.eval()
    model.requires_grad_(False)
    reports = []
    for task in tasks or TASKS:
        reports.append(embed_task(config, task, model, ids["[PAD]"], initialization, run_name, max_context))
    summary_path = output_file(config, *folder, "summary.json")
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        reports = list({row["task"]: row for row in previous["tasks"] + reports}.values())
    write_json(
        summary_path,
        {
            "source_checkpoint": str(checkpoint.parent) if initialization != "random_init" else None,
            "initialization": initialization,
            "run_name": run_name,
            "initialization_seed": 42 if initialization == "random_init" else None,
            "backbone_frozen": True,
            "pooling": "last non-PAD hidden state of observed prefix",
            "synthetic_eos": False,
            "tasks": reports,
        },
    )


# ── Logistic regression ──

# Logistic regression C grid of EHRSHOT (ehrshot/utils.py at publication).
C_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e2, 1e3, 1e4, 1e5, 1e6]
BOOTSTRAP_ROUNDS = 1_000
BOOTSTRAP_SEED = 42


def arrays_for_split(data: pl.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(embeddings [n, 768], binary labels, patient IDs) of one split."""
    part = data.filter(pl.col("split") == split)
    x = np.asarray(part["embedding"].to_list(), dtype=np.float32)
    y = np.asarray(part["label"].to_list(), dtype=np.int8)
    patients = part["person_id"].to_numpy()
    if x.ndim != 2 or x.shape[1] != 768:
        raise ValueError(f"{split}: embeddings are not 768-dimensional")
    if len(np.unique(y)) != 2:
        raise ValueError(f"{split}: AUROC/AUPRC need both positives and negatives")
    return x, y, patients


def scores(y: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
    }


def patient_bootstrap(y: np.ndarray, probabilities: np.ndarray, patients: np.ndarray) -> dict:
    """95% CIs of AUROC and AUPRC by resampling patients with replacement.

    Samples of a patient are drawn together; rounds with a single class are skipped.
    """
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    rows_by_patient = [[] for _ in unique_patients]
    for row, patient in enumerate(inverse):
        rows_by_patient[patient].append(row)
    rows_by_patient = [np.asarray(rows) for rows in rows_by_patient]
    random = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {"auroc": [], "auprc": []}
    for _ in range(BOOTSTRAP_ROUNDS):
        chosen = random.integers(0, len(unique_patients), size=len(unique_patients))
        rows = np.concatenate([rows_by_patient[patient] for patient in chosen])
        if len(np.unique(y[rows])) != 2:
            continue
        values = scores(y[rows], probabilities[rows])
        for name in samples:
            samples[name].append(values[name])
    return {
        f"{name}_ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in samples.items()
    } | {"bootstrap_rounds_used": len(samples["auroc"])}


def patient_weights(patients: np.ndarray) -> np.ndarray:
    """Weight 1/(samples of the patient), so every patient sums to 1."""
    _, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def split_counts(y: np.ndarray, patients: np.ndarray) -> dict:
    return {"samples": int(len(y)), "patients": int(len(np.unique(patients))), "positives": int(y.sum())}


def evaluate_logistic_task(config: dict, task: str, initialization: str = "pretrained", run_name: str | None = None) -> dict:
    """Pick C by validation ROC AUC, then evaluate test once."""
    input_folder = result_folder_parts("embeddings", initialization, run_name)
    output_folder = result_folder_parts("evaluation", initialization, run_name)
    data = pl.read_parquet(output_file(config, *input_folder, f"{task}.parquet"))
    x_train, y_train, p_train = arrays_for_split(data, "train")
    x_val, y_val, p_val = arrays_for_split(data, "validation")
    x_test, y_test, p_test = arrays_for_split(data, "test")
    tuning = []
    best_model = None
    best_c = None
    best_auroc = -1.0
    for c_value in C_VALUES:
        model = make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(
                C=c_value, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000,
            ),
        )
        model.fit(x_train, y_train)
        val_scores = scores(y_val, model.predict_proba(x_val)[:, 1])
        tuning.append({"C": c_value, **val_scores})
        if val_scores["auroc"] > best_auroc:
            best_auroc = val_scores["auroc"]
            best_c = c_value
            best_model = model
    assert best_model is not None
    test_prob = best_model.predict_proba(x_test)[:, 1]
    result = {
        "task": task,
        "initialization": initialization,
        "run_name": run_name,
        "C_candidates": C_VALUES,
        "selected_C": best_c,
        "selection_metric": "validation ROC AUC",
        "scaler": "MaxAbsScaler (train only)",
        "train": split_counts(y_train, p_train),
        "validation": {
            **split_counts(y_val, p_val),
            **next(row for row in tuning if row["C"] == best_c),
        },
        "test": {
            **split_counts(y_test, p_test),
            **scores(y_test, test_prob),
            "auroc_patient_weighted": float(roc_auc_score(y_test, test_prob, sample_weight=patient_weights(p_test))),
            **patient_bootstrap(y_test, test_prob, p_test),
        },
        "validation_tuning": tuning,
        "backbone": "frozen",
        "class_weight": None,
    }
    joblib.dump(best_model, output_file(config, *output_folder, f"{task}_model.joblib"))
    print(
        f"{task}: C={best_c}, test AUROC={result['test']['auroc']:.4f} "
        f"{result['test']['auroc_ci95']}, AUPRC={result['test']['auprc']:.4f}",
        flush=True,
    )
    return result


# ── Count baselines ──

MODELS = ["age_sex_lr", "count_lr", "count_gbm", "count_answers_gbm"]
# LightGBM grid of EHRSHOT appendix Table 9.
GBM_GRID = {
    "learning_rate": [0.02, 0.1, 0.5],
    "max_depth": [3, 6, -1],
    "num_leaves": [10, 25, 100],
}
# EHRSHOT has no tree-count grid; trees are set by early stopping under this cap.
GBM_MAX_TREES = 5_000
GBM_EARLY_STOPPING = 50
# Min train patients per (item, answer) column; filters free text such as reader names.
ANSWER_MIN_TRAIN_PATIENTS = 20
AGE_BIN_YEARS = 5
AGE_BIN_LAST = 100
MALE_CONCEPT_ID = 8507
RESULT_FOLDER = "count_baselines"


def ancestor_matrix(config: dict, vocab: pl.DataFrame) -> tuple[csr_matrix, list[int]]:
    """0/1 matrix of token ID x ancestor concept; token counts @ matrix = ancestor counts."""
    concept_of = dict(zip(vocab["token_id"].to_list(), vocab["concept_id"].to_list()))
    tokens = vocab.select(
        "token_id",
        "domain",
        pl.coalesce(
            pl.col("concept_id"),
            pl.col("base_id").replace_strict(concept_of, default=None, return_dtype=pl.Int64),
        ).alias("base_concept_id"),
    ).filter(pl.col("domain").is_in(["condition", "drug"]) & pl.col("base_concept_id").is_not_null())
    raw_root = config["paths"]["raw_root"]
    concepts = pl.scan_parquet(source_parts(raw_root, "CONCEPT")).select(
        pl.col("concept_id").cast(pl.Int64),
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
    )
    pairs = (
        pl.scan_parquet(source_parts(raw_root, "CONCEPT_ANCESTOR"))
        .filter(pl.col("min_levels_of_separation") > 0)
        .select(
            pl.col("descendant_concept_id").cast(pl.Int64),
            pl.col("ancestor_concept_id").cast(pl.Int64),
        )
        .join(
            tokens.lazy(), left_on="descendant_concept_id", right_on="base_concept_id", how="inner"
        )
        .join(concepts, left_on="ancestor_concept_id", right_on="concept_id", how="inner")
        .filter(
            (
                (pl.col("domain") == "condition")
                & (pl.col("vocabulary_id") == "SNOMED")
                & (pl.col("standard_concept") == "S")
            )
            | (
                (pl.col("domain") == "drug")
                & (
                    (
                        pl.col("vocabulary_id").is_in(["RxNorm", "RxNorm Extension"])
                        & (pl.col("concept_class_id") == "Ingredient")
                    )
                    | (pl.col("vocabulary_id") == "ATC")
                )
            )
        )
        .select("token_id", "ancestor_concept_id")
        .unique()
        .collect()
    )
    ancestors = sorted(pairs["ancestor_concept_id"].unique().to_list())
    column_of = {concept: index for index, concept in enumerate(ancestors)}
    rows = pairs["token_id"].to_numpy()
    columns = np.asarray([column_of[value] for value in pairs["ancestor_concept_id"].to_list()])
    vocab_size = int(vocab["token_id"].max()) + 1
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(vocab_size, len(ancestors))
    )
    print(f"{len(ancestors):,} ancestor concepts, {len(rows):,} token-ancestor links", flush=True)
    return matrix, ancestors


def token_count_matrix(data: pl.DataFrame, vocab_size: int, ignored: list[int]) -> csr_matrix:
    """Sparse token counts per sample, without the ignored IDs."""
    lengths = data["input_tokens"].list.len().to_numpy()
    tokens = data["input_tokens"].explode().to_numpy().astype(np.int32)
    rows = np.repeat(np.arange(len(data), dtype=np.int32), lengths)
    if tokens.min() < 0 or tokens.max() >= vocab_size:
        raise ValueError("token ID outside the vocabulary")
    keep = ~np.isin(tokens, ignored)
    counts = coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], tokens[keep])),
        shape=(len(data), vocab_size),
    ).tocsr()
    counts.sum_duplicates()
    return counts


def answer_events(config: dict) -> tuple[pl.DataFrame, list[str]]:
    """Observation/measurement answers the tokenizer discards, as (person, date, column) plus column keys.

    Rows with an answer concept or result text but no numeric or NEG/POS value; rare (item, answer) pairs are dropped.
    """
    frames = []
    for domain in ("observation", "measurement"):
        answer_concept = pl.col("value_as_concept_id").fill_null(0)
        frames.append(
            pl.scan_parquet(output_file(config, "results", f"{domain}.parquet"))
            .filter(
                pl.col("numeric_value_raw").is_null()
                & pl.col("qual_result").is_null()
                & (pl.col("concept_id") > 0)
                & pl.col("event_hold_reason").is_null()
            )
            .with_columns(pl.col("result_text").fill_null("").str.strip_chars().str.to_uppercase().alias("text"))
            .filter((answer_concept > 0) | (pl.col("text") != ""))
            .select(
                "person_id",
                "split",
                "event_date",
                pl.when(answer_concept > 0)
                .then(pl.format("{}:{}={}", pl.lit(domain), "concept_id", "value_as_concept_id"))
                .otherwise(pl.format("{}:{}='{}'", pl.lit(domain), "concept_id", "text"))
                .alias("key"),
            )
        )
    events = pl.concat(frames).collect()
    kept = (
        events.filter(pl.col("split") == "train")
        .group_by("key")
        .agg(pl.col("person_id").n_unique().alias("train_patients"))
        .filter(pl.col("train_patients") >= ANSWER_MIN_TRAIN_PATIENTS)
        .sort("key")
        .with_row_index("column")
    )
    used = events.join(kept.select("key", "column"), on="key", how="inner").select("person_id", "event_date", "column")
    print(
        f"{events.height:,} discarded answer rows, {used.height:,} used, {kept.height:,} (item, answer) columns",
        flush=True,
    )
    return used, kept["key"].to_list()


def answer_count_matrix(data: pl.DataFrame, events: pl.DataFrame, n_columns: int) -> csr_matrix:
    """Counts of (item, answer) pairs per sample up to the day before the cutoff.

    Same-day answers leak the future: discharge-status answers carry the admission date, cultures the specimen date.
    """
    pairs = (
        data.select("person_id", "cutoff_date")
        .with_row_index("row")
        .lazy()
        .join(events.lazy(), on="person_id", how="inner")
        .filter(pl.col("event_date") < pl.col("cutoff_date"))
        .group_by("row", "column")
        .len()
        .collect(engine="streaming")
    )
    return csr_matrix(
        (pairs["len"].to_numpy().astype(np.float32), (pairs["row"].to_numpy(), pairs["column"].to_numpy())),
        shape=(len(data), n_columns),
    )


def age_at_cutoff(data: pl.DataFrame, person: pl.DataFrame) -> pl.DataFrame:
    """Age in full years at the cutoff (birth month aware) and a male indicator."""
    joined = data.select("person_id", "cutoff_date").join(person, on="person_id", how="left")
    if joined["year_of_birth"].null_count():
        raise ValueError("some samples have no year of birth")
    return joined.select(
        (
            pl.col("cutoff_date").dt.year()
            - pl.col("year_of_birth")
            - (pl.col("cutoff_date").dt.month() < pl.col("month_of_birth")).cast(pl.Int32)
        )
        .clip(0, None)
        .alias("age"),
        (pl.col("gender_concept_id") == MALE_CONCEPT_ID).cast(pl.Float32).alias("male"),
    )


def age_sex_matrix(demographics: pl.DataFrame) -> csr_matrix:
    """One-hot 5-year age bins (100+ pooled) and a male column."""
    bins = np.minimum(demographics["age"].to_numpy() // AGE_BIN_YEARS, AGE_BIN_LAST // AGE_BIN_YEARS)
    n_bins = AGE_BIN_LAST // AGE_BIN_YEARS + 1
    one_hot = csr_matrix(
        (np.ones(len(bins), dtype=np.float32), (np.arange(len(bins)), bins)), shape=(len(bins), n_bins)
    )
    return hstack([one_hot, csr_matrix(demographics["male"].to_numpy().reshape(-1, 1))]).tocsr()


def fit_logistic(features: csr_matrix, labels: np.ndarray, train: np.ndarray, validation: np.ndarray, name: str):
    """Pick C with the same LR setup as the embedding evaluation."""
    tuning = []
    best = (-1.0, None, None)
    for c_value in C_VALUES:
        model = make_pipeline(
            MaxAbsScaler(),
            LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", class_weight=None, max_iter=1000),
        )
        model.fit(features[train], labels[train])
        val_scores = scores(labels[validation], model.predict_proba(features[validation])[:, 1])
        tuning.append({"C": c_value, **val_scores})
        print(f"{name}: C={c_value}, validation AUROC={val_scores['auroc']:.4f}", flush=True)
        if val_scores["auroc"] > best[0]:
            best = (val_scores["auroc"], c_value, model)
    _, best_c, best_model = best
    return best_model, {"C": best_c}, tuning


def fit_gbm(
    features: csr_matrix, labels: np.ndarray, train: np.ndarray, validation: np.ndarray, name: str, threads: int
):
    """Compare the 27 EHRSHOT grid points by validation ROC AUC; tree count by early stopping."""
    tuning = []
    best = (-1.0, None, None)
    x_train, y_train = features[train], labels[train]
    x_val, y_val = features[validation], labels[validation]
    for learning_rate, max_depth, num_leaves in itertools.product(*GBM_GRID.values()):
        params = {"learning_rate": learning_rate, "max_depth": max_depth, "num_leaves": num_leaves}
        model = lightgbm.LGBMClassifier(
            **params,
            n_estimators=GBM_MAX_TREES,
            n_jobs=threads,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[lightgbm.early_stopping(GBM_EARLY_STOPPING, verbose=False)],
        )
        val_scores = scores(y_val, model.predict_proba(x_val, num_iteration=model.best_iteration_)[:, 1])
        tuning.append({**params, "trees": int(model.best_iteration_), **val_scores})
        print(
            f"{name}: {params}, {model.best_iteration_} trees, validation AUROC={val_scores['auroc']:.4f}",
            flush=True,
        )
        if val_scores["auroc"] > best[0]:
            best = (val_scores["auroc"], {**params, "trees": int(model.best_iteration_)}, model)
    _, best_params, best_model = best
    return best_model, best_params, tuning


def evaluate_count_task(
    config: dict,
    task: str,
    models: list[str],
    vocab: pl.DataFrame,
    ancestors: csr_matrix | None,
    person: pl.DataFrame,
    threads: int,
    answers: tuple[pl.DataFrame, list[str]] | None = None,
) -> list[dict]:
    pattern = str(output_file(config, "cohorts", "inputs", "bucket=*.parquet"))
    data = (
        pl.scan_parquet(pattern)
        .filter(pl.col("task") == task)
        .select("person_id", "index_visit_id", "cutoff_date", "split", "label", "input_tokens")
        .collect()
    )
    if data.select("person_id", "index_visit_id", "cutoff_date").is_duplicated().any():
        raise ValueError(f"{task}: duplicate index samples")
    labels = data["label"].to_numpy().astype(np.int8)
    splits = data["split"].to_numpy()
    patients = data["person_id"].to_numpy()
    train, validation, test = (splits == "train"), (splits == "validation"), (splits == "test")
    if not (train | validation | test).all():
        raise ValueError(f"{task}: unknown split")
    demographics = age_at_cutoff(data, person)
    vocab_size = int(vocab["token_id"].max()) + 1
    ids = special_ids(output_file(config, "tokens", "vocab.parquet"))
    ignored = [ids["[BOS]"], ids["[EOS]"], ids["[PAD]"]]

    count_features = None
    if {"count_lr", "count_gbm", "count_answers_gbm"} & set(models):
        counts = token_count_matrix(data, vocab_size, ignored)
        ancestor_counts = counts @ ancestors
        dense = demographics.select(pl.col("age").cast(pl.Float32), "male").to_numpy()
        count_features = hstack([counts, ancestor_counts, csr_matrix(dense)]).tocsr().astype(np.float32)
        print(
            f"{task}: {len(data):,} samples, {count_features.shape[1]:,} features "
            f"({counts.shape[1]:,} tokens + {ancestor_counts.shape[1]:,} ancestors + age/sex 2), "
            f"{count_features.nnz:,} nonzeros",
            flush=True,
        )
    answer_features = None
    if "count_answers_gbm" in models:
        events, keys = answers
        answer_counts = answer_count_matrix(data, events, len(keys))
        answer_features = hstack([count_features, answer_counts]).tocsr().astype(np.float32)
        print(
            f"{task}: added {len(keys):,} discarded-answer columns, samples with any answer: "
            f"{int((answer_counts.getnnz(axis=1) > 0).sum()):,}",
            flush=True,
        )

    results = []
    for model_name in models:
        name = f"{task}/{model_name}"
        if model_name == "age_sex_lr":
            features = age_sex_matrix(demographics)
            model, chosen, tuning = fit_logistic(features, labels, train, validation, name)
            test_probability = model.predict_proba(features[test])[:, 1]
            description = "age at cutoff in 5-year bins (one-hot, 100+ pooled) + male indicator"
        elif model_name == "count_lr":
            features = count_features
            model, chosen, tuning = fit_logistic(features, labels, train, validation, name)
            test_probability = model.predict_proba(features[test])[:, 1]
            description = "token counts + SNOMED/RxNorm-ingredient/ATC ancestor counts + age + male"
        else:
            features = count_features if model_name == "count_gbm" else answer_features
            model, chosen, tuning = fit_gbm(features, labels, train, validation, name, threads)
            test_probability = model.predict_proba(features[test], num_iteration=model.best_iteration_)[:, 1]
            description = "token counts + SNOMED/RxNorm-ingredient/ATC ancestor counts + age + male"
            if model_name == "count_answers_gbm":
                description += " + counts of discarded (item, answer) pairs from observation/measurement"
        test_patients = patients[test]
        test_scores = {
            **scores(labels[test], test_probability),
            "auroc_patient_weighted": float(
                roc_auc_score(labels[test], test_probability, sample_weight=patient_weights(test_patients))
            ),
            **patient_bootstrap(labels[test], test_probability, test_patients),
        }
        best_validation = max(tuning, key=lambda row: row["auroc"])
        results.append(
            {
                "task": task,
                "model": model_name,
                "features": description,
                "n_features": int(features.shape[1]),
                "foundation_model_used": False,
                "selected": chosen,
                "selection_metric": "validation ROC AUC",
                "train": split_counts(labels[train], patients[train]),
                "validation": {**split_counts(labels[validation], patients[validation]), **best_validation},
                "test": {**split_counts(labels[test], test_patients), **test_scores},
                "validation_tuning": tuning,
            }
        )
        data.filter(pl.Series(test)).select("person_id", "index_visit_id", "cutoff_date", "label").with_columns(
            pl.Series("probability", test_probability)
        ).write_parquet(output_file(config, "evaluation", RESULT_FOLDER, f"{model_name}_{task}_test_predictions.parquet"))
        joblib.dump(model, output_file(config, "evaluation", RESULT_FOLDER, f"{model_name}_{task}_model.joblib"))
        print(
            f"{name}: test AUROC={test_scores['auroc']:.4f} {test_scores['auroc_ci95']}, "
            f"AUPRC={test_scores['auprc']:.4f}",
            flush=True,
        )
    return results


def run_logistic(
    config_path: str, tasks: list[str], initialization: str = "pretrained", run_name: str | None = None
) -> None:
    """Fit and evaluate logistic regression per task; merge into results.json."""
    output_folder = result_folder_parts("evaluation", initialization, run_name)
    config = load_config(config_path)
    results = [evaluate_logistic_task(config, task, initialization, run_name) for task in tasks]
    results_path = output_file(config, *output_folder, "results.json")
    if results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        results = list({row["task"]: row for row in previous["tasks"] + results}.values())
    write_json(results_path, {"tasks": results})


def run_count_baselines(config_path: str, tasks: list[str], models: list[str], threads: int) -> None:
    """Fit and evaluate count baselines per task; merge into results.json."""
    config = load_config(config_path)
    vocab = pl.read_parquet(output_file(config, "tokens", "vocab.parquet"))
    person = pl.read_parquet(output_file(config, "demographics", "person.parquet"))
    ancestors = None
    if {"count_lr", "count_gbm", "count_answers_gbm"} & set(models):
        ancestors, _ = ancestor_matrix(config, vocab)
    answers = answer_events(config) if "count_answers_gbm" in models else None
    results_path = output_file(config, "evaluation", RESULT_FOLDER, "results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))["tasks"] if results_path.exists() else []
    for task in tasks:
        new_rows = evaluate_count_task(config, task, models, vocab, ancestors, person, threads, answers)
        done = {(row["task"], row["model"]) for row in new_rows}
        results = [row for row in results if (row["task"], row["model"]) not in done] + new_rows
        write_json(results_path, {"tasks": results})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    for stage, help_text in [("embed", "extract frozen GPT-2 embeddings"), ("logistic", "L2 logistic regression on embeddings")]:
        sub = stages.add_parser(stage, parents=[common_args], help=help_text)
        sub.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
        sub.add_argument("--initialization", choices=["pretrained", "random_init", "continued"], default="pretrained")
        sub.add_argument("--run-name", help="continued-training result folder, e.g. epoch20")
    counts = stages.add_parser("count-baselines", parents=[common_args], help="code-count baselines without pretraining")
    counts.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    counts.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    counts.add_argument("--threads", type=int, default=24, help="LightGBM threads")
    args = parser.parse_args()
    if args.stage == "embed":
        run_embed(args.config, args.tasks, args.initialization, args.run_name)
    elif args.stage == "logistic":
        run_logistic(args.config, args.tasks, args.initialization, args.run_name)
    else:
        run_count_baselines(args.config, args.tasks, args.models, args.threads)


if __name__ == "__main__":
    main()
