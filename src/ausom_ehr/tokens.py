"""Token vocabulary, per-event tokens and per-patient training sequences.

Stages: vocab, events (base folder); link-base, joint-vocab, joint-events, sequences (model folder)
Usage: python -m ausom_ehr.tokens <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ausom_ehr.common import (
    config_parent, copy_reference_file, export_finish_date, load_config, output_file,
    reference_root, source_parts, write_json,
)
from ausom_ehr.data import (
    DOMAINS, check_event_exceptions, clean_event_rows, excluded_concept_filter,
)
from ausom_ehr.values import check_parsing_exceptions, check_unit_exceptions


# ── Vocabulary and event tokens ──

SPECIAL_TOKENS = [
    "[PAD]",
    "[BOS]",
    "[EOS]",
    "[LOW]",
    "[HIGH]",
    "[NEG]",
    "[POS]",
    "[NUMERIC_UNSEEN]",
]
SPECIAL_TOKENS += [f"[Q{i}]" for i in range(1, 11)]
SPECIAL_TOKENS += ["[DAY_GAP_0]"]
SPECIAL_TOKENS += [f"[DAY_GAP_{i}]" for i in range(1, 31)]
SPECIAL_TOKENS += [
    "[DAY_GAP_31_60]",
    "[DAY_GAP_61_90]",
    "[DAY_GAP_91_180]",
    "[DAY_GAP_181_365]",
    "[DAY_GAP_LT]",
]
# Sex and 5-year age band at the first input date (100+ is one band).
SPECIAL_TOKENS += ["[SEX_M]", "[SEX_F]", "[SEX_UNKNOWN]"]
SPECIAL_TOKENS += [f"[AGE_{start}_{start + 4}]" for start in range(0, 100, 5)]
SPECIAL_TOKENS += ["[AGE_100_PLUS]", "[AGE_UNKNOWN]"]


def build_vocab(config_path: str) -> None:
    """Vocabulary of concepts seen in train, validation and test patients, plus a concept-name file."""
    config = load_config(config_path)
    check_event_exceptions(config)
    raw_root: Path = config["paths"]["raw_root"]
    export_date = export_finish_date(raw_root)
    destination = output_file(config, "tokens", "vocab.parquet")
    names_path = output_file(config, "tokens", "concept_names.parquet")
    if destination.exists():
        raise FileExistsError(f"Token vocabulary already exists: {destination}")
    reference = reference_root(config)
    if reference is not None:
        # New site: reuse the M0 vocabulary; events with unknown concepts are dropped in the events stage.
        for name in ["vocab.parquet", "concept_names.parquet", "vocab_summary.json"]:
            copy_reference_file(reference, config, "tokens", name)
        print(f"Using reference vocabulary: {reference / 'tokens' / 'vocab.parquet'}", flush=True)
        return

    concept_scans = []
    for domain in DOMAINS:
        source = output_file(config, "events", f"{domain}.parquet")
        if not source.exists():
            raise FileNotFoundError(f"Run event extraction first: {source}")
        concept_scans.append(
            pl.scan_parquet(source)
            .filter(
                (pl.col("concept_id") > 0)
                & excluded_concept_filter(domain)
                & pl.col("event_date").is_not_null()
                & pl.col("datetime_date").is_not_null()
                & ~pl.col("date_mismatch")
                & (pl.col("event_date") <= pl.lit(export_date).str.to_date())
            )
            .select("domain", "concept_id")
            .unique()
        )

    concepts = (
        pl.concat(concept_scans)
        .unique()
        .collect(engine="streaming")
        .with_columns(
            pl.col("domain")
            .replace_strict({name: i for i, name in enumerate(DOMAINS)})
            .alias("domain_order")
        )
        .sort("domain_order", "concept_id")
    )
    records = [
        {"token_id": i, "token": token, "domain": None, "concept_id": None}
        for i, token in enumerate(SPECIAL_TOKENS)
    ]
    for offset, row in enumerate(concepts.iter_rows(named=True), start=len(records)):
        records.append(
            {
                "token_id": offset,
                "token": f"{row['domain'].upper()}:{row['concept_id']}",
                "domain": row["domain"],
                "concept_id": row["concept_id"],
            }
        )

    vocab = pl.DataFrame(records).with_columns(
        pl.col("token_id").cast(pl.Int32),
        pl.col("concept_id").cast(pl.Int64),
    )
    vocab.write_parquet(destination, compression="zstd")

    used_ids = concepts["concept_id"].unique().to_list()
    descriptions = (
        pl.scan_parquet(source_parts(raw_root, "CONCEPT"))
        .filter(pl.col("concept_id").is_in(used_ids))
        .select(pl.col("concept_id").cast(pl.Int64), "concept_name", "vocabulary_id")
        .collect(engine="streaming")
    )
    concepts.select("domain", "concept_id").join(
        descriptions, on="concept_id", how="left"
    ).write_parquet(names_path, compression="zstd")

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(
        output_file(config, "tokens", "vocab_summary.json"),
        {
            "special_token_count": len(SPECIAL_TOKENS),
            "concept_token_count": concepts.height,
            "total_vocab_size": vocab.height,
            "construction_splits": ["train", "validation", "test"],
            "validation_test_concept_types_used": True,
            "uses_label_information": False,
            "vocab_sha256": digest,
        },
    )
    print(f"Saved {vocab.height:,} tokens: {destination}", flush=True)


BUCKET_COUNT = 128


def special_ids(vocab_path: Path) -> dict[str, int]:
    """Name-to-ID map of the non-concept tokens in the vocabulary."""
    rows = (
        pl.read_parquet(vocab_path)
        .filter(pl.col("domain").is_null())
        .select("token", "token_id")
        .to_dicts()
    )
    return {row["token"]: int(row["token_id"]) for row in rows}


def quantile_token_expression(ids: dict[str, int]) -> pl.Expr:
    """Q token ID of numeric_value; a value equal to a cut point goes to the higher Q."""
    crossed = [
        (pl.col("numeric_value") >= pl.col(f"d{i}")).fill_null(False).cast(pl.Int32)
        for i in range(1, 10)
    ]
    q_number = pl.sum_horizontal(*crossed) + 1
    return (
        pl.when(q_number == 1)
        .then(pl.lit(ids["[Q1]"]))
        .when(q_number == 2)
        .then(pl.lit(ids["[Q2]"]))
        .when(q_number == 3)
        .then(pl.lit(ids["[Q3]"]))
        .when(q_number == 4)
        .then(pl.lit(ids["[Q4]"]))
        .when(q_number == 5)
        .then(pl.lit(ids["[Q5]"]))
        .when(q_number == 6)
        .then(pl.lit(ids["[Q6]"]))
        .when(q_number == 7)
        .then(pl.lit(ids["[Q7]"]))
        .when(q_number == 8)
        .then(pl.lit(ids["[Q8]"]))
        .when(q_number == 9)
        .then(pl.lit(ids["[Q9]"]))
        .otherwise(pl.lit(ids["[Q10]"]))
    )


def build_domain_tokens(
    config: dict,
    domain: str,
    vocab: pl.LazyFrame,
    boundaries: pl.LazyFrame,
    ids: dict[str, int],
    export_date: str,
) -> pl.LazyFrame:
    """Per-event token lists (concept, numeric state, NEG/POS) for one domain."""
    if domain in ["measurement", "observation"]:
        source = output_file(config, "units", f"{domain}.parquet")
    else:
        source = output_file(config, "events", f"{domain}.parquet")
    if not source.exists():
        raise FileNotFoundError(f"Preprocessed input not found: {source}")

    valid_identity = pl.col("concept_id") > 0
    rows = pl.scan_parquet(source)
    if domain not in ["measurement", "observation"]:
        # Measurement and observation rows were already cleaned in the values stage.
        rows = clean_event_rows(rows, domain)
    data = (
        rows
        .filter(
            valid_identity
            & excluded_concept_filter(domain)
            & pl.col("event_date").is_not_null()
            & pl.col("datetime_date").is_not_null()
            & ~pl.col("date_mismatch")
            & (pl.col("event_date") <= pl.lit(export_date).str.to_date())
        )
        .join(vocab, on=["domain", "concept_id"], how="left")
    )
    if reference_root(config) is not None:
        # Drop whole events whose concept is not in the reference vocabulary (no value-only events).
        data = data.filter(pl.col("token_id").is_not_null())

    if domain in ["measurement", "observation"]:
        data = data.join(boundaries, on=["domain", "concept_id", "unit_group"], how="left")
        usable_number = (
            pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_finite()
        )
        # LOW/HIGH only where the units stage marked the range usable; other numbers get train deciles.
        usable_range = usable_number & pl.col("range_usable").fill_null(False)
        quantile_id = quantile_token_expression(ids)
        data = data.with_columns(
            pl.when(~usable_number)
            .then(pl.lit("no_number"))
            .when(usable_range & (pl.col("range_status") == "LOW"))
            .then(pl.lit("range_low"))
            .when(usable_range & (pl.col("range_status") == "HIGH"))
            .then(pl.lit("range_high"))
            .when(usable_range)
            .then(pl.lit("range_normal"))
            .when(pl.col("numeric_group_seen").is_null())
            .then(pl.lit("numeric_unseen"))
            .otherwise(pl.lit("decile"))
            .alias("numeric_representation"),
            pl.when(~usable_number)
            .then(pl.lit(None, dtype=pl.Int32))
            .when(usable_range & (pl.col("range_status") == "LOW"))
            .then(pl.lit(ids["[LOW]"]))
            .when(usable_range & (pl.col("range_status") == "HIGH"))
            .then(pl.lit(ids["[HIGH]"]))
            .when(usable_range)
            .then(pl.lit(None, dtype=pl.Int32))
            .when(pl.col("numeric_group_seen").is_null())
            .then(pl.lit(ids["[NUMERIC_UNSEEN]"]))
            .otherwise(quantile_id)
            .cast(pl.Int32)
            .alias("numeric_token_id"),
            pl.when(pl.col("qual_result") == "NEG")
            .then(pl.lit(ids["[NEG]"]))
            .when(pl.col("qual_result") == "POS")
            .then(pl.lit(ids["[POS]"]))
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .cast(pl.Int32)
            .alias("qual_token_id"),
        )
    else:
        data = data.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("numeric_token_id"),
            pl.lit(None, dtype=pl.Int32).alias("qual_token_id"),
            pl.lit("not_applicable").alias("numeric_representation"),
        )

    data = data.with_columns(
        pl.concat_list("token_id", "numeric_token_id", "qual_token_id")
        .list.drop_nulls()
        .alias("event_tokens"),
        pl.col("domain")
        .replace_strict({name: i for i, name in enumerate(DOMAINS)})
        .cast(pl.Int8)
        .alias("domain_order"),
        (pl.col("person_id") % BUCKET_COUNT).cast(pl.Int16).alias("bucket"),
    )
    return data.select(
        "person_id",
        "split",
        "event_date",
        "source_datetime",
        "domain",
        "concept_id",
        "domain_order",
        "event_id",
        "visit_occurrence_id",
        "condition_status_concept_id",
        "event_tokens",
        pl.col("token_id").alias("concept_token_id"),
        pl.col("event_tokens").list.len().cast(pl.Int8).alias("event_token_count"),
        "numeric_token_id",
        "qual_token_id",
        "numeric_representation",
        "bucket",
    )


def summarize_tokens(config: dict, ids: dict[str, int]) -> list[dict]:
    """Shares of events with numeric, NEG/POS or concept-only tokens; also saves per-Q counts."""
    source = config["paths"]["output_root"] / "tokens" / "events"
    files = sorted(source.glob("bucket=*/*.parquet"))
    if files:
        data = pl.scan_parquet(files)
    else:
        # No events left: write empty stats; the sequences stage reports exclusions and stops.
        data = pl.DataFrame(schema={
            "split": pl.String, "domain": pl.String, "concept_id": pl.Int64,
            "event_token_count": pl.Int8, "numeric_token_id": pl.Int32,
            "qual_token_id": pl.Int32, "numeric_representation": pl.String,
        }).lazy()
    expressions = [
        pl.len().alias("events"),
        pl.sum("event_token_count").alias("event_tokens"),
        (pl.col("event_token_count") == 1).sum().alias("concept_only_events"),
        (pl.col("event_token_count") == 3).sum().alias("three_token_events"),
        (
            pl.col("numeric_representation").is_in([
                "range_low", "range_high", "range_normal", "decile", "numeric_unseen",
            ]) & pl.col("qual_token_id").is_not_null()
        ).sum().alias("numeric_and_qualitative_events"),
        pl.col("numeric_representation").str.starts_with("range_").sum().alias("range_events"),
        (pl.col("qual_token_id") == ids["[NEG]"]).sum().alias("neg_events"),
        (pl.col("qual_token_id") == ids["[POS]"]).sum().alias("pos_events"),
    ]
    for name in ["range_low", "range_high", "range_normal", "decile", "numeric_unseen", "no_number"]:
        expressions.append((pl.col("numeric_representation") == name).sum().alias(f"{name}_events"))
    counts = (
        data.group_by("split", "domain")
        .agg(*expressions)
        .sort("split", "domain")
        .collect(engine="streaming")
    )
    records = counts.to_dicts()
    for row in records:
        event_count = row["events"]
        for name, count in list(row.items()):
            if name.endswith("_events"):
                row[name.replace("_events", "_fraction")] = count / event_count if event_count else 0.0

    # All of Q1-Q10 as columns, so empty bins from tied cut points stay visible.
    q_counts = data.filter(pl.col("domain").is_in(["measurement", "observation"]))
    q_counts = q_counts.group_by("split", "domain", "concept_id").agg(*[
        (pl.col("numeric_token_id") == ids[f"[Q{i}]"]).sum().alias(f"q{i}_events")
        for i in range(1, 11)
    ])
    q_counts.sink_parquet(
        output_file(config, "tokens", "quantile_counts.parquet"),
        compression="zstd", engine="streaming",
    )
    counts.write_parquet(output_file(config, "tokens", "representation_counts.parquet"), compression="zstd")
    return records


def tokenize_events(config_path: str) -> None:
    config = load_config(config_path)
    check_event_exceptions(config)
    check_parsing_exceptions(config)
    check_unit_exceptions(config)
    vocab_path = output_file(config, "tokens", "vocab.parquet")
    boundary_path = output_file(config, "deciles", "boundaries.parquet")
    if not vocab_path.exists() or not boundary_path.exists():
        raise FileNotFoundError("Build the token vocabulary and train-only decile boundaries first.")

    ids = special_ids(vocab_path)
    vocabulary = (
        pl.scan_parquet(vocab_path)
        .filter(pl.col("concept_id").is_not_null())
        .select("domain", "concept_id", "token_id")
    )
    boundaries = pl.scan_parquet(boundary_path).select(
        "domain", "concept_id", "unit_group", "numeric_group_seen", *[f"d{i}" for i in range(1, 10)]
    )
    export_date = export_finish_date(config["paths"]["raw_root"])
    scans = [
        build_domain_tokens(config, domain, vocabulary, boundaries, ids, export_date)
        for domain in DOMAINS
    ]
    missing_concepts = pl.concat(scans).select(
        pl.col("concept_token_id").is_null().sum().alias("missing_concept_tokens")
    ).collect(engine="streaming").item()
    if missing_concepts:
        raise ValueError(f"{missing_concepts:,} valid event rows are missing from the token vocabulary.")

    destination = config["paths"]["output_root"] / "tokens" / "events"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Token event buckets already exist: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    # Buckets are not date-sorted; the sequences stage sorts by patient, date, domain and PK.
    pl.concat(scans).sink_parquet(
        pl.PartitionBy(destination, key="bucket", include_key=True),
        compression="zstd",
        engine="streaming",
    )
    counts = summarize_tokens(config, ids)
    write_json(
        output_file(config, "tokens", "event_summary.json"),
        {
            "bucket_count": BUCKET_COUNT,
            "source_splits": ["train", "validation", "test"],
            "same_day_order": "domain_order then source event_id",
            "event_token_order": "concept, numeric status if any, qualitative status if any",
            "normal_token_present": False,
            "tokenization_mode": config.get("tokenization", {}).get("mode", "original"),
            "numeric_and_qualitative_results_kept_together": True,
            "total_events": sum(row["events"] for row in counts),
            "by_split_domain": counts,
            "quantile_counts_path": "tokens/quantile_counts.parquet",
        },
    )
    print(f"Wrote {BUCKET_COUNT} event token buckets: {destination}", flush=True)


# ── Joint tokens (concept + value state as one ID) ──

def link_read_only_inputs(config: dict) -> None:
    """Symlink the unchanged sample, events, results and labels from the base output."""
    root: Path = config["paths"]["output_root"]
    base = Path(config["paths"]["base_output_root"]).resolve()
    if root == base or not (base / "tokens" / "vocab.parquet").is_file():
        raise ValueError("Needs a separate output folder and a finished base token vocabulary.")
    names = ["sample", "events", "units", "results", "deciles"]
    if (base / "demographics").exists():
        names.append("demographics")
    for name in names:
        target = base / name
        link = root / name
        if not target.exists():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"linked to a different input: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(target, target_is_directory=True)
    cohort_dir = root / "cohorts"
    cohort_dir.mkdir(exist_ok=True)
    cohort_names = ["visits.parquet", "candidates.parquet"]
    if (base / "cohorts" / "labels_summary.json").is_file():
        cohort_names.append("labels_summary.json")
    for name in cohort_names:
        target = base / "cohorts" / name
        link = cohort_dir / name
        if not target.is_file():
            raise FileNotFoundError(target)
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"linked to a different cohort: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(target)
    print(f"Linked read-only shared inputs: {root}", flush=True)


def value_keys(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Join keys for concept+value states, with -1 for a missing state (NULL never joins)."""
    return frame.with_columns(
        pl.col("event_tokens").list.first().cast(pl.Int32).alias("base_id"),
        pl.col("numeric_token_id").fill_null(-1).cast(pl.Int32).alias("number_id"),
        pl.col("qual_token_id").fill_null(-1).cast(pl.Int32).alias("quality_id"),
    )


def build_joint_vocab(config: dict) -> None:
    """Add only the observed concept+value-state pairs to the vocabulary."""
    source_root = Path(config["paths"]["source_variant_root"])
    destination = output_file(config, "tokens", "vocab.parquet")
    if destination.exists():
        print(f"Reusing existing vocabulary: {destination}", flush=True)
        return
    reference = reference_root(config, "variant_root")
    if reference is not None:
        # New site: reuse the M0 joint vocabulary; unknown pairs stay as separate concept and value tokens.
        for name in ["vocab.parquet", "vocab_summary.json"]:
            copy_reference_file(reference, config, "tokens", name)
        print(f"Using reference joint vocabulary: {reference / 'tokens' / 'vocab.parquet'}", flush=True)
        return
    original = pl.read_parquet(source_root / "tokens" / "vocab.parquet")
    if "source_code" not in original.columns:
        original = original.with_columns(pl.lit(None, dtype=pl.String).alias("source_code"))
    unique_parts = []
    for bucket in range(BUCKET_COUNT):
        files = sorted((source_root / "tokens" / "events" / f"bucket={bucket}").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"Source variant bucket {bucket} not found.")
        part = (
            value_keys(pl.scan_parquet(files))
            .filter((pl.col("number_id") != -1) | (pl.col("quality_id") != -1))
            .select("base_id", "number_id", "quality_id")
            .unique()
            .collect(engine="streaming")
        )
        unique_parts.append(part)
    combinations = pl.concat(unique_parts).unique().sort("base_id", "number_id", "quality_id")
    lookup = dict(zip(original["token_id"], original["token"]))
    domains = dict(zip(original["token_id"], original["domain"]))
    records = []
    for offset, (base_id, number_id, quality_id) in enumerate(combinations.iter_rows()):
        parts = [lookup[base_id]]
        if number_id != -1:
            parts.append(lookup[number_id])
        if quality_id != -1:
            parts.append(lookup[quality_id])
        records.append({
            "token_id": original.height + offset,
            "token": "+".join(parts),
            "domain": domains[base_id],
            "concept_id": None,
            "source_code": None,
            "base_id": base_id,
            "number_id": number_id,
            "quality_id": quality_id,
        })
    base = original.with_columns(
        pl.lit(None, dtype=pl.Int32).alias("base_id"),
        pl.lit(None, dtype=pl.Int32).alias("number_id"),
        pl.lit(None, dtype=pl.Int32).alias("quality_id"),
    )
    additions = pl.DataFrame(records, schema={
        "token_id": pl.Int32, "token": pl.String, "domain": pl.String,
        "concept_id": pl.Int64, "source_code": pl.String, "base_id": pl.Int32,
        "number_id": pl.Int32, "quality_id": pl.Int32,
    })
    vocabulary = pl.concat([base, additions])
    vocabulary.write_parquet(destination, compression="zstd")
    write_json(output_file(config, "tokens", "vocab_summary.json"), {
        "mode": "concept_value_joint",
        "base_vocab_size": original.height,
        "joint_value_tokens": additions.height,
        "total_vocab_size": vocabulary.height,
        "equality_rule": "exact base concept/source token plus numeric and qualitative state IDs",
        "vocab_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    })
    print(f"{additions.height:,} concept+value tokens, vocabulary size {vocabulary.height:,}", flush=True)


def convert_joint_events(config: dict) -> None:
    """Keep source events as they are and replace each known value pair with its joint ID."""
    source_root = Path(config["paths"]["source_variant_root"])
    vocab = pl.scan_parquet(output_file(config, "tokens", "vocab.parquet"))
    mapping = vocab.filter(pl.col("base_id").is_not_null()).select(
        "base_id", "number_id", "quality_id", pl.col("token_id").alias("joint_id")
    )
    root = config["paths"]["output_root"] / "tokens" / "events"
    for bucket in range(BUCKET_COUNT):
        destination = root / f"bucket={bucket}" / "part-0.parquet"
        if destination.exists():
            print(f"Joint token bucket {bucket}: reused", flush=True)
            continue
        source_files = sorted((source_root / "tokens" / "events" / f"bucket={bucket}").glob("*.parquet"))
        if not source_files:
            raise FileNotFoundError(f"Source bucket {bucket} not found.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = pl.scan_parquet(source_files)
        schema = original.collect_schema().names()
        converted = value_keys(original).join(
            mapping, on=["base_id", "number_id", "quality_id"], how="left", maintain_order="left"
        ).with_columns(
            pl.when(pl.col("joint_id").is_not_null())
            .then(pl.concat_list("joint_id"))
            .otherwise(pl.col("event_tokens"))
            .alias("event_tokens"),
            pl.when(pl.col("joint_id").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.col("event_token_count"))
            .cast(pl.Int8).alias("event_token_count"),
        ).select(schema)
        temporary = destination.with_name(destination.name + ".tmp")
        converted.sink_parquet(temporary, compression="zstd", engine="streaming")
        temporary.replace(destination)
        if bucket % 16 == 0:
            print(f"Joint token bucket {bucket}/{BUCKET_COUNT}", flush=True)
    write_json(output_file(config, "tokens", "event_summary.json"), {
        "bucket_count": BUCKET_COUNT,
        "tokenization_mode": "concept_value_joint",
        "source_event_rows_preserved": True,
        "event_token_order": "one concept/source-and-value joint ID per event; day gaps separate",
    })


# ── Patient sequences ──

MAX_TOKENS = 16_384
GENDER_TOKENS = {8507: "[SEX_M]", 8532: "[SEX_F]"}
DEMOGRAPHIC_TOKEN_COUNT = 2


def uses_demographic_tokens(config: dict) -> bool:
    return bool(config.get("tokenization", {}).get("demographic_tokens", False))


def load_demographics(config: dict, bucket: int | None = None) -> dict[int, tuple]:
    """person_id -> (gender_concept_id, year_of_birth, month_of_birth), optionally for one bucket."""
    path = output_file(config, "demographics", "person.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Demographics file not found; run data demographics first: {path}")
    data = pl.scan_parquet(path)
    if bucket is not None:
        data = data.filter(pl.col("person_id") % BUCKET_COUNT == bucket)
    rows = data.select("person_id", "gender_concept_id", "year_of_birth", "month_of_birth").collect()
    return {row[0]: row[1:] for row in rows.iter_rows()}


def age_token(birth_year: int | None, birth_month: int | None, on_date: date, ids: dict[str, int]) -> int:
    """5-year age-band token at on_date, comparing year and month only; negative ages count as 0."""
    if birth_year is None:
        return ids["[AGE_UNKNOWN]"]
    age = on_date.year - birth_year - (1 if on_date.month < (birth_month or 1) else 0)
    age = max(age, 0)
    if age >= 100:
        return ids["[AGE_100_PLUS]"]
    start = age // 5 * 5
    return ids[f"[AGE_{start}_{start + 4}]"]


def demographic_tokens(person: tuple, first_date: date, ids: dict[str, int]) -> list[int]:
    gender, birth_year, birth_month = person
    return [ids[GENDER_TOKENS.get(gender, "[SEX_UNKNOWN]")], age_token(birth_year, birth_month, first_date, ids)]


def context_limit(config: dict) -> int:
    configured = int(config.get("model", {}).get("max_context", MAX_TOKENS))
    if configured == MAX_TOKENS:
        return configured
    raise ValueError("max_context must be 16,384; other lengths are not supported.")


def gap_token(days: int, ids: dict[str, int]) -> int:
    if days < 0:
        raise ValueError(f"Dates out of order: gap of {days} days")
    if days == 0:
        return ids["[DAY_GAP_0]"]
    if days <= 30:
        return ids[f"[DAY_GAP_{days}]"]
    if days <= 60:
        return ids["[DAY_GAP_31_60]"]
    if days <= 90:
        return ids["[DAY_GAP_61_90]"]
    if days <= 180:
        return ids["[DAY_GAP_91_180]"]
    if days <= 365:
        return ids["[DAY_GAP_181_365]"]
    return ids["[DAY_GAP_LT]"]


def flatten_days(
    days: list[tuple[date, list[int], int]], ids: dict[str, int], eos: bool,
    person: tuple | None = None,
) -> list[int]:
    """BOS, sex/age tokens if person is given, DAY_GAP + tokens per day, then EOS if eos."""
    tokens = [ids["[BOS]"]]
    if person is not None and days:
        tokens.extend(demographic_tokens(person, days[0][0], ids))
    previous = None
    for event_date, day_tokens, _event_count in days:
        elapsed = 0 if previous is None else (event_date - previous).days
        tokens.append(gap_token(elapsed, ids))
        tokens.extend(day_tokens)
        previous = event_date
    if eos:
        tokens.append(ids["[EOS]"])
    return tokens


def choose_recent_days(
    days: list[tuple[date, list[int], int]], limit: int, eos: bool, reserved: int = 0,
) -> list[tuple[date, list[int], int]]:
    """Most recent whole days that fit in limit, counting BOS, reserved, day gaps and EOS.

    days is oldest first and so is the result; the first day that does not fit stops the selection.
    """
    if limit < 1 + reserved + int(eos):
        raise ValueError("limit must fit BOS, demographic tokens and the optional EOS.")
    used = 1 + reserved + int(eos)
    chosen_reversed = []
    for day in reversed(days):
        needed = 1 + len(day[1])  # DAY_GAP before the day
        if used + needed > limit:
            break
        used += needed
        chosen_reversed.append(day)
    chosen_reversed.reverse()
    return chosen_reversed


def record_for_patient(
    person_id: int,
    split: str,
    days: list[tuple[date, list[int], int]],
    ids: dict[str, int],
    max_tokens: int = MAX_TOKENS,
    person: tuple | None = None,
) -> dict:
    """Full-history and recent-input tokens of one patient, with excluded day/event/token counts."""
    reserved = DEMOGRAPHIC_TOKEN_COUNT if person is not None else 0
    full = flatten_days(days, ids, eos=True, person=person)
    chosen = choose_recent_days(days, max_tokens, eos=True, reserved=reserved)
    usable = flatten_days(chosen, ids, eos=True, person=person) if chosen else []
    full_events = sum(day[2] for day in days)
    kept_events = sum(day[2] for day in chosen)
    return {
        "person_id": person_id,
        "split": split,
        "full_tokens": full,
        "input_tokens": usable,
        "full_length": len(full),
        "input_length": len(usable),
        "full_days": len(days),
        "input_days": len(chosen),
        "full_events": full_events,
        "input_events": kept_events,
        "excluded_days": len(days) - len(chosen),
        "excluded_events": full_events - kept_events,
        "excluded_tokens": len(full) - len(usable) if chosen else len(full),
        "oversized_day_count": sum(1 + len(day[1]) + 2 + reserved > max_tokens for day in days),
    }


def process_sequence_bucket(config: dict, bucket: int, ids: dict[str, int]) -> Path:
    max_tokens = context_limit(config)
    source_root = config["paths"]["output_root"] / "tokens" / "events"
    if not output_file(config, "tokens", "event_summary.json").exists():
        raise FileNotFoundError("Event token summary not found; check that tokenization finished.")
    source_files = sorted((source_root / f"bucket={bucket}").glob("*.parquet"))
    destination = output_file(config, "sequences", f"bucket={bucket:03d}.parquet")
    if destination.exists():
        print(f"bucket {bucket}: reusing existing sequences", flush=True)
        return destination

    # Same-day order is domain then source PK, not clock time.
    if source_files:
        events = (
            pl.scan_parquet(source_files)
            .select("person_id", "split", "event_date", "domain_order", "event_id", "event_tokens")
            .sort("person_id", "event_date", "domain_order", "event_id")
            .select("person_id", "split", "event_date", "event_tokens")
            .collect(engine="streaming")
        )
    else:
        events = pl.DataFrame(schema={
            "person_id": pl.Int64, "split": pl.String,
            "event_date": pl.Date, "event_tokens": pl.List(pl.Int32),
        })

    schema = pa.schema([
        ("person_id", pa.int64()),
        ("split", pa.string()),
        ("full_tokens", pa.list_(pa.int32())),
        ("input_tokens", pa.list_(pa.int32())),
        ("full_length", pa.int32()),
        ("input_length", pa.int32()),
        ("full_days", pa.int32()),
        ("input_days", pa.int32()),
        ("full_events", pa.int32()),
        ("input_events", pa.int32()),
        ("excluded_days", pa.int32()),
        ("excluded_events", pa.int32()),
        ("excluded_tokens", pa.int32()),
        ("oversized_day_count", pa.int32()),
    ])
    demographics = load_demographics(config, bucket) if uses_demographic_tokens(config) else None
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    batch = []
    person_id = None
    split = None
    days = []
    current_date = None
    current_tokens = []
    current_event_count = 0

    def finish_day() -> None:
        nonlocal current_tokens, current_event_count
        if current_date is not None:
            days.append((current_date, current_tokens, current_event_count))
            current_tokens = []
            current_event_count = 0

    def finish_patient() -> None:
        nonlocal days, batch, current_date
        if person_id is None:
            return
        finish_day()
        person = None
        if demographics is not None:
            person = demographics.get(person_id, (None, None, None))
        batch.append(record_for_patient(person_id, split, days, ids, max_tokens, person))
        days = []
        current_date = None
        if len(batch) >= 256:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch = []

    for row_person, row_split, row_date, row_tokens in events.iter_rows():
        if row_person != person_id:
            finish_patient()
            person_id = row_person
            split = row_split
            current_date = None
        if row_date != current_date:
            finish_day()
            current_date = row_date
        current_tokens.extend(row_tokens)
        current_event_count += 1
    finish_patient()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    writer.close()
    temporary.replace(destination)
    print(f"bucket {bucket:03d}: sequenced {events.height:,} events", flush=True)
    return destination


def summarize_sequences(config: dict) -> None:
    """Length and exclusion stats; stops if any patient's latest day alone exceeds the context."""
    root = config["paths"]["output_root"] / "sequences"
    max_tokens = context_limit(config)
    data = pl.scan_parquet(str(root / "bucket=*.parquet"))
    rows = data.select(
        pl.len().alias("patients_with_events"),
        (pl.col("full_length") > max_tokens).sum().alias("patients_over_context"),
        (pl.col("input_length") == 0).sum().alias("patients_without_input"),
        (pl.col("oversized_day_count") > 0).sum().alias("patients_with_oversized_day"),
        pl.sum("oversized_day_count").alias("oversized_days"),
        # Sum in Int64: totals can overflow Int32.
        pl.col("full_events").cast(pl.Int64).sum().alias("full_events"),
        pl.col("input_events").cast(pl.Int64).sum().alias("input_events"),
        pl.col("full_days").cast(pl.Int64).sum().alias("full_days"),
        pl.col("input_days").cast(pl.Int64).sum().alias("input_days"),
        pl.col("full_length").cast(pl.Int64).sum().alias("full_tokens"),
        pl.col("input_length").cast(pl.Int64).sum().alias("input_tokens"),
        pl.col("excluded_days").cast(pl.Int64).sum().alias("excluded_days"),
        pl.col("excluded_events").cast(pl.Int64).sum().alias("excluded_events"),
        pl.col("excluded_tokens").cast(pl.Int64).sum().alias("excluded_tokens"),
    ).collect().to_dicts()[0]
    by_split = data.group_by("split").agg(
        pl.len().alias("patients_with_events"),
        (pl.col("input_length") > 0).sum().alias("pretrain_usable_patients"),
        pl.col("input_length").cast(pl.Int64).sum().alias("actual_input_tokens"),
        pl.col("input_length").median().alias("median_input_length"),
        pl.col("input_length").quantile(0.25).alias("p25_input_length"),
        pl.col("input_length").quantile(0.75).alias("p75_input_length"),
        pl.col("input_length").quantile(0.9).alias("p90_input_length"),
        pl.col("input_length").quantile(0.95).alias("p95_input_length"),
        pl.col("input_length").quantile(0.99).alias("p99_input_length"),
        pl.col("input_length").min().alias("min_input_length"),
        pl.col("input_length").max().alias("max_input_length"),
        pl.col("full_length").median().alias("median_full_length"),
        pl.col("full_length").quantile(0.25).alias("p25_full_length"),
        pl.col("full_length").quantile(0.75).alias("p75_full_length"),
        pl.col("full_length").quantile(0.9).alias("p90_full_length"),
        pl.col("full_length").quantile(0.95).alias("p95_full_length"),
        pl.col("full_length").quantile(0.99).alias("p99_full_length"),
        pl.col("full_length").min().alias("min_full_length"),
        pl.col("full_length").max().alias("max_full_length"),
    ).collect(engine="streaming")

    selected_path = output_file(config, "sample", "selected_patients.parquet")
    if not selected_path.exists():
        raise FileNotFoundError("Sample list needed for initial split and empty-patient counts.")
    initial = (
        pl.scan_parquet(selected_path).group_by("split")
        .agg(pl.len().alias("initial_patients"))
        .collect(engine="streaming")
    )
    by_split = initial.join(by_split, on="split", how="left").with_columns(
        pl.col("patients_with_events").fill_null(0),
        pl.col("pretrain_usable_patients").fill_null(0),
        pl.col("actual_input_tokens").fill_null(0),
    ).with_columns(
        (pl.col("initial_patients") - pl.col("patients_with_events")).alias("excluded_empty_patients"),
        (pl.col("patients_with_events") - pl.col("pretrain_usable_patients")).alias("patients_without_input"),
    ).sort("split").to_dicts()
    rows["initial_patients"] = sum(row["initial_patients"] for row in by_split)
    rows["excluded_empty_patients"] = sum(row["excluded_empty_patients"] for row in by_split)
    for unit in ["days", "events", "tokens"]:
        full_count = rows[f"full_{unit}"]
        rows[f"excluded_{unit}_fraction"] = rows[f"excluded_{unit}"] / full_count if full_count else 0.0
    rows["patients_over_context_fraction"] = (
        rows["patients_over_context"] / rows["patients_with_events"] if rows["patients_with_events"] else 0.0
    )

    data.select(
        "person_id", "split", "full_length", "input_length",
        "full_days", "input_days", "full_events", "input_events",
        "excluded_days", "excluded_events", "excluded_tokens", "oversized_day_count",
    ).sink_parquet(output_file(config, "sequences", "lengths.parquet"), compression="zstd", engine="streaming")
    write_json(output_file(config, "sequences", "summary.json"), {
        "max_context": max_tokens,
        "whole_day_truncation": True,
        "first_retained_day_gap": "[DAY_GAP_0]",
        "length_distribution_path": "sequences/lengths.parquet",
        "ready_for_training": rows["patients_without_input"] == 0 and rows["patients_with_events"] > 0,
        **rows,
        "by_split": by_split,
    })
    print(f"Patients with events: {rows['patients_with_events']:,}, oversized days: {rows['oversized_days']:,}", flush=True)
    if rows["patients_without_input"]:
        raise ValueError(
            f"{rows['patients_without_input']:,} patients have no input because their latest day exceeds the context. "
            "Check sequences/summary.json and decide a handling rule first."
        )
    if not rows["patients_with_events"]:
        raise ValueError("No patients have events after preprocessing; cannot start training.")


def run_sequences(config_path: str, bucket: int | None = None) -> None:
    """All buckets plus the summary, or only the given bucket."""
    config = load_config(config_path)
    ids = special_ids(output_file(config, "tokens", "vocab.parquet"))
    if bucket is None:
        for number in range(BUCKET_COUNT):
            process_sequence_bucket(config, number, ids)
        summarize_sequences(config)
    else:
        process_sequence_bucket(config, bucket, ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    stages.add_parser("vocab", parents=[common_args], help="token vocabulary (base folder)")
    stages.add_parser("events", parents=[common_args], help="events to integer tokens (base folder)")
    stages.add_parser("link-base", parents=[common_args], help="link base outputs read-only into the model folder")
    stages.add_parser("joint-vocab", parents=[common_args], help="concept+value joint token vocabulary")
    stages.add_parser("joint-events", parents=[common_args], help="convert event tokens to joint tokens")
    sequences = stages.add_parser("sequences", parents=[common_args], help="per-patient training sequences")
    sequences.add_argument("--bucket", type=int)
    args = parser.parse_args()
    if args.stage == "vocab":
        build_vocab(args.config)
    elif args.stage == "events":
        tokenize_events(args.config)
    elif args.stage == "sequences":
        if args.bucket is not None and not 0 <= args.bucket < BUCKET_COUNT:
            sequences.error(f"--bucket must be between 0 and {BUCKET_COUNT - 1}.")
        run_sequences(args.config, args.bucket)
    else:
        config = load_config(args.config)
        if args.stage == "link-base":
            link_read_only_inputs(config)
        elif args.stage == "joint-vocab":
            build_joint_vocab(config)
        else:
            convert_joint_events(config)


if __name__ == "__main__":
    main()
