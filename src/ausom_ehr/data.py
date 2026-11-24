"""Patient sampling and extraction of events, demographics and visits from raw OMOP Parquet.

Stages: sample, events, demographics, visits
Usage: python -m ausom_ehr.data <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from ausom_ehr.common import (
    config_parent, export_finish_date, load_config, output_file, source_fingerprint, source_parts,
    write_json,
)


# ── Sampling and event extraction ──

SAMPLE_SIZE = 300_000
TRAIN_SIZE = 210_000
VALIDATION_SIZE = 45_000
SEED = 42


def choose_patients(person_parts: str) -> tuple[pl.DataFrame, int]:
    """Uniform sample without replacement from PERSON with splits; also returns the unique patient count."""
    people = (
        pl.scan_parquet(person_parts)
        .select(pl.col("person_id").cast(pl.Int64))
        .filter(pl.col("person_id").is_not_null())
        .unique()
        .sort("person_id")
        .collect(engine="streaming")
    )
    all_ids = people["person_id"].to_numpy()
    if len(all_ids) < SAMPLE_SIZE:
        raise ValueError(f"PERSON has fewer than {SAMPLE_SIZE:,} unique patients.")

    random = np.random.default_rng(SEED)
    positions = random.choice(len(all_ids), size=SAMPLE_SIZE, replace=False)
    selected_ids = all_ids[positions]
    random.shuffle(selected_ids)

    split_names = np.empty(SAMPLE_SIZE, dtype="U10")
    split_names[:TRAIN_SIZE] = "train"
    split_names[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE] = "validation"
    split_names[TRAIN_SIZE + VALIDATION_SIZE :] = "test"

    selected = pl.DataFrame({"person_id": selected_ids, "split": split_names})
    return selected.sort("person_id"), len(all_ids)


def validate_patient_split(selected: pl.LazyFrame) -> dict:
    """Check that each patient is in exactly one split and split sizes match; returns the counts."""
    counts = selected.select(
        pl.len().alias("selected_rows"),
        pl.col("person_id").n_unique().alias("unique_patients"),
        pl.col("person_id").is_null().sum().alias("missing_patient_ids"),
        (~pl.col("split").is_in(["train", "validation", "test"]))
        .fill_null(True)
        .sum()
        .alias("invalid_split_rows"),
    ).collect(engine="streaming").to_dicts()[0]
    split_rows = selected.group_by("split").len().collect(engine="streaming")
    split_counts = dict(zip(split_rows["split"].to_list(), split_rows["len"].to_list()))
    expected = {
        "train": TRAIN_SIZE,
        "validation": VALIDATION_SIZE,
        "test": SAMPLE_SIZE - TRAIN_SIZE - VALIDATION_SIZE,
    }
    if counts["missing_patient_ids"] or counts["invalid_split_rows"]:
        raise ValueError(f"Invalid patient ID or split in sample: {counts}")
    if counts["selected_rows"] != counts["unique_patients"]:
        raise ValueError("Duplicate patient IDs in sample; each patient must be in exactly one split.")
    if split_counts != expected:
        raise ValueError(f"Sample split counts differ: actual={split_counts}, expected={expected}")
    return {**counts, "split_counts": split_counts}


def run_given_sample(config: dict, patients_file: Path, destination: Path, summary_path: Path) -> None:
    """Save a given patient list (a virtual site from federated site-make) as the sample."""
    selected = pl.read_parquet(patients_file, columns=["person_id", "split"]).with_columns(
        pl.col("person_id").cast(pl.Int64)
    ).sort("person_id")
    split_audit = validate_patient_split(selected.lazy())
    temp = destination.with_name(destination.name + ".tmp")
    selected.write_parquet(temp, compression="zstd")
    temp.replace(destination)
    write_json(summary_path, {
        "source_fingerprint": source_fingerprint(config["paths"]["raw_root"]),
        "sample_method": "given patient list (virtual site from sites.py)",
        "patients_file": str(patients_file),
        "sample_size": selected.height,
        "initial_split_counts": split_audit["split_counts"],
        "split_overlap_patient_count": 0,
        "selected_patients_path": str(destination),
    })
    print(f"Saved given patient list ({selected.height:,} patients): {destination}")


def run_sample(config_path: str) -> None:
    config = load_config(config_path)
    raw_root: Path = config["paths"]["raw_root"]
    destination = output_file(config, "sample", "selected_patients.parquet")
    summary_path = output_file(config, "sample", "summary.json")

    if destination.exists():
        raise FileExistsError(
            f"Sample file already exists: {destination}. Reuse the existing sample."
        )

    patients_file = config.get("sample", {}).get("patients_file")
    if patients_file is not None:
        # Virtual site: use the prebuilt (person_id, split) list as is.
        run_given_sample(config, Path(patients_file).expanduser().resolve(), destination, summary_path)
        return
    selected, population_size = choose_patients(source_parts(raw_root, "PERSON"))
    split_audit = validate_patient_split(selected.lazy())
    temp = destination.with_name(destination.name + ".tmp")
    selected.write_parquet(temp, compression="zstd")
    temp.replace(destination)

    split_counts = split_audit["split_counts"]

    write_json(
        summary_path,
        {
            "source_fingerprint": source_fingerprint(raw_root),
            "person_population_unique": population_size,
            "seed": SEED,
            "sample_method": "numpy Generator.choice without replacement after sorting unique person_id",
            "sample_size": SAMPLE_SIZE,
            "initial_split_counts": split_counts,
            "split_overlap_patient_count": 0,
            "selection_uses_event_history": False,
            "exclusion_policy": "keep original split; no replacement or resplitting",
            "selected_patients_path": str(destination),
        },
    )
    print(f"Sampled {SAMPLE_SIZE:,} of {population_size:,} patients")
    print(
        f"train={split_counts['train']:,}, "
        f"validation={split_counts['validation']:,}, test={split_counts['test']:,}"
    )
    print(f"Saved: {destination}")


# Dict order is the fixed same-day domain order.
DOMAINS = {
    "condition": (
        "CONDITION_OCCURRENCE",
        "condition_occurrence_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_start_datetime",
    ),
    "drug": (
        "DRUG_EXPOSURE",
        "drug_exposure_id",
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_exposure_start_datetime",
    ),
    "procedure": (
        "PROCEDURE_OCCURRENCE",
        "procedure_occurrence_id",
        "procedure_concept_id",
        "procedure_date",
        "procedure_datetime",
    ),
    "measurement": (
        "MEASUREMENT",
        "measurement_id",
        "measurement_concept_id",
        "measurement_date",
        "measurement_datetime",
    ),
    "observation": (
        "OBSERVATION",
        "observation_id",
        "observation_concept_id",
        "observation_date",
        "observation_datetime",
    ),
}


# Final diagnoses are stored at visit start; events.move_discharge_codes_to_visit_end moves them.
FINAL_DIAGNOSIS_STATUS = 4230359
# Diagnosis codes mapped to non-condition concepts have no status; found by source vocabulary instead.
DIAGNOSIS_SOURCE_VOCABULARIES = ["KCD7", "KCD8 (temp)"]
SOURCE_CONCEPT_COLUMNS = {
    "condition": "condition_source_concept_id",
    "drug": "drug_source_concept_id",
    "procedure": "procedure_source_concept_id",
    "measurement": "measurement_source_concept_id",
    "observation": "observation_source_concept_id",
}


def moves_codes_to_visit_end(config: dict) -> bool:
    return bool(config.get("events", {}).get("move_discharge_codes_to_visit_end", False))


def move_codes_to_visit_end(data: pl.LazyFrame, raw_root: Path, domain: str) -> pl.LazyFrame:
    """Move final diagnoses and KCD-sourced rows to the visit end when the visit ends after the event.

    Original values are kept in original_event_date and original_source_datetime.
    """
    if domain == "condition":
        scope = (pl.col("condition_status_concept_id") == FINAL_DIAGNOSIS_STATUS).fill_null(False)
    elif domain in ("procedure", "measurement", "observation"):
        diagnosis_codes = (
            pl.scan_parquet(source_parts(raw_root, "CONCEPT"))
            .filter(pl.col("vocabulary_id").is_in(DIAGNOSIS_SOURCE_VOCABULARIES))
            .select(pl.col("concept_id").cast(pl.Int64).alias("source_concept_id"))
            .with_columns(pl.lit(True).alias("_diagnosis_source"))
        )
        data = data.join(diagnosis_codes, on="source_concept_id", how="left")
        scope = pl.col("_diagnosis_source").fill_null(False)
    else:
        scope = pl.lit(False)
    visits = pl.scan_parquet(source_parts(raw_root, "VISIT_OCCURRENCE")).select(
        pl.col("visit_occurrence_id").cast(pl.Int64),
        pl.col("visit_end_date").cast(pl.String).str.slice(0, 10)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_visit_end_date"),
        pl.col("visit_end_datetime").cast(pl.String).alias("_visit_end_datetime"),
    )
    moved = pl.col("moved_to_visit_end")
    helper_columns = ["_visit_end_date", "_visit_end_datetime"]
    if domain in ("procedure", "measurement", "observation"):
        helper_columns.append("_diagnosis_source")
    return (
        data.join(visits, on="visit_occurrence_id", how="left")
        .with_columns(
            (
                scope
                & pl.col("event_date").is_not_null()
                & pl.col("_visit_end_date").is_not_null()
                & (pl.col("_visit_end_date") > pl.col("event_date"))
            ).fill_null(False).alias("moved_to_visit_end"),
            pl.col("event_date").alias("original_event_date"),
            pl.col("source_datetime").alias("original_source_datetime"),
        )
        .with_columns(
            pl.when(moved).then(pl.col("_visit_end_date")).otherwise(pl.col("event_date")).alias("event_date"),
            pl.when(moved).then(pl.col("_visit_end_date")).otherwise(pl.col("datetime_date")).alias("datetime_date"),
            pl.when(moved).then(pl.col("_visit_end_datetime")).otherwise(pl.col("source_datetime"))
            .alias("source_datetime"),
        )
        .drop(helper_columns)
    )


def source_or_null(schema: pl.Schema, name: str, dtype: pl.DataType, alias: str) -> pl.Expr:
    if name in schema:
        return pl.col(name).cast(dtype, strict=False).alias(alias)
    return pl.lit(None, dtype=dtype).alias(alias)


def build_common_scan(
    raw_root: Path, selected_path: Path, domain: str, move_to_visit_end: bool = False
) -> pl.LazyFrame:
    """Lazy scan of one domain for the sampled patients; rows without visit or concept IDs are kept."""
    table, id_col, concept_col, date_col, time_col = DOMAINS[domain]
    source = pl.scan_parquet(source_parts(raw_root, table))
    schema = source.collect_schema()
    selected = pl.scan_parquet(selected_path).select("person_id", "split")

    if domain == "measurement":
        result_text_col = "value_source_value"
    elif domain == "observation":
        result_text_col = "value_as_string"
    else:
        result_text_col = ""

    if domain == "observation":
        qualifier_text_col = "qualifier_source_value"
    else:
        qualifier_text_col = ""

    source_code_col = {
        "condition": "condition_source_value",
        "drug": "drug_source_value",
        "procedure": "procedure_source_value",
        "measurement": "measurement_source_value",
        "observation": "observation_source_value",
    }[domain]

    columns = source.select(
        pl.col("person_id").cast(pl.Int64),
        source_or_null(schema, id_col, pl.Int64, "event_id"),
        pl.col(concept_col).cast(pl.Int64).alias("concept_id"),
        source_or_null(schema, date_col, pl.String, "source_date"),
        source_or_null(schema, time_col, pl.String, "source_datetime"),
        source_or_null(schema, "visit_occurrence_id", pl.Int64, "visit_occurrence_id"),
        source_or_null(schema, "value_as_number", pl.Float64, "value_as_number"),
        source_or_null(schema, "value_as_concept_id", pl.Int64, "value_as_concept_id"),
        source_or_null(schema, result_text_col, pl.String, "result_text"),
        source_or_null(schema, qualifier_text_col, pl.String, "qualifier_text"),
        source_or_null(schema, "operator_concept_id", pl.Int64, "operator_concept_id"),
        source_or_null(schema, "unit_concept_id", pl.Int64, "unit_concept_id"),
        source_or_null(schema, "unit_source_value", pl.String, "unit_source_value"),
        source_or_null(schema, "range_low", pl.Float64, "range_low"),
        source_or_null(schema, "range_high", pl.Float64, "range_high"),
        source_or_null(
            schema, "condition_status_concept_id", pl.Int64, "condition_status_concept_id"
        ),
        source_or_null(schema, source_code_col, pl.String, "source_code"),
        source_or_null(schema, SOURCE_CONCEPT_COLUMNS[domain], pl.Int64, "source_concept_id"),
    )

    # Dates are the first 10 chars (YYYY-MM-DD); parse failures stay NULL and are audited.
    export_date = export_finish_date(raw_root)
    dated = columns.join(selected, on="person_id", how="inner").with_columns(
        pl.col("source_date")
        .str.slice(0, 10)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("event_date"),
        pl.col("source_datetime")
        .str.slice(0, 10)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("datetime_date"),
        pl.lit(domain).alias("domain"),
    )
    if move_to_visit_end:
        dated = move_codes_to_visit_end(dated, raw_root, domain)
    return (
        dated
        .with_columns(
            (
                pl.col("event_date").is_not_null()
                & pl.col("datetime_date").is_not_null()
                & (pl.col("event_date") != pl.col("datetime_date"))
            ).alias("date_mismatch"),
            pl.when(
                pl.lit(domain == "observation")
                & (pl.col("event_date") > pl.lit(export_date).str.to_date())
            )
            .then(pl.lit("observation_after_export"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("event_hold_reason"),
        )
    )


def summarize_output(
    event_path: Path, table: str, raw_root: Path, domain: str
) -> dict:
    """Row counts per exclusion and hold reason for one extracted domain.

    OBSERVATION after export is held; other date problems on mapped rows need a handling rule.
    """
    export_date = export_finish_date(raw_root)
    data = pl.scan_parquet(event_path)
    mapped = (pl.col("concept_id") > 0).fill_null(False)
    after_export = (pl.col("event_date") > pl.lit(export_date).str.to_date()).fill_null(False)
    confirmed_hold = after_export & pl.lit(domain == "observation")
    needs_date_policy = mapped & ~confirmed_hold
    missing_date = pl.col("event_date").is_null()
    missing_datetime = pl.col("datetime_date").is_null()
    date_mismatch = pl.col("date_mismatch").fill_null(False)
    missing_event_id = pl.col("event_id").is_null()
    unknown_future = after_export & pl.lit(domain != "observation")
    pending_event = needs_date_policy & (
        missing_date | missing_datetime | date_mismatch | missing_event_id | unknown_future
    )
    candidate = (
        mapped & ~missing_date & ~missing_datetime & ~date_mismatch
        & ~missing_event_id & ~after_export
    )
    counts = data.select(
        pl.len().alias("selected_raw_rows"),
        (~mapped).sum().alias("unmapped_rows"),
        mapped.sum().alias("mapped_rows"),
        missing_date.sum().alias("missing_or_bad_date_rows"),
        missing_datetime.sum().alias("missing_or_bad_datetime_rows"),
        date_mismatch.sum().alias("date_mismatch_rows"),
        after_export.sum().alias("after_export_rows"),
        (mapped & confirmed_hold).sum().alias("held_after_export_rows"),
        (needs_date_policy & missing_date).sum().alias("pending_missing_or_bad_date_rows"),
        (needs_date_policy & missing_datetime).sum().alias("pending_missing_or_bad_datetime_rows"),
        (needs_date_policy & date_mismatch).sum().alias("pending_date_mismatch_rows"),
        (needs_date_policy & missing_event_id).sum().alias("pending_missing_event_id_rows"),
        (mapped & unknown_future).sum().alias("pending_unconfirmed_future_rows"),
        pending_event.sum().alias("pending_event_rows"),
        candidate.sum().alias("model_event_candidate_rows"),
        (
            pl.col("moved_to_visit_end").sum() if "moved_to_visit_end" in data.collect_schema().names()
            else pl.lit(0)
        ).alias("moved_to_visit_end_rows"),
    ).collect(engine="streaming").to_dicts()[0]
    pending_decisions = {
        key.removeprefix("pending_"): counts[key]
        for key in [
            "pending_missing_or_bad_date_rows",
            "pending_missing_or_bad_datetime_rows",
            "pending_date_mismatch_rows",
            "pending_missing_event_id_rows",
            "pending_unconfirmed_future_rows",
        ]
        if counts[key]
    }
    total = counts["selected_raw_rows"]
    table_marker = json.loads(
        (raw_root / "dbo" / table / "_COMPLETE.json").read_text(encoding="utf-8")
    )
    schema = pl.scan_parquet(source_parts(raw_root, table)).collect_schema()
    return {
        "domain": domain,
        "source_table": table,
        "source_table_rows": table_marker["rows_written"],
        "source_columns": list(schema.names()),
        "source_fingerprint": source_fingerprint(raw_root),
        "export_finish_date": export_date,
        **counts,
        "unmapped_ratio": counts["unmapped_rows"] / total if total else 0.0,
        "model_event_candidate_ratio": counts["model_event_candidate_rows"] / total if total else 0.0,
        "pending_decisions": pending_decisions,
        "hold_reason": (
            "OBSERVATION event_date is after export completion date"
            if domain == "observation" else None
        ),
        "retained_source_path": str(event_path),
        "status": "needs_user_decision" if pending_decisions else "ready",
    }


def extract_one(config: dict, domain: str, resume: bool) -> dict:
    raw_root: Path = config["paths"]["raw_root"]
    selected_path = output_file(config, "sample", "selected_patients.parquet")
    if not selected_path.exists():
        raise FileNotFoundError("Run data sample first.")

    table = DOMAINS[domain][0]
    destination = output_file(config, "events", f"{domain}.parquet")
    summary_path = output_file(config, "events", f"{domain}_summary.json")
    temp = destination.with_name(destination.name + ".tmp")

    if destination.exists() and not resume:
        raise FileExistsError(f"Extracted file already exists: {destination}")
    if destination.exists() and resume:
        print(f"{domain}: reusing extracted file", flush=True)
    else:
        if temp.exists():
            temp.unlink()
        print(f"{domain}: scanning source", flush=True)
        query = build_common_scan(raw_root, selected_path, domain, moves_codes_to_visit_end(config))
        query.sink_parquet(temp, compression="zstd", engine="streaming")
        temp.replace(destination)
        print(f"{domain}: extraction saved", flush=True)

    summary = summarize_output(destination, table, raw_root, domain)
    write_json(summary_path, summary)
    print(
        f"{domain}: {summary['selected_raw_rows']:,} sampled source rows, "
        f"{summary['model_event_candidate_rows']:,} valid event candidates",
        flush=True,
    )
    print(f"{domain}: audit {summary_path}", flush=True)
    return summary


def require_sample_ready(config: dict) -> None:
    """Check that the existing sample matches the current source and the fixed split sizes."""
    selected_path = output_file(config, "sample", "selected_patients.parquet")
    summary_path = output_file(config, "sample", "summary.json")
    if not selected_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run data sample first to create the sample and its summary.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fingerprint = source_fingerprint(config["paths"]["raw_root"])
    if summary["source_fingerprint"] != fingerprint:
        raise ValueError("Sample was built from a different source; check the source settings.")
    given = summary.get("sample_method", "").startswith("given patient list")
    if summary["sample_size"] != SAMPLE_SIZE or (not given and summary.get("seed") != SEED):
        # A given patient list (virtual site) has no seed.
        raise ValueError("Sample seed or size differs from the fixed settings.")
    validate_patient_split(pl.scan_parquet(selected_path).select("person_id", "split"))


def check_domain_exceptions(config: dict, domains: list[str]) -> None:
    """Write the combined event audit and stop if any domain has unresolved exceptions."""
    fingerprint = source_fingerprint(config["paths"]["raw_root"])
    domain_audits = []
    pending = {}
    for domain in domains:
        path = output_file(config, "events", f"{domain}_summary.json")
        if not path.exists():
            raise FileNotFoundError(f"Event audit not found; run data events: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("source_fingerprint") != fingerprint:
            raise ValueError(f"{domain} event audit was built from a different source.")
        if "pending_decisions" not in summary:
            raise ValueError(
                "Old event audit format; refresh it with data events --all --resume."
            )
        domain_audits.append({
            "domain": domain,
            "selected_raw_rows": summary["selected_raw_rows"],
            "held_after_export_rows": summary["held_after_export_rows"],
            "pending_event_rows": summary["pending_event_rows"],
            "pending_decisions": summary["pending_decisions"],
            "retained_source_path": summary["retained_source_path"],
        })
        if summary["pending_decisions"]:
            pending[domain] = summary["pending_decisions"]

    destination = output_file(config, "events", "exceptions_summary.json")
    write_json(destination, {
        "source_fingerprint": fingerprint,
        "audited_domains": domains,
        "status": "needs_user_decision" if pending else "ready",
        "domain_audits": domain_audits,
        "pending_reasons": {
            "missing_or_bad_date_rows": "event date missing or unparseable",
            "missing_or_bad_datetime_rows": "event datetime missing or unparseable",
            "date_mismatch_rows": "date and datetime fall on different calendar days",
            "missing_event_id_rows": "source row ID needed for fixed ordering is missing",
            "unconfirmed_future_rows": "date after export completion in a non-OBSERVATION domain",
        },
        "confirmed_hold_reason": "OBSERVATION after export completion: source rows kept, held out of model input",
    })
    if pending:
        raise ValueError(
            f"Unresolved event exceptions; decide a handling rule first. Counts={pending}. Audit: {destination}"
        )


def check_event_exceptions(config: dict) -> None:
    """Stop on unresolved exceptions in any domain; call before values, deciles, vocab and tokens."""
    check_domain_exceptions(config, list(DOMAINS))


def run_events(config_path: str, domains: list[str], resume: bool = False) -> None:
    config = load_config(config_path)
    require_sample_ready(config)
    for domain in domains:
        extract_one(config, domain, resume)
    check_domain_exceptions(config, domains)


# Excluded from model input but kept in the extracted files.
EXCLUDED_CONCEPTS = {
    ("observation", 3046965): "echocardiography report header (Sex)",
    ("observation", 3022304): "echocardiography report header (Age)",
    ("observation", 3022007): "echocardiography report header (Birth date)",
    ("observation", 3031159): "echocardiography report header (Ultrasound date)",
}

# Rows equal on this key are copies of one record; source code counts only for unmapped rows.
DUPLICATE_KEY = [
    "person_id", "concept_id", "_unmapped_source_code", "event_date", "source_datetime",
    "value_as_number", "value_as_concept_id", "result_text", "unit_source_value",
    "range_low", "range_high", "condition_status_concept_id",
]


def excluded_concept_filter(domain: str) -> pl.Expr:
    excluded = [concept for (name, concept) in EXCLUDED_CONCEPTS if name == domain]
    if not excluded:
        return pl.lit(True)
    return ~pl.col("concept_id").is_in(excluded).fill_null(False)


def clean_event_rows(data: pl.LazyFrame, domain: str) -> pl.LazyFrame:
    """Drop excluded concepts and keep the lowest event_id of each exact duplicate group."""
    columns = data.collect_schema().names()
    unmapped_code = pl.lit(None, dtype=pl.String)
    if "source_code" in columns:
        unmapped_code = (
            pl.when((pl.col("concept_id") > 0).fill_null(False))
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("source_code"))
        )
    data = data.with_columns(unmapped_code.alias("_unmapped_source_code"))
    key = [column for column in DUPLICATE_KEY if column in [*columns, "_unmapped_source_code"]]
    return (
        data.filter(excluded_concept_filter(domain))
        .sort("event_id")
        .unique(subset=key, keep="first")
        .drop("_unmapped_source_code")
    )


def run_demographics(config_path: str) -> None:
    """Sex and birth year/month of the sampled patients (birth day is always 1 in the source)."""
    config = load_config(config_path)
    destination = output_file(config, "demographics", "person.parquet")
    if destination.exists():
        raise FileExistsError(f"Demographics file already exists: {destination}")
    selected = pl.scan_parquet(output_file(config, "sample", "selected_patients.parquet")).select("person_id")
    person = (
        pl.scan_parquet(source_parts(config["paths"]["raw_root"], "PERSON"))
        .select(
            pl.col("person_id").cast(pl.Int64),
            pl.col("gender_concept_id").cast(pl.Int64),
            pl.col("year_of_birth").cast(pl.Int32),
            pl.col("month_of_birth").cast(pl.Int32),
        )
        .join(selected, on="person_id", how="inner")
        .unique("person_id", keep="first")
        .collect(engine="streaming")
    )
    temporary = destination.with_name(destination.name + ".tmp")
    person.write_parquet(temporary, compression="zstd")
    temporary.replace(destination)
    counts = person.group_by("gender_concept_id").len().sort("gender_concept_id").to_dicts()
    write_json(output_file(config, "demographics", "summary.json"), {
        "patients": person.height,
        "gender_counts": counts,
        "missing_year_of_birth": int(person["year_of_birth"].null_count()),
        "missing_month_of_birth": int(person["month_of_birth"].null_count()),
    })
    print(f"Saved demographics for {person.height:,} patients: {destination}", flush=True)


# ── Visits, deaths and last record dates ──

def run_visits(config_path: str) -> None:
    config = load_config(config_path)
    raw_root: Path = config["paths"]["raw_root"]
    selected = pl.scan_parquet(output_file(config, "sample", "selected_patients.parquet")).select(
        "person_id", "split"
    )
    visit_path = output_file(config, "cohorts", "visits.parquet")
    death_path = output_file(config, "cohorts", "death.parquet")

    if not visit_path.exists():
        visits = (
            pl.scan_parquet(source_parts(raw_root, "VISIT_OCCURRENCE"))
            .select(
                pl.col("person_id").cast(pl.Int64),
                pl.col("visit_occurrence_id").cast(pl.Int64),
                pl.col("visit_concept_id").cast(pl.Int64),
                "visit_start_date",
                "visit_start_datetime",
                "visit_end_date",
                "visit_end_datetime",
                pl.col("discharge_to_concept_id").cast(pl.Int64),
                "discharge_to_source_value",
            )
            .join(selected, on="person_id", how="inner")
        )
        temporary = visit_path.with_name(visit_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        visits.sink_parquet(temporary, compression="zstd", engine="streaming")
        temporary.replace(visit_path)

    if not death_path.exists():
        deaths = (
            pl.scan_parquet(source_parts(raw_root, "DEATH"))
            .select(pl.col("person_id").cast(pl.Int64), "death_date", "death_datetime")
            .join(selected, on="person_id", how="inner")
        )
        temporary = death_path.with_name(death_path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        deaths.sink_parquet(temporary, compression="zstd", engine="streaming")
        temporary.replace(death_path)

    record_end_path = output_file(config, "cohorts", "record_end.parquet")
    if not record_end_path.exists():
        write_record_end(config, visit_path, death_path, record_end_path)

    summary = {
        "record_end_patients": pl.scan_parquet(record_end_path).select(pl.len()).collect().item(),
        "selected_visits": pl.scan_parquet(visit_path).select(pl.len()).collect().item(),
        "selected_death_rows": pl.scan_parquet(death_path).select(pl.len()).collect().item(),
        "visit_types": pl.scan_parquet(visit_path)
        .group_by("visit_concept_id")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect()
        .to_dicts(),
        "death_usage": "outcome-window censoring only",
    }
    write_json(output_file(config, "cohorts", "visits_summary.json"), summary)
    print(f"Extracted {summary['selected_visits']:,} visits of sampled patients", flush=True)


def write_record_end(config: dict, visit_path: Path, death_path: Path, destination: Path) -> None:
    """Last record date per patient (latest event, visit or death date); dates after export are ignored."""
    export_date = pl.lit(export_finish_date(config["paths"]["raw_root"])).str.to_date()
    dates = []
    for domain in DOMAINS:
        source = output_file(config, "events", f"{domain}.parquet")
        if not source.exists():
            raise FileNotFoundError(f"Run event extraction first: {source}")
        dates.append(
            pl.scan_parquet(source).select("person_id", pl.col("event_date").alias("day"))
        )
    for column in ("visit_start_datetime", "visit_end_datetime"):
        dates.append(
            pl.scan_parquet(visit_path).select(
                "person_id", pl.col(column).str.slice(0, 10).str.to_date(strict=False).alias("day")
            )
        )
    dates.append(
        pl.scan_parquet(death_path).select(
            "person_id", pl.col("death_date").str.slice(0, 10).str.to_date(strict=False).alias("day")
        )
    )
    record_end = (
        pl.concat(dates)
        .filter(pl.col("day").is_not_null() & (pl.col("day") <= export_date))
        .group_by("person_id")
        .agg(pl.col("day").max().alias("last_record_date"))
        .collect(engine="streaming")
    )
    temporary = destination.with_name(destination.name + ".tmp")
    record_end.write_parquet(temporary, compression="zstd")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    stages.add_parser("sample", parents=[common_args], help="sample patients and assign train/validation/test")
    events = stages.add_parser("events", parents=[common_args], help="extract the five event domains")
    target = events.add_mutually_exclusive_group(required=True)
    target.add_argument("--domain", choices=list(DOMAINS))
    target.add_argument("--all", action="store_true", help="extract all five domains")
    events.add_argument("--resume", action="store_true", help="reuse existing event files and refresh the audit")
    stages.add_parser("demographics", parents=[common_args], help="sex and birth year/month for demographic tokens")
    stages.add_parser("visits", parents=[common_args], help="visits, deaths and last record dates")
    args = parser.parse_args()
    if args.stage == "sample":
        run_sample(args.config)
    elif args.stage == "events":
        run_events(args.config, list(DOMAINS) if args.all else [args.domain], args.resume)
    elif args.stage == "demographics":
        run_demographics(args.config)
    else:
        run_visits(args.config)


if __name__ == "__main__":
    main()
