"""Parse result values, convert units and compute train-only decile boundaries.

Stages (after data events): parse, units, deciles
Usage: python -m ausom_ehr.values <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

import polars as pl

from ausom_ehr.common import (
    config_parent, copy_reference_file, export_finish_date, load_config, output_file,
    reference_root, source_parts, write_json,
)
from ausom_ehr.data import check_event_exceptions, clean_event_rows


# ── Result parsing ──

# The whole value must be one number (optionally after a comparator); numbers inside text are not guessed.
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PLAIN_PATTERN = rf"^\s*({NUMBER})\s*$"
OPERATOR_PATTERN = rf"^\s*(?:<=|>=|<|>)\s*({NUMBER})\s*$"

# Thousands separators are accepted only after a comparator and only at exact 3-digit groups.
THOUSANDS_PATTERN = r"^\s*(?:<=|>=|<|>)\s*([+-]?\d{1,3}(?:,\d{3})+)\s*$"
# Korean-word comparators (at most / below / at least / above): keep the number, as with '<' or '>'.
KOREAN_OPERATOR_PATTERN = rf"^\s*({NUMBER})\s*(?:이하|미만|이상|초과)\s*$"
# Censoring direction, matched after comparator spellings are normalized.
LESS_TEXT = rf"^\s*<|^\s*{NUMBER}\s*(?:이하|미만)\s*$"
GREATER_TEXT = rf"^\s*>|^\s*{NUMBER}\s*(?:이상|초과)\s*$"
# Semi-quantitative grades 1+ to 4+ (e.g. urine dipstick) count as POS.
SEMIQUANTITATIVE_POSITIVE = r"^\s*[1-4]\s*\+\s*$"
# Digits and punctuation only; if this does not parse as one number it is malformed and kept as concept only.
NUMERIC_LIKE_TEXT = r"^\s*[+-]?[0-9.][0-9.,:/~+\s-]*$"

COMPARATOR_SPELLINGS = {"≤": "<=", "≦": "<=", "≥": ">=", "≧": ">=", "＜": "<", "＞": ">"}
FULLWIDTH_DIGITS = {chr(0xFF10 + digit): str(digit) for digit in range(10)} | {"．": "."}
# OMOP operator_concept_id for <, <= and >, >=.
LESS_OPERATOR_IDS = [4171756, 4171754]
GREATER_OPERATOR_IDS = [4172704, 4171755]

# Exact matches only; other wordings are not guessed.
NEGATIVE_TEXT = ["NEGATIVE", "NEG", "음성"]
POSITIVE_TEXT = ["POSITIVE", "POS", "양성"]
RESULT_DOMAINS = ("measurement", "observation")

# Default only; the parse stage passes a map built from observed IDs and CONCEPT names.
QUALITATIVE_CONCEPTS = {9189: "NEG", 9191: "POS"}

# Unresolved exceptions: rows and counts are saved, then later stages stop.
PARSING_EXCEPTION_COUNTS = (
    "qual_conflict_rows",
    "structured_raw_number_conflict_rows",
    "nonfinite_number_rows",
    "nonfinite_text_rows",
    "numeric_parse_failure_rows",
    "censoring_conflict_rows",
    "nonfinite_range_low_rows",
    "nonfinite_range_high_rows",
)
# Kept as concept only (no number token) and not blocking, e.g. titers '1:16', times '2:30', typos '2..34'.
CONCEPT_ONLY_COUNTS = (
    "malformed_operator_number_rows",
    "malformed_numeric_text_rows",
)


def add_parsed_results(
    data: pl.LazyFrame, qualitative_concepts: dict[int, str] | None = None
) -> pl.LazyFrame:
    """Add numeric value, NEG/POS and exception flag columns; source rows and columns are kept."""
    if qualitative_concepts is None:
        qualitative_concepts = QUALITATIVE_CONCEPTS
    negative_ids = [key for key, value in qualitative_concepts.items() if value == "NEG"]
    positive_ids = [key for key, value in qualitative_concepts.items() if value == "POS"]
    source_columns = data.collect_schema().names()
    clean_text = pl.col("result_text").fill_null("").str.strip_chars()
    for spelling, operator in (COMPARATOR_SPELLINGS | FULLWIDTH_DIGITS).items():
        clean_text = clean_text.str.replace_all(spelling, operator, literal=True)
    data = data.with_columns(
        clean_text.alias("_clean_text"),
        pl.col("value_as_number").is_not_null().alias("_has_structured_number"),
    ).with_columns(
        pl.col("_clean_text")
        .str.extract(OPERATOR_PATTERN, 1)
        .cast(pl.Float64, strict=False)
        .alias("_operator_number"),
        pl.col("_clean_text")
        .str.extract(THOUSANDS_PATTERN, 1)
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
        .alias("_thousands_number"),
        pl.col("_clean_text")
        .str.extract(KOREAN_OPERATOR_PATTERN, 1)
        .cast(pl.Float64, strict=False)
        .alias("_korean_operator_number"),
        pl.col("_clean_text").str.contains(SEMIQUANTITATIVE_POSITIVE).alias("_semiquantitative_positive"),
        pl.col("_clean_text")
        .str.extract(PLAIN_PATTERN, 1)
        .cast(pl.Float64, strict=False)
        .alias("_plain_number"),
        pl.col("_clean_text").str.to_uppercase().alias("_upper_text"),
    )

    # Structured number wins; a comparator gives only its bound number (no comparator token).
    data = data.with_columns(
        pl.when(pl.col("_has_structured_number"))
        .then(pl.col("value_as_number"))
        .otherwise(pl.coalesce(
            "_operator_number", "_korean_operator_number", "_thousands_number", "_plain_number",
        ))
        .alias("numeric_value_raw"),
        pl.when(pl.col("_has_structured_number"))
        .then(pl.lit("structured"))
        .when(pl.col("_operator_number").is_not_null())
        .then(pl.lit("raw_operator"))
        .when(pl.col("_korean_operator_number").is_not_null())
        .then(pl.lit("raw_korean_operator"))
        .when(pl.col("_thousands_number").is_not_null())
        .then(pl.lit("raw_operator_thousands"))
        .when(pl.col("_plain_number").is_not_null())
        .then(pl.lit("raw_number"))
        .otherwise(pl.lit("none"))
        .alias("numeric_source"),
        pl.when(pl.col("value_as_concept_id").is_in(negative_ids))
        .then(pl.lit("NEG"))
        .when(pl.col("value_as_concept_id").is_in(positive_ids))
        .then(pl.lit("POS"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("_coded_qual"),
        pl.when(pl.col("_upper_text").is_in(NEGATIVE_TEXT))
        .then(pl.lit("NEG"))
        .when(pl.col("_upper_text").is_in(POSITIVE_TEXT) | pl.col("_semiquantitative_positive"))
        .then(pl.lit("POS"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("_text_qual"),
        pl.coalesce("_operator_number", "_korean_operator_number", "_thousands_number", "_plain_number")
        .alias("_raw_number"),
    )
    data = data.with_columns(
        pl.coalesce("_coded_qual", "_text_qual").alias("qual_result"),
        (
            pl.col("_coded_qual").is_not_null()
            & pl.col("_text_qual").is_not_null()
            & (pl.col("_coded_qual") != pl.col("_text_qual"))
        )
        .fill_null(False)
        .alias("qual_conflict"),
        (
            pl.col("_has_structured_number")
            & pl.col("_raw_number").is_not_null()
            & (pl.col("value_as_number") != pl.col("_raw_number"))
        )
        .fill_null(False)
        .alias("structured_raw_number_conflict"),
        (
            pl.col("numeric_value_raw").is_not_null()
            & ~pl.col("numeric_value_raw").is_finite()
        )
        .fill_null(False)
        .alias("nonfinite_number"),
        (
            pl.col("_upper_text").str.contains(
                r"^(?:<=|>=|<|>)?\s*[+-]?(?:NAN|INF(?:INITY)?|∞)$"
            )
        ).alias("nonfinite_text"),
        (
            pl.col("_clean_text").str.contains(r"^\s*(?:<=|>=|<|>)\s*[+-]?(?:[0-9]|\.)")
            & pl.col("_operator_number").is_null()
            & pl.col("_thousands_number").is_null()
        ).alias("malformed_operator_number"),
        (
            pl.col("_clean_text").str.contains(NUMERIC_LIKE_TEXT)
            & pl.col("_clean_text").str.contains(r"[0-9]")
            & ~pl.col("_semiquantitative_positive")
            & pl.col("_plain_number").is_null()
        ).alias("malformed_numeric_text"),
        (
            (
                pl.col("_clean_text").str.contains(PLAIN_PATTERN)
                | pl.col("_clean_text").str.contains(OPERATOR_PATTERN)
                | pl.col("_clean_text").str.contains(KOREAN_OPERATOR_PATTERN)
                | pl.col("_clean_text").str.contains(THOUSANDS_PATTERN)
            )
            & pl.col("_raw_number").is_null()
        ).alias("numeric_parse_failure"),
        *[
            (
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                if column in source_columns else pl.lit(False)
            ).fill_null(False).alias(f"nonfinite_{column}")
            for column in ("range_low", "range_high")
        ],
    )
    # Censoring is not a token; it only keeps '<X' from being HIGH and '>X' from being LOW.
    operator = (
        pl.col("operator_concept_id") if "operator_concept_id" in source_columns
        else pl.lit(None, dtype=pl.Int64)
    )
    less = (
        operator.is_in(LESS_OPERATOR_IDS)
        | pl.col("_clean_text").str.contains(LESS_TEXT)
    ).fill_null(False)
    greater = (
        operator.is_in(GREATER_OPERATOR_IDS)
        | pl.col("_clean_text").str.contains(GREATER_TEXT)
    ).fill_null(False)
    has_number = pl.col("numeric_value_raw").is_not_null()
    data = data.with_columns(
        pl.when(has_number & less & ~greater).then(pl.lit("less"))
        .when(has_number & greater & ~less).then(pl.lit("greater"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("numeric_censoring"),
        (has_number & less & greater).alias("censoring_conflict"),
    )
    return data.drop(
        "_clean_text",
        "_upper_text",
        "_has_structured_number",
        "_operator_number",
        "_thousands_number",
        "_korean_operator_number",
        "_semiquantitative_positive",
        "_plain_number",
        "_raw_number",
        "_coded_qual",
        "_text_qual",
    )


def build_qualitative_dictionary(config: dict) -> dict[int, str]:
    """Map observed value_as_concept_id to NEG/POS by exact CONCEPT name; saves the evidence table."""
    observed = pl.concat([
        pl.scan_parquet(output_file(config, "events", f"{domain}.parquet"))
        .filter(pl.col("value_as_concept_id") > 0)
        .group_by("value_as_concept_id")
        .agg(pl.len().alias("rows"))
        for domain in RESULT_DOMAINS
    ]).group_by("value_as_concept_id").agg(pl.sum("rows"))
    observed_ids = observed.collect(engine="streaming")
    ids = observed_ids["value_as_concept_id"].to_list()
    concepts = (
        pl.scan_parquet(source_parts(config["paths"]["raw_root"], "CONCEPT"))
        .filter(pl.col("concept_id").is_in(ids))
        .select(
            pl.col("concept_id").alias("value_as_concept_id"),
            "concept_name", "concept_code", "vocabulary_id",
        )
        .collect(engine="streaming")
    )
    dictionary = observed_ids.join(concepts, on="value_as_concept_id", how="left")
    name = pl.col("concept_name").str.strip_chars().str.to_uppercase()
    dictionary = dictionary.with_columns(
        pl.when(name.is_in(NEGATIVE_TEXT)).then(pl.lit("NEG"))
        .when(name.is_in(POSITIVE_TEXT)).then(pl.lit("POS"))
        .otherwise(pl.lit(None, dtype=pl.String)).alias("qual_result"),
    )
    dictionary = dictionary.with_columns(
        pl.when(pl.col("qual_result").is_not_null())
        .then(pl.lit("CONCEPT name exactly matches a NEG/POS term"))
        .otherwise(pl.lit("source concept not read as a qualitative token"))
        .alias("evidence")
    )
    destination = output_file(config, "results", "qualitative_dictionary.parquet")
    temporary = destination.with_name(destination.name + ".tmp")
    dictionary.write_parquet(temporary, compression="zstd")
    temporary.replace(destination)

    text_counts = pl.concat([
        pl.scan_parquet(output_file(config, "events", f"{domain}.parquet"))
        .select("domain", "result_text") for domain in RESULT_DOMAINS
    ]).group_by("domain", "result_text").agg(pl.len().alias("rows"))
    text_destination = output_file(config, "results", "observed_result_texts.parquet")
    text_temporary = text_destination.with_name(text_destination.name + ".tmp")
    text_counts.sink_parquet(text_temporary, compression="zstd", engine="streaming")
    text_temporary.replace(text_destination)
    return {
        int(row["value_as_concept_id"]): row["qual_result"]
        for row in dictionary.filter(pl.col("qual_result").is_not_null()).to_dicts()
    }


def parse_result_domain(
    config: dict, domain: str, resume: bool,
    qualitative_concepts: dict[int, str] | None = None,
) -> dict:
    source = output_file(config, "events", f"{domain}.parquet")
    if not source.exists():
        raise FileNotFoundError(f"Event extraction has not finished: {source}")
    destination = output_file(config, "results", f"{domain}.parquet")
    summary_path = output_file(config, "results", f"{domain}_summary.json")

    if destination.exists() and not resume:
        raise FileExistsError(f"Result file already exists: {destination}")
    required_audit_columns = [
        key.removesuffix("_rows") for key in (*PARSING_EXCEPTION_COUNTS, *CONCEPT_ONLY_COUNTS)
    ]
    needs_parsing = not destination.exists()
    if destination.exists() and resume:
        saved_columns = pl.scan_parquet(destination).collect_schema().names()
        needs_parsing = any(column not in saved_columns for column in required_audit_columns)
    if needs_parsing:
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        events = clean_event_rows(pl.scan_parquet(source), domain)
        add_parsed_results(events, qualitative_concepts).sink_parquet(
            temporary, compression="zstd", engine="streaming"
        )
        temporary.replace(destination)

    data = pl.scan_parquet(destination)
    counts = data.select(
        pl.len().alias("rows"),
        pl.col("numeric_value_raw").is_not_null().sum().alias("numeric_rows"),
        (pl.col("numeric_source") == "raw_operator").sum().alias("raw_operator_rows"),
        (pl.col("numeric_source") == "raw_number").sum().alias("raw_number_rows"),
        (pl.col("numeric_source") == "raw_operator_thousands").sum().alias("raw_operator_thousands_rows"),
        (pl.col("numeric_source") == "raw_korean_operator").sum().alias("raw_korean_operator_rows"),
        (pl.col("qual_result") == "NEG").sum().alias("negative_rows"),
        (pl.col("qual_result") == "POS").sum().alias("positive_rows"),
        pl.col("qual_conflict").sum().alias("qual_conflict_rows"),
        pl.col("structured_raw_number_conflict").sum().alias("structured_raw_number_conflict_rows"),
        pl.col("nonfinite_number").sum().alias("nonfinite_number_rows"),
        pl.col("nonfinite_text").sum().alias("nonfinite_text_rows"),
        pl.col("malformed_operator_number").sum().alias("malformed_operator_number_rows"),
        pl.col("malformed_numeric_text").sum().alias("malformed_numeric_text_rows"),
        pl.col("numeric_parse_failure").sum().alias("numeric_parse_failure_rows"),
        (pl.col("numeric_censoring") == "less").sum().alias("less_than_rows"),
        (pl.col("numeric_censoring") == "greater").sum().alias("greater_than_rows"),
        pl.col("censoring_conflict").sum().alias("censoring_conflict_rows"),
        pl.col("nonfinite_range_low").sum().alias("nonfinite_range_low_rows"),
        pl.col("nonfinite_range_high").sum().alias("nonfinite_range_high_rows"),
        (
            pl.col("numeric_value_raw").is_not_null()
            & pl.col("qual_result").is_not_null()
        ).sum().alias("numeric_and_qualitative_rows"),
    ).collect(engine="streaming").to_dicts()[0]
    write_json(summary_path, {"domain": domain, **counts})
    exception_columns = [key.removesuffix("_rows") for key in PARSING_EXCEPTION_COUNTS]
    exceptions = data.filter(pl.any_horizontal(*[pl.col(key) for key in exception_columns]))
    audit_path = output_file(config, "results", f"{domain}_exceptions.parquet")
    temporary = audit_path.with_name(audit_path.name + ".tmp")
    exceptions.sink_parquet(temporary, compression="zstd", engine="streaming")
    temporary.replace(audit_path)
    concept_only_columns = [key.removesuffix("_rows") for key in CONCEPT_ONLY_COUNTS]
    concept_only = data.filter(pl.any_horizontal(*[pl.col(key) for key in concept_only_columns]))
    concept_only_path = output_file(config, "results", f"{domain}_concept_only_values.parquet")
    temporary = concept_only_path.with_name(concept_only_path.name + ".tmp")
    concept_only.sink_parquet(temporary, compression="zstd", engine="streaming")
    temporary.replace(concept_only_path)
    print(f"{domain}: parsed {counts['rows']:,} rows, summary {summary_path}", flush=True)
    return counts


def check_parsing_exceptions(config: dict) -> None:
    """Raise if any unresolved parsing exception was counted."""
    unexpected = {}
    for domain in RESULT_DOMAINS:
        path = output_file(config, "results", f"{domain}_summary.json")
        if not path.exists():
            raise FileNotFoundError(f"Result parsing summary not found: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        missing_counts = [key for key in PARSING_EXCEPTION_COUNTS if key not in report]
        if missing_counts:
            raise ValueError(
                f"{domain}: existing parsing summary lacks new checks: {missing_counts}. "
                "Rerun values parse --resume to update it."
            )
        counts = {key: report.get(key, 0) for key in PARSING_EXCEPTION_COUNTS if report.get(key, 0) > 0}
        if counts:
            unexpected[domain] = counts
    if unexpected:
        raise ValueError(
            f"Saved result exception types and counts: {unexpected}. "
            "Check results/*_exceptions.parquet and decide a handling rule first."
        )


def parse_results(config: dict, resume: bool = False) -> None:
    check_event_exceptions(config)
    qualitative_concepts = build_qualitative_dictionary(config)
    for domain in RESULT_DOMAINS:
        parse_result_domain(config, domain, resume, qualitative_concepts)
    check_parsing_exceptions(config)


# ── Units ──
# Units are not tokens; they only define decile groups. Spellings follow UCUM: https://ucum.org/ucum

def normalize_unit_text(text: str | None) -> str:
    """Lowercase, space-free unit text; superscripts become '^n' before NFKC would turn '10³' into '103'."""
    if not text:
        return ""
    for superscript, plain in {"³": "^3", "⁶": "^6", "²": "^2"}.items():
        text = text.replace(superscript, plain)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("μ", "u").replace("µ", "u").replace("×", "x")
    text = text.lower().replace(" ", "")
    return text.replace("mcg", "ug")


# Canonical unit -> normalized spellings seen in unit text and UCUM unit concepts.
CANONICAL_UNIT_SPELLINGS = {
    "%": ["%", "%rapid", "%dead", "%live", "%normal"],
    "ratio": ["ratio", "{ratio}"],
    "index": ["index", "{signal}/[{co'ratio}]"],
    "MoM": ["mom"],
    "C(t)": ["c(t)"],
    "class": ["class"],
    "deg": ["angle"],
    "g/dL": ["g/dl"],
    "g/L": ["g/l"],
    "mg/dL": ["mg/dl"],
    "mg/L": ["mg/l", "mg/l(ddu)"],
    "ug/dL": ["ug/dl"],
    "ug/L": ["ug/l"],
    "ug/mL": ["ug/ml", "ug/mlfeu"],
    "ng/mL": ["ng/ml", "ng/mld-du"],
    "ng/dL": ["ng/dl"],
    "pg/mL": ["pg/ml"],
    "mmol/L": ["mmol/l"],
    "umol/L": ["umol/l", "um/l"],
    "nmol/L": ["nmol/l"],
    "nmol/mL": ["nm/ml"],
    "mEq/L": ["meq/l", "10*-3.eq/l"],
    "uEq/L": ["ueq/l", "10*-6.eq/l"],
    "/uL": ["/ul"],
    "10^3/uL": ["x10^3/ul", "10*3/ul", "x10*3/ul", "10^3/ul"],
    "10^6/uL": ["x10^6/ul", "x10*6/ul", "10*6/ul", "10^6/ul"],
    "10^6/mL": ["10*6/ml", "x10^6/ml"],
    "/HPF": ["/hpf", "/[hpf]"],
    "/LPF": ["/lpf"],
    "/100WBC": ["/wbc100"],
    "{copies}/mL": ["copies/ml", "{copies}/ml"],
    # U and IU are treated as the same unit within a test.
    "U/L": ["u/l", "[u]/l", "iu/l", "[iu]/l"],
    "U/mL": ["u/ml", "[u]/ml", "iu/ml", "[iu]/ml"],
    "kU/L": ["ku/l", "10*3.[u]/l"],
    "kU/mL": ["ku/ml"],
    "mIU/mL": ["miu/ml", "10*-3.[iu]/ml"],
    "uIU/mL": ["uiu/ml", "ulu/ml", "uu/ml", "miu/l", "10*-6.[iu]/ml", "10*-6.[u]/ml"],
    # Assay-specific units; never converted.
    "mAU/mL": ["mau/ml", "[arb'u]/l"],
    "EU/mL": ["eu/ml", "[eu]/ml"],
    "EU/dL": ["e.u/dl"],
    "RU/mL": ["ru/ml"],
    "RU": ["r.u"],
    "CU": ["cu"],
    "GPL": ["gpl", "[gpl'u]"],
    "MPL": ["mpl", "[mpl'u]"],
    "AAU": ["aau"],
    "ARU": ["aru"],
    "PRU": ["pru", "basepru"],
    "mosm/kg": ["mosm/kg", "mosm/k"],
    "fL": ["fl"],
    "pg": ["pg"],
    "mm[Hg]": ["mm[hg]", "mmhg"],
    "/min": ["/min"],
    "Cel": ["cel", "°c"],
    "kg": ["kg"],
    "cm": ["cm"],
    "mm": ["mm"],
    "kg/m2": ["kg/m2"],
    "m2": ["m^2"],
    "s": ["sec", "s"],
    "min": ["min"],
    "mm/h": ["mm/hr", "mm/h"],
    "L": ["l"],
    "mL": ["ml"],
    "L/min": ["l/min"],
    "L/s": ["l/sec", "l/s"],
    "mL/dL": ["ml/dl"],
    "mL/min/1.73m2": ["ml/min/1.73m^2"],
    "mL/min/mm[Hg]": ["ml/min/mmhg", "ml/min/mm[hg]"],
    "g/d": ["g/day"],
    "mg/d": ["mg/day", "mg/(24.h)"],
    "ug/d": ["ug/day", "ug/(24.h)"],
    "ug/min": ["ug/min"],
    "mmol/d": ["mmol/day", "mmol/d"],
    "mg/g": ["mg/g", "mg/gcreatinine"],
    "ug/g": ["ug/gcreatinine", "ug/gcrea.", "ug/g{creat}"],
    "ug/mg": ["ug/mgcr"],
    "nmol/mmol": ["nm/crmm", "nm/mmcr"],
    "nmol{BCE}/mmol": ["nmbce/mmcr", "nmbce/mmcre"],
    "ng/mL/h": ["ng/ml/hr"],
    "mg/kg": ["mg/kg"],
}
SPELLING_TO_UNIT = {
    spelling: unit for unit, spellings in CANONICAL_UNIT_SPELLINGS.items() for spelling in spellings
}

# unit: (dimension, scale); target value = value * scale / target scale.
UNIT_SCALES = {
    "g/dL": ("mass_per_volume", 10.0),
    "g/L": ("mass_per_volume", 1.0),
    "mg/dL": ("mass_per_volume", 1e-2),
    "mg/L": ("mass_per_volume", 1e-3),
    "ug/dL": ("mass_per_volume", 1e-5),
    "ug/L": ("mass_per_volume", 1e-6),
    "ug/mL": ("mass_per_volume", 1e-3),
    "ng/mL": ("mass_per_volume", 1e-6),
    "ng/dL": ("mass_per_volume", 1e-8),
    "pg/mL": ("mass_per_volume", 1e-9),
    "mmol/L": ("amount_per_volume", 1e-3),
    "umol/L": ("amount_per_volume", 1e-6),
    "nmol/L": ("amount_per_volume", 1e-9),
    "nmol/mL": ("amount_per_volume", 1e-6),
    "mEq/L": ("equivalent_per_volume", 1e-3),
    "uEq/L": ("equivalent_per_volume", 1e-6),
    "/uL": ("count_per_volume", 1.0),
    "10^3/uL": ("count_per_volume", 1e3),
    "10^6/uL": ("count_per_volume", 1e6),
    "10^6/mL": ("count_per_volume", 1e3),
    "U/L": ("units_per_volume", 1.0),
    "U/mL": ("units_per_volume", 1e3),
    "kU/L": ("units_per_volume", 1e3),
    "kU/mL": ("units_per_volume", 1e6),
    "mIU/mL": ("units_per_volume", 1.0),
    "uIU/mL": ("units_per_volume", 1e-3),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "L": ("volume", 1.0),
    "mL": ("volume", 1e-3),
    "L/min": ("flow", 1.0),
    "L/s": ("flow", 60.0),
    "mm": ("length", 1e-3),
    "cm": ("length", 1e-2),
    "g/d": ("mass_per_day", 1.0),
    "mg/d": ("mass_per_day", 1e-3),
    "ug/d": ("mass_per_day", 1e-6),
    "ug/min": ("mass_per_day", 1e-6 * 1440),
    "mg/g": ("mass_ratio", 1e-3),
    "ug/g": ("mass_ratio", 1e-6),
    "ug/mg": ("mass_ratio", 1e-3),
}

# Per-analyte mass/amount scales: Mg 24.305 g/mol, Na 22.990 g/mol; urobilinogen 1 EU/dL = 1 mg/dL.
ANALYTE_UNIT_SCALES = {
    3021197: {"mmol/L": ("magnesium_amount", 1.0), "mg/dL": ("magnesium_amount", 10 / 24.305)},
    3003181: {"mmol/L": ("sodium_amount", 1.0), "mg/dL": ("sodium_amount", 10 / 22.990)},
    3037426: {"mg/dL": ("urobilinogen", 1.0), "EU/dL": ("urobilinogen", 1.0)},
}


def unit_scale(concept_id: int, unit: str) -> tuple[str, float]:
    """(dimension, scale) of a unit for this concept; an unknown unit is its own dimension."""
    analyte = ANALYTE_UNIT_SCALES.get(concept_id, {})
    if unit in analyte:
        return analyte[unit]
    return UNIT_SCALES.get(unit, (f"unit:{unit}", 1.0))


def interpret_unit(
    unit_id: int, source_text: str, ucum_codes: dict[int, str]
) -> tuple[str | None, str, bool]:
    """Return (canonical unit or None, evidence, text/ID disagree); unit text wins over unit_concept_id."""
    text_unit = SPELLING_TO_UNIT.get(normalize_unit_text(source_text))
    code = ucum_codes.get(unit_id)
    id_unit = SPELLING_TO_UNIT.get(normalize_unit_text(code)) if code else None
    disagree = bool(text_unit and id_unit and text_unit != id_unit)
    if normalize_unit_text(source_text):
        if text_unit:
            return text_unit, "unit_text", disagree
        return None, "unknown_unit_text", False
    if id_unit:
        return id_unit, "unit_concept_id", False
    return None, "unit_missing", False


def collect_observed_units(config: dict, resume: bool = False) -> Path:
    destination = output_file(config, "units", "observed_units.parquet")
    if destination.exists() and resume and "median_value" in pl.scan_parquet(destination).collect_schema():
        return destination
    if destination.exists() and not resume:
        raise FileExistsError(f"Unit counts already exist: {destination}")

    scans = []
    for domain in RESULT_DOMAINS:
        source = output_file(config, "results", f"{domain}.parquet")
        if not source.exists():
            raise FileNotFoundError(f"Run result parsing first: {source}")
        counts = (
            pl.scan_parquet(source)
            .filter(
                (pl.col("concept_id") > 0)
                & pl.col("numeric_value_raw").is_not_null()
                & pl.col("numeric_value_raw").is_finite()
            )
            .with_columns(
                pl.col("unit_concept_id").fill_null(0).alias("unit_concept_id"),
                pl.col("unit_source_value").fill_null("").alias("unit_source_value"),
            )
            .group_by("domain", "concept_id", "unit_concept_id", "unit_source_value")
            .agg(
                pl.len().alias("numeric_rows"),
                pl.col("numeric_value_raw").median().alias("median_value"),
            )
        )
        scans.append(counts)

    temp = destination.with_name(destination.name + ".tmp")
    if temp.exists():
        temp.unlink()
    pl.concat(scans).sink_parquet(temp, compression="zstd", engine="streaming")
    temp.replace(destination)
    return destination


def load_unit_concepts(config: dict, unit_ids: list[int]) -> dict[int, str]:
    """UCUM concept_code for each unit_concept_id found in CONCEPT."""
    if not unit_ids:
        return {}
    raw_root: Path = config["paths"]["raw_root"]
    rows = (
        pl.scan_parquet(source_parts(raw_root, "CONCEPT"))
        .filter(pl.col("concept_id").is_in(unit_ids) & (pl.col("vocabulary_id") == "UCUM"))
        .select("concept_id", "concept_code")
        .collect(engine="streaming")
        .to_dicts()
    )
    return {int(row["concept_id"]): row["concept_code"] for row in rows}


def conversion_contradicts_values(
    raw_median: float | None, factor: float, reference_median: float | None
) -> bool:
    """True if the unconverted median is closer (log scale) to the target median, i.e. the unit text is wrong."""
    if raw_median is None or reference_median is None or factor == 1.0:
        return False
    if raw_median <= 0 or reference_median <= 0:
        return False
    before = abs(math.log(raw_median / reference_median))
    after = abs(math.log(raw_median * factor / reference_median))
    return before < after


def build_conversion_rows(signatures: list[dict], ucum_codes: dict[int, str]) -> list[dict]:
    """Target unit, factor and status per signature; the target is the concept's most frequent unit.

    Unknown units and unit_text_mismatch keep factor 1.0 in the target group; other dimensions get their own group.
    """
    by_concept: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in signatures:
        unit, evidence, disagree = interpret_unit(
            int(row["unit_concept_id"]), row["unit_source_value"], ucum_codes
        )
        by_concept[(row["domain"], int(row["concept_id"]))].append(
            {**row, "canonical_unit": unit, "evidence": evidence, "unit_id_text_disagree": disagree}
        )

    conversion_rows = []
    for (domain, concept_id), rows in by_concept.items():
        unit_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if row["canonical_unit"]:
                unit_counts[row["canonical_unit"]] += int(row["numeric_rows"])
        target = sorted(unit_counts, key=lambda unit: (-unit_counts[unit], unit))[0] if unit_counts else None
        target_dimension, target_scale = unit_scale(concept_id, target) if target else (None, None)
        # Reference median: the target-unit spelling with the most rows.
        target_rows = [row for row in rows if target and row["canonical_unit"] == target]
        reference = max(target_rows, key=lambda row: int(row["numeric_rows"]), default=None)
        reference_median = reference.get("median_value") if reference else None
        for row in rows:
            unit = row["canonical_unit"]
            factor = 1.0
            if unit is None:
                status = row["evidence"]
                group = target or "unknown"
            elif unit == target:
                status, group = "same_unit", target
            else:
                dimension, scale = unit_scale(concept_id, unit)
                if dimension == target_dimension:
                    factor = scale / target_scale
                    status, group = "converted", target
                    if conversion_contradicts_values(row.get("median_value"), factor, reference_median):
                        factor, status = 1.0, "unit_text_mismatch"
                else:
                    status, group = "separate_unit_group", unit
            conversion_rows.append({
                "domain": domain,
                "concept_id": concept_id,
                "unit_concept_id": int(row["unit_concept_id"]),
                "unit_source_value": row["unit_source_value"],
                "numeric_rows": int(row["numeric_rows"]),
                "median_value": row.get("median_value"),
                "canonical_unit": unit,
                "target_unit": target,
                "unit_group": group,
                "conversion_factor": factor,
                "unit_status": status,
                "evidence": row["evidence"],
                "unit_id_text_disagree": row["unit_id_text_disagree"],
            })
    return conversion_rows


def build_conversions(config: dict, resume: bool = False) -> Path:
    observed = collect_observed_units(config, resume)
    destination = output_file(config, "units", "conversions.parquet")
    if destination.exists() and resume:
        write_unit_summary(config, destination)
        return destination
    if destination.exists():
        raise FileExistsError(f"Conversion table already exists: {destination}")
    reference = reference_root(config)
    if reference is not None:
        # New site: reuse the reference table; unseen signatures stay unconverted and become [NUMERIC_UNSEEN].
        copy_reference_file(reference, config, "units", "conversions.parquet")
        write_unit_summary(config, destination)
        print(f"Using reference conversions: {reference / 'units' / 'conversions.parquet'}", flush=True)
        return destination

    small = pl.read_parquet(observed)
    unit_ids = [int(x) for x in small["unit_concept_id"].unique().to_list() if x > 0]
    conversion_rows = build_conversion_rows(small.to_dicts(), load_unit_concepts(config, unit_ids))

    temp = destination.with_name(destination.name + ".tmp")
    # Explicit schema so an empty table still has columns.
    conversion_schema = {
        "domain": pl.String, "concept_id": pl.Int64,
        "unit_concept_id": pl.Int64, "unit_source_value": pl.String,
        "numeric_rows": pl.UInt64, "median_value": pl.Float64, "canonical_unit": pl.String,
        "target_unit": pl.String, "unit_group": pl.String,
        "conversion_factor": pl.Float64, "unit_status": pl.String,
        "evidence": pl.String, "unit_id_text_disagree": pl.Boolean,
    }
    pl.DataFrame(conversion_rows, schema=conversion_schema).write_parquet(temp, compression="zstd")
    temp.replace(destination)
    write_unit_summary(config, destination)
    print(f"Saved {len(conversion_rows):,} unit conversions: {destination}", flush=True)
    return destination


def write_unit_summary(config: dict, conversion_path: Path) -> None:
    table = pl.read_parquet(conversion_path)
    report = (
        table.group_by("unit_status")
        .agg(pl.sum("numeric_rows").alias("numeric_rows"), pl.len().alias("unit_signatures"))
        .sort("numeric_rows", descending=True)
        .to_dicts()
    )
    write_json(
        output_file(config, "units", "summary.json"),
        {
            "method": "normalize unit text to canonical units; convert same-dimension units to the target by scale",
            "target": "canonical unit with the most numeric rows per concept",
            "conversion_source": "https://ucum.org/ucum",
            "status_counts": report,
            "unit_id_text_disagree_rows": int(
                table.filter(pl.col("unit_id_text_disagree"))["numeric_rows"].sum()
            ),
            "separate_unit_groups": int(table.filter(pl.col("unit_status") == "separate_unit_group").height),
            "unit_text_mismatch_rows": int(
                table.filter(pl.col("unit_status") == "unit_text_mismatch")["numeric_rows"].sum()
            ),
            "unknown_unit_text_rows": int(
                table.filter(pl.col("unit_status") == "unknown_unit_text")["numeric_rows"].sum()
            ),
            "reference_range_conversion": "same factor as the result",
            "unit_in_tokens": False,
        },
    )


# MIMIC-Extract variable_ranges.csv: concept_id: (name, outlier low, valid low, valid high, outlier high)
VITAL_LIMITS = {
    3027018: ("Heart rate", 0.0, 0.0, 350.0, 390.0),
    3024171: ("Respiratory rate", 0.0, 0.0, 300.0, 330.0),
    40762499: ("Oxygen saturation", 0.0, 0.0, 100.0, 150.0),
    3016502: ("Oxygen saturation", 0.0, 0.0, 100.0, 150.0),
    3004249: ("Systolic blood pressure", 0.0, 0.0, 375.0, 375.0),
    3012888: ("Diastolic blood pressure", 0.0, 0.0, 375.0, 375.0),
    3020891: ("Temperature", 14.2, 26.0, 45.0, 47.0),
    3025315: ("Weight", 0.0, 0.0, 250.0, 250.0),
    3036277: ("Height", 0.0, 0.0, 240.0, 275.0),
}
# Placeholders for unmeasured vitals (e.g. blood pressure 1 mmHg), not real measurements.
PLACEHOLDER_VITALS = {3027018, 3024171, 40762499, 3004249, 3012888}
PLACEHOLDER_VALUES = [-1.0, 0.0, 1.0]


def clean_vital_values(data: pl.LazyFrame) -> pl.LazyFrame:
    """Null placeholders and outliers, clip to the valid range; only numeric_value changes, rows are kept."""
    value = pl.col("numeric_value")
    placeholder = (
        pl.col("concept_id").is_in(list(PLACEHOLDER_VITALS)) & value.is_in(PLACEHOLDER_VALUES)
    ).fill_null(False)
    status = pl.when(placeholder).then(pl.lit("placeholder_removed"))
    cleaned = pl.when(placeholder).then(pl.lit(None, dtype=pl.Float64))
    for concept_id, (_name, outlier_low, valid_low, valid_high, outlier_high) in VITAL_LIMITS.items():
        this = pl.col("concept_id") == concept_id
        outlier = this & ((value < outlier_low) | (value > outlier_high))
        status = (
            status.when(outlier).then(pl.lit("outlier_removed"))
            .when(this & (value < valid_low)).then(pl.lit("clipped_low"))
            .when(this & (value > valid_high)).then(pl.lit("clipped_high"))
        )
        cleaned = (
            cleaned.when(outlier).then(pl.lit(None, dtype=pl.Float64))
            .when(this & (value < valid_low)).then(pl.lit(valid_low))
            .when(this & (value > valid_high)).then(pl.lit(valid_high))
        )
    return data.with_columns(
        cleaned.otherwise(value).alias("numeric_value"),
        status.otherwise(pl.lit(None, dtype=pl.String)).alias("vital_cleaning"),
    )


# One conversion-table row; reference ranges are checked per signature.
SIGNATURE_KEYS = ["domain", "concept_id", "_unit_id_key", "_unit_text_key"]
# Range is rejected when at least MIN_ROWS comparable rows and LIMIT share fall on one side.
RANGE_CHECK_MIN_ROWS = 20
RANGE_ONE_SIDED_LIMIT = 0.9


def converted_scan(config: dict, domain: str) -> pl.LazyFrame:
    """Converted values and ranges with cleaned vitals; keeps the _unit_id_key and _unit_text_key columns."""
    source = output_file(config, "results", f"{domain}.parquet")
    conversion = output_file(config, "units", "conversions.parquet")
    if not conversion.exists():
        raise FileNotFoundError("Build the unit conversion table first.")
    data = pl.scan_parquet(source).with_columns(
        pl.col("unit_concept_id").fill_null(0).alias("_unit_id_key"),
        pl.col("unit_source_value").fill_null("").alias("_unit_text_key"),
    )
    table = pl.scan_parquet(conversion).select(
        "domain", "concept_id",
        pl.col("unit_concept_id").alias("_unit_id_key"),
        pl.col("unit_source_value").alias("_unit_text_key"),
        "canonical_unit", "target_unit", "unit_group", "conversion_factor", "unit_status",
    )
    factor = pl.col("conversion_factor").fill_null(1.0)
    data = data.join(
        table, on=["domain", "concept_id", "_unit_id_key", "_unit_text_key"],
        how="left", validate="m:1",
    ).with_columns(
        (pl.col("numeric_value_raw") * factor).alias("numeric_value"),
        (pl.col("range_low") * factor).alias("range_low_converted"),
        (pl.col("range_high") * factor).alias("range_high_converted"),
        pl.when(pl.col("numeric_value_raw").is_null())
        .then(pl.lit("no_numeric_result"))
        .otherwise(pl.col("unit_status").fill_null("no_conversion_signature"))
        .alias("unit_status"),
    )
    if "numeric_censoring" not in data.collect_schema().names():
        data = data.with_columns(pl.lit(None, dtype=pl.String).alias("numeric_censoring"))
    return clean_vital_values(data).with_columns(range_status_expression().alias("range_status"))


def valid_range_expression() -> pl.Expr:
    low, high = pl.col("range_low_converted"), pl.col("range_high_converted")
    return (
        low.is_not_null() & high.is_not_null() & low.is_finite() & high.is_finite() & (low < high)
    ).fill_null(False)


def range_status_expression() -> pl.Expr:
    """LOW/HIGH/NORMAL against the row's range, else None; '<X' is never HIGH and '>X' never LOW."""
    value = pl.col("numeric_value")
    censoring = pl.col("numeric_censoring").fill_null("")
    comparable = (value.is_not_null() & value.is_finite()).fill_null(False) & valid_range_expression()
    return (
        pl.when(~comparable).then(pl.lit(None, dtype=pl.String))
        .when((value < pl.col("range_low_converted")) & (censoring != "greater")).then(pl.lit("LOW"))
        .when((value > pl.col("range_high_converted")) & (censoring != "less")).then(pl.lit("HIGH"))
        .otherwise(pl.lit("NORMAL"))
    )


def reference_range_checks(data: pl.LazyFrame) -> pl.DataFrame:
    """Per unit spelling, check that results and reference ranges share a scale (failures fall back to deciles).

    OK when the median is within [low / 10, high * 10] and, with >= 20 rows, under 90% are LOW or HIGH.
    """
    status = pl.col("range_status")
    stats = (
        data.filter(status.is_not_null())
        .group_by(SIGNATURE_KEYS)
        .agg(
            pl.len().alias("range_rows"),
            pl.col("numeric_value").median().alias("median_value"),
            pl.col("range_low_converted").median().alias("median_range_low"),
            pl.col("range_high_converted").median().alias("median_range_high"),
            (status == "LOW").mean().alias("low_share"),
            (status == "HIGH").mean().alias("high_share"),
        )
        .collect(engine="streaming")
    )
    # Zero or negative bounds (e.g. base excess) skip the scale check.
    low, high, median = pl.col("median_range_low"), pl.col("median_range_high"), pl.col("median_value")
    scale_ok = ((low <= 0) | (median >= low / 10)) & ((high <= 0) | (median <= high * 10))
    one_sided = (pl.col("range_rows") >= RANGE_CHECK_MIN_ROWS) & (
        (pl.col("low_share") >= RANGE_ONE_SIDED_LIMIT) | (pl.col("high_share") >= RANGE_ONE_SIDED_LIMIT)
    )
    return stats.with_columns(
        scale_ok.alias("range_scale_ok"),
        one_sided.alias("range_one_sided"),
        (scale_ok & ~one_sided).alias("range_check_ok"),
    )


def apply_unit_conversions(config: dict, domain: str, resume: bool = False) -> None:
    destination = output_file(config, "units", f"{domain}.parquet")
    if destination.exists() and resume and "range_usable" in pl.scan_parquet(destination).collect_schema():
        write_converted_domain_summary(config, domain, destination)
        return
    if destination.exists() and not resume:
        raise FileExistsError(f"Converted unit file already exists: {destination}")

    reference = reference_root(config)
    if reference is not None:
        # New site: reuse reference range checks; spellings unseen there get no range.
        checks = pl.read_parquet(reference / "units" / f"{domain}_range_checks.parquet").rename(
            {"unit_concept_id": "_unit_id_key", "unit_source_value": "_unit_text_key"}
        )
        copy_reference_file(reference, config, "units", f"{domain}_range_checks.parquet")
    else:
        checks = reference_range_checks(converted_scan(config, domain))
        checks.rename({"_unit_id_key": "unit_concept_id", "_unit_text_key": "unit_source_value"}).write_parquet(
            output_file(config, "units", f"{domain}_range_checks.parquet"), compression="zstd"
        )
    data = converted_scan(config, domain).join(
        checks.lazy().select(*SIGNATURE_KEYS, "range_check_ok"), on=SIGNATURE_KEYS, how="left",
    ).with_columns(
        (pl.col("range_status").is_not_null() & pl.col("range_check_ok").fill_null(False))
        .alias("range_usable")
    ).drop("_unit_id_key", "_unit_text_key", "range_check_ok")
    temp = destination.with_name(destination.name + ".tmp")
    if temp.exists():
        temp.unlink()
    data.sink_parquet(temp, compression="zstd", engine="streaming")
    temp.replace(destination)
    write_converted_domain_summary(config, domain, destination)
    print(f"{domain}: saved converted units {destination}", flush=True)


def write_converted_domain_summary(config: dict, domain: str, source: Path) -> None:
    data = pl.scan_parquet(source)
    statuses = data.group_by("unit_status").agg(
        pl.len().alias("rows"),
        pl.col("numeric_value").is_not_null().sum().alias("numeric_rows"),
    ).sort("rows", descending=True).collect(engine="streaming").to_dicts()
    vitals = data.filter(pl.col("vital_cleaning").is_not_null()).group_by("concept_id", "vital_cleaning").agg(
        pl.len().alias("rows")
    ).sort("concept_id", "vital_cleaning").collect(engine="streaming").to_dicts()
    counts = data.select(
        *[
            (pl.col(column).is_not_null() & ~pl.col(column).is_finite())
            .sum().alias(f"nonfinite_{column}_rows")
            for column in ("numeric_value", "range_low_converted", "range_high_converted")
        ],
        pl.col("range_usable").sum().alias("range_usable_rows"),
        (pl.col("range_usable") & (pl.col("range_status") == "LOW")).sum().alias("range_low_rows"),
        (pl.col("range_usable") & (pl.col("range_status") == "HIGH")).sum().alias("range_high_rows"),
        (pl.col("range_status").is_not_null() & ~pl.col("range_usable")).sum().alias("range_rejected_rows"),
        (
            ((pl.col("numeric_censoring") == "less") & (pl.col("range_status") == "NORMAL")
             & (pl.col("numeric_value") > pl.col("range_high_converted")))
            | ((pl.col("numeric_censoring") == "greater") & (pl.col("range_status") == "NORMAL")
               & (pl.col("numeric_value") < pl.col("range_low_converted")))
        ).sum().alias("censored_kept_normal_rows"),
        (pl.col("numeric_value").is_not_null() & pl.col("numeric_value").is_finite()).sum().alias("numeric_rows"),
    ).collect(engine="streaming").to_dicts()[0]
    write_json(output_file(config, "units", f"{domain}_summary.json"), {
        "domain": domain, "status_counts": statuses, "vital_cleaning": vitals, **counts,
    })


def check_unit_exceptions(config: dict) -> None:
    """Raise if converted values or ranges became NaN or Inf."""
    unexpected = {}
    for domain in RESULT_DOMAINS:
        path = output_file(config, "units", f"{domain}_summary.json")
        if not path.exists():
            raise FileNotFoundError(f"Unit summary not found: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if "range_usable_rows" not in summary:
            raise ValueError(
                f"{domain}: unit summary has an old format. Rerun values units."
            )
        counts = {key: value for key, value in summary.items() if key.startswith("nonfinite_") and value > 0}
        if counts:
            unexpected[domain] = counts
    if unexpected:
        raise ValueError(
            f"Saved unit exception types and counts: {unexpected}. "
            "Check the raw and converted rows in units/ and decide a handling rule first."
        )


def process_units(config: dict, resume: bool = False) -> None:
    check_event_exceptions(config)
    check_parsing_exceptions(config)
    build_conversions(config, resume)
    for domain in RESULT_DOMAINS:
        apply_unit_conversions(config, domain, resume)
    check_unit_exceptions(config)


# ── Deciles ──

QUANTILES = [i / 10 for i in range(1, 10)]
# Groups with at most this many distinct train values get one bin per value.
FEW_VALUE_LIMIT = 10


def load_train_numbers(config: dict) -> pl.LazyFrame:
    """Finite numeric results of train patients, dated on or before the export date."""
    export_date = export_finish_date(config["paths"]["raw_root"])
    scans = []
    for domain in RESULT_DOMAINS:
        source = output_file(config, "units", f"{domain}.parquet")
        if not source.exists():
            raise FileNotFoundError(f"Unit stage output not found: {source}")
        train_numbers = (
            pl.scan_parquet(source)
            .filter(
                (pl.col("split") == "train")
                & (pl.col("concept_id") > 0)
                & pl.col("event_date").is_not_null()
                & pl.col("datetime_date").is_not_null()
                & ~pl.col("date_mismatch")
                & (pl.col("event_date") <= pl.lit(export_date).str.to_date())
                & pl.col("numeric_value").is_not_null()
                & pl.col("numeric_value").is_finite()
            )
            .select("domain", "concept_id", "unit_group", "person_id", "numeric_value", "range_usable")
        )
        scans.append(train_numbers)
    return pl.concat(scans)


def choose_cutpoints(quantiles: list[float], smallest_values: list[float], unique_values: int) -> tuple[list[float], str]:
    """Return (cutpoints, rule): one bin per value if few values, else deduplicated quantiles above the minimum.

    A value equal to a cutpoint goes to the bin above it, so the minimum is always [Q1].
    """
    if unique_values <= FEW_VALUE_LIMIT:
        return sorted(smallest_values)[1:], "each_value"
    minimum = min(smallest_values)
    cutpoints = sorted({value for value in quantiles if value is not None and value > minimum})
    return cutpoints, "deduplicated_quantiles"


def calculate_decile_boundaries(all_train: pl.LazyFrame) -> pl.DataFrame:
    """Per group: raw quantiles q1-q9 (higher interpolation) and chosen cutpoints d1-d9, null-padded."""
    expressions = [
        pl.col("numeric_value").quantile(q, interpolation="higher").alias(f"q{i}")
        for i, q in enumerate(QUANTILES, start=1)
    ]
    groups = all_train.group_by("domain", "concept_id", "unit_group").agg(
        pl.len().alias("numeric_rows"),
        pl.col("person_id").n_unique().alias("patients"),
        pl.col("numeric_value").n_unique().alias("unique_values"),
        pl.col("numeric_value").unique().sort().head(FEW_VALUE_LIMIT + 1).alias("smallest_values"),
        *expressions,
    ).collect(engine="streaming")
    rows = []
    for row in groups.to_dicts():
        quantiles = [row[f"q{i}"] for i in range(1, 10)]
        cutpoints, rule = choose_cutpoints(quantiles, row["smallest_values"], row["unique_values"])
        padded = cutpoints + [None] * (9 - len(cutpoints))
        rows.append({
            **{key: row[key] for key in ("domain", "concept_id", "unit_group", "numeric_rows", "patients", "unique_values")},
            **{f"q{i}": quantiles[i - 1] for i in range(1, 10)},
            **{f"d{i}": padded[i - 1] for i in range(1, 10)},
            "cutpoints": len(cutpoints),
            "cut_rule": rule,
            "numeric_group_seen": True,
        })
    schema = {
        "domain": pl.String, "concept_id": pl.Int64, "unit_group": pl.String,
        "numeric_rows": pl.UInt32, "patients": pl.UInt32, "unique_values": pl.UInt32,
        **{f"q{i}": pl.Float64 for i in range(1, 10)},
        **{f"d{i}": pl.Float64 for i in range(1, 10)},
        "cutpoints": pl.Int8, "cut_rule": pl.String, "numeric_group_seen": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema)


def count_quantile_frequencies(
    all_train: pl.LazyFrame, boundaries: pl.LazyFrame
) -> pl.LazyFrame:
    """Q1-Q10 counts per group, empty bins as 0; a value equal to a cutpoint goes to the higher Q, as in tokens."""
    thresholds = [f"d{i}" for i in range(1, 10)]
    keys = ["domain", "concept_id", "unit_group"]
    data = all_train.join(
        boundaries.select(*keys, *thresholds), on=keys, how="left", validate="m:1",
    )
    q_number = pl.sum_horizontal(*[
        (pl.col("numeric_value") >= pl.col(column)).fill_null(False).cast(pl.Int8)
        for column in thresholds
    ]) + 1
    counts = data.with_columns(q_number.alias("q_number")).group_by(*keys, "q_number").agg(
        pl.len().alias("numeric_rows"),
        pl.col("person_id").n_unique().alias("patients"),
        pl.col("range_usable").sum().alias("reference_range_rows"),
        (~pl.col("range_usable")).sum().alias("decile_input_rows"),
    )
    groups = boundaries.select(*keys)
    q_numbers = pl.DataFrame({"q_number": range(1, 11)}, schema={"q_number": pl.Int8}).lazy()
    return groups.join(q_numbers, how="cross").join(
        counts, on=[*keys, "q_number"], how="left",
    ).with_columns(
        pl.col("numeric_rows", "patients", "reference_range_rows", "decile_input_rows")
        .fill_null(0)
    ).sort(*keys, "q_number")


def build_deciles(config: dict, resume: bool = False) -> None:
    check_event_exceptions(config)
    check_parsing_exceptions(config)
    check_unit_exceptions(config)
    destination = output_file(config, "deciles", "boundaries.parquet")
    if destination.exists() and not resume:
        raise FileExistsError(f"Boundaries already exist: {destination}")

    all_train = load_train_numbers(config)
    reference = reference_root(config)
    if reference is not None and not destination.exists():
        # New site: reuse reference boundaries; only the frequencies use this site's data.
        copy_reference_file(reference, config, "deciles", "boundaries.parquet")
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".tmp")
        calculate_decile_boundaries(all_train).write_parquet(temporary, compression="zstd")
        temporary.replace(destination)

    audited = pl.read_parquet(destination)
    frequencies = count_quantile_frequencies(all_train, audited.lazy())
    frequency_path = output_file(config, "deciles", "q_frequencies.parquet")
    temporary = frequency_path.with_name(frequency_path.name + ".tmp")
    frequencies.sink_parquet(temporary, compression="zstd", engine="streaming")
    temporary.replace(frequency_path)
    q_totals = pl.scan_parquet(frequency_path).group_by("q_number").agg(
        pl.sum("numeric_rows"), pl.sum("decile_input_rows"),
    ).sort("q_number").collect(engine="streaming").to_dicts()

    summary = {
        "method": "Polars higher quantiles, repeated cutpoints merged; one bin per value if at most 10 values",
        "source_split": "train only",
        "group_key": "domain, concept_id, unit_group",
        "groups_with_train_numbers": audited.height,
        "numeric_rows": int(audited["numeric_rows"].sum()),
        "groups_by_rule": audited.group_by("cut_rule").len().to_dicts(),
        "groups_with_fewer_than_9_cutpoints": int(audited.filter(pl.col("cutpoints") < 9).height),
        "groups_without_cutpoints": int(audited.filter(pl.col("cutpoints") == 0).height),
        "rare_groups_under_10_rows": int(audited.filter(pl.col("numeric_rows") < 10).height),
        "minimum_row_rule": "none",
        "same_value_rule": "a value equal to a cutpoint goes to the bin above it; equal numbers share a Q",
        "quantile_frequencies": q_totals,
        "frequency_path": str(frequency_path),
    }
    write_json(output_file(config, "deciles", "summary.json"), summary)
    print(
        f"{summary['numeric_rows']:,} train numeric rows, "
        f"saved boundaries for {summary['groups_with_train_numbers']:,} concept/unit groups",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    for stage, help_text in [
        ("parse", "read numbers and NEG/POS from result values"),
        ("units", "convert units and check reference ranges"),
        ("deciles", "compute decile boundaries from train patients"),
    ]:
        stages.add_parser(stage, parents=[common_args], help=help_text).add_argument(
            "--resume", action="store_true", help="reuse finished outputs and fill in missing summaries"
        )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "parse":
        parse_results(config, args.resume)
    elif args.stage == "units":
        process_units(config, args.resume)
    else:
        build_deciles(config, args.resume)


if __name__ == "__main__":
    main()
