"""Index points, labels and model inputs for los7, readmit30 and outpatient90 (EHRSHOT definitions).

Stages: labels (base preprocessing folder), inputs (model folder)
Usage: python -m ausom_ehr.tasks <stage> --config CONFIG
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ausom_ehr.common import config_parent, load_config, output_file, write_json
from ausom_ehr.tokens import (
    BUCKET_COUNT, DEMOGRAPHIC_TOKEN_COUNT, MAX_TOKENS, choose_recent_days, context_limit,
    flatten_days, load_demographics, special_ids, uses_demographic_tokens,
)


# ── Labels ──

# Inpatient, ER-to-inpatient and ICU visits; ICU stays are separate visits in AUSOM.
ADMISSION_TYPES = {9201, 262, 32037}
OUTPATIENT_TYPE = 9202
TASKS = ["los7", "readmit30", "outpatient90"]
OUTCOME_WINDOWS = {"readmit30": 30, "outpatient90": 90}
LONG_STAY = timedelta(hours=168)
OUTPATIENT_SAMPLE_LIMIT = 200_000
SAMPLE_SEED = 42


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def make_visit(row: tuple) -> dict:
    person_id, split, visit_id, visit_type, start_text, end_text = row
    return {
        "person_id": person_id,
        "split": split,
        "visit_id": visit_id,
        "type": visit_type,
        "start": parse_time(start_text),
        "end": parse_time(end_text),
    }


def merge_admissions(visits: list[dict]) -> list[dict]:
    """Merge admissions only when they overlap or touch; visits ending before they start are skipped."""
    admissions = sorted(
        (visit for visit in visits if visit["type"] in ADMISSION_TYPES),
        key=lambda visit: (visit["start"], visit["end"], visit["visit_id"]),
    )
    episodes = []
    current = None
    for visit in admissions:
        if visit["end"] < visit["start"]:
            continue
        if current is None or visit["start"] > current["end"]:
            current = {
                "start": visit["start"],
                "end": visit["end"],
                "first_visit_id": visit["visit_id"],
                "member_ids": [visit["visit_id"]],
                "terminal_visit_id": visit["visit_id"],
            }
            episodes.append(current)
        else:
            current["member_ids"].append(visit["visit_id"])
            if visit["end"] > current["end"]:
                current["end"] = visit["end"]
                current["terminal_visit_id"] = visit["visit_id"]
    return episodes


def admission_start_dates(visits: list[dict], episodes: list[dict]) -> list[date]:
    """Sorted admission start dates, including invalid visits (end < start) that still mark an admission."""
    starts = [episode["start"].date() for episode in episodes]
    starts += [
        visit["start"].date() for visit in visits
        if visit["type"] in ADMISSION_TYPES and visit["end"] < visit["start"]
    ]
    return sorted(starts)


def first_admission_after(start_dates: list[date], index_date: date) -> date | None:
    """First admission date strictly after index_date, or None."""
    position = bisect_right(start_dates, index_date)
    return start_dates[position] if position < len(start_dates) else None


def outcome_label(
    index_date: date, next_admission: date | None, death_date: date | None,
    last_record: date | None, days: int,
) -> tuple[int | None, str]:
    """(label, reason) for the window from the day after index_date to index_date + days.

    1 if an admission starts in it; None if death comes first or the record ends before the window end; else 0.
    """
    horizon = index_date + timedelta(days=days)
    has_admission = next_admission is not None and index_date < next_admission <= horizon
    has_death = death_date is not None and index_date < death_date <= horizon
    if has_death and (not has_admission or death_date <= next_admission):
        return None, "death_before_admission"
    if has_admission:
        return 1, "observed_admission"
    if last_record is None or last_record < horizon:
        return None, "record_ends_before_window"
    return 0, "no_admission_in_window"


def outpatient_overlaps_admission(outpatient: dict, episodes: list[dict]) -> bool:
    return any(
        outpatient["start"] <= episode["end"] and outpatient["end"] >= episode["start"]
        for episode in episodes
    )


def candidates_for_patient(
    visits: list[dict], death_date: date | None, last_record: date | None,
) -> tuple[list[dict], dict]:
    """Index samples of one patient and exclusion counts by reason."""
    episodes = merge_admissions(visits)
    start_dates = admission_start_dates(visits, episodes)
    patient_id = visits[0]["person_id"]
    split = visits[0]["split"]
    rows = []
    audit = {
        "invalid_visit_intervals": sum(visit["end"] < visit["start"] for visit in visits),
        "merged_admission_visits": sum(len(episode["member_ids"]) - 1 for episode in episodes),
        "index_on_or_after_death": 0,
        "los7_same_day_discharge": 0,
        "readmit30_same_day_new_admission": 0,
        "outpatient90_admission_overlap": 0,
        "outpatient90_same_day_admission": 0,
    }

    def add(task: str, index_date: date, index_time: datetime, visit_id: int,
            label: int | None, reason: str, days: int | None) -> None:
        # Visits on or after the death date cannot be index points.
        if death_date is not None and death_date <= index_date:
            audit["index_on_or_after_death"] += 1
            return
        key = f"{task}_{reason}"
        audit[key] = audit.get(key, 0) + 1
        if label is None:
            return
        rows.append({
            "person_id": patient_id,
            "split": split,
            "task": task,
            "cutoff_date": index_date,
            "index_datetime": index_time.isoformat(),
            "index_visit_id": visit_id,
            "label": label,
            "label_reason": reason,
            "result_window_days": days,
            "last_record_date": last_record,
        })

    for index, episode in enumerate(episodes):
        start_day = episode["start"].date()
        end_day = episode["end"].date()
        if end_day <= start_day:
            audit["los7_same_day_discharge"] += 1
        else:
            long_stay = int(episode["end"] - episode["start"] >= LONG_STAY)
            add("los7", start_day, episode["start"], episode["first_visit_id"],
                long_stay, "duration_168h", None)

        if any(
            other["start"].date() == end_day and other["start"] > episode["end"]
            for position, other in enumerate(episodes) if position != index
        ):
            audit["readmit30_same_day_new_admission"] += 1
            continue
        label, reason = outcome_label(
            end_day, first_admission_after(start_dates, end_day), death_date, last_record, 30,
        )
        add("readmit30", end_day, episode["end"], episode["terminal_visit_id"], label, reason, 30)

    for visit in visits:
        if visit["type"] != OUTPATIENT_TYPE or visit["end"] < visit["start"]:
            continue
        if outpatient_overlaps_admission(visit, episodes):
            audit["outpatient90_admission_overlap"] += 1
            continue
        index_day = visit["start"].date()
        if index_day in start_dates:
            audit["outpatient90_same_day_admission"] += 1
            continue
        label, reason = outcome_label(
            index_day, first_admission_after(start_dates, index_day), death_date, last_record, 90,
        )
        add("outpatient90", index_day, visit["start"], visit["visit_id"], label, reason, 90)

    return rows, audit


CANDIDATE_SCHEMA = pa.schema([
    ("person_id", pa.int64()), ("split", pa.string()), ("task", pa.string()),
    ("cutoff_date", pa.date32()), ("index_datetime", pa.string()),
    ("index_visit_id", pa.int64()), ("label", pa.int8()), ("label_reason", pa.string()),
    ("result_window_days", pa.int16()), ("last_record_date", pa.date32()),
])


def sample_outpatient(all_candidates: Path, destination: Path) -> dict:
    """Downsample outpatient samples above the cap with a fixed seed and write the final file."""
    data = pl.read_parquet(all_candidates).sort("task", "person_id", "cutoff_date", "index_visit_id")
    outpatient = data.filter(pl.col("task") == "outpatient90")
    eligible = outpatient.height
    if eligible > OUTPATIENT_SAMPLE_LIMIT:
        outpatient = outpatient.sample(n=OUTPATIENT_SAMPLE_LIMIT, seed=SAMPLE_SEED)
    final = pl.concat([data.filter(pl.col("task") != "outpatient90"), outpatient]).sort(
        "person_id", "task", "cutoff_date", "index_visit_id"
    )
    temporary = destination.with_name(destination.name + ".tmp")
    final.write_parquet(temporary, compression="zstd")
    temporary.replace(destination)
    return {"outpatient90_labeled_before_sampling": eligible, "outpatient90_sampled": outpatient.height}


def run_labels(config_path: str) -> None:
    config = load_config(config_path)
    root: Path = config["paths"]["output_root"]
    destination = output_file(config, "cohorts", "candidates.parquet")
    if destination.exists():
        raise FileExistsError(f"candidates already exist: {destination}")
    record_end_path = root / "cohorts" / "record_end.parquet"
    if not record_end_path.exists():
        raise FileNotFoundError("record_end.parquet not found; run data visits first.")
    deaths = pl.read_parquet(root / "cohorts" / "death.parquet").select("person_id", "death_date").iter_rows()
    death_by_person = {}
    for person_id, day in deaths:
        if day:
            # Several death rows: keep the earliest date.
            parsed = date.fromisoformat(day[:10])
            death_by_person[person_id] = min(parsed, death_by_person.get(person_id, parsed))
    last_record_by_person = dict(pl.read_parquet(record_end_path).iter_rows())
    visits = (
        pl.read_parquet(
            root / "cohorts" / "visits.parquet",
            columns=[
                "person_id", "split", "visit_occurrence_id", "visit_concept_id",
                "visit_start_datetime", "visit_end_datetime",
            ],
        )
        .sort("person_id", "visit_start_datetime", "visit_occurrence_id")
    )
    all_path = destination.with_name("all_candidates.parquet.tmp")
    if all_path.exists():
        all_path.unlink()
    writer = pq.ParquetWriter(all_path, CANDIDATE_SCHEMA, compression="zstd")
    current_person = None
    patient_visits = []
    batch = []
    audit_total = {}

    def finish_patient() -> None:
        nonlocal patient_visits, batch
        if not patient_visits:
            return
        rows, audit = candidates_for_patient(
            patient_visits, death_by_person.get(current_person), last_record_by_person.get(current_person),
        )
        batch.extend(rows)
        for key, count in audit.items():
            audit_total[key] = audit_total.get(key, 0) + count
        patient_visits = []
        if len(batch) >= 10_000:
            writer.write_table(pa.Table.from_pylist(batch, schema=CANDIDATE_SCHEMA))
            batch = []

    for row in visits.iter_rows():
        if row[0] != current_person:
            finish_patient()
            current_person = row[0]
        patient_visits.append(make_visit(row))
    finish_patient()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=CANDIDATE_SCHEMA))
    writer.close()
    sampling = sample_outpatient(all_path, destination)
    all_path.unlink()

    by_task = (
        pl.scan_parquet(destination)
        .group_by("task", "split")
        .agg(
            pl.len().alias("samples"),
            pl.col("person_id").n_unique().alias("patients"),
            pl.col("label").cast(pl.Int64).sum().alias("positives"),
        )
        .sort("task", "split")
        .collect()
        .to_dicts()
    )
    write_json(
        output_file(config, "cohorts", "labels_summary.json"),
        {
            "definition_source": "same conditions as EHRSHOT Long LOS and 30-day Readmission (outpatient90 applies the readmission rule to outpatient visits)",
            "sampling_unit": "one index point (admission, discharge or outpatient visit); several samples per patient, patient-level split",
            "prediction_time": "23:59 on the index date",
            "admission_visit_concepts": sorted(ADMISSION_TYPES),
            "negative_label_rule": "negative only if no new admission in the window and the last record date is on or after the window end",
            "discharge_disposition_used": False,
            "outpatient_sample_limit": OUTPATIENT_SAMPLE_LIMIT,
            "outpatient_sample_seed": SAMPLE_SEED,
            **sampling,
            "source_rows": visits.height,
            "audit": dict(sorted(audit_total.items())),
            "samples_by_task_split": by_task,
            "ready_for_cohort_inputs": True,
        },
    )
    print(f"saved {sum(row['samples'] for row in by_task):,} index samples", flush=True)


# ── Inputs ──

FINAL_DISCHARGE_DIAGNOSIS = 4230359


def make_prefix(
    events: list[tuple],
    cutoff_date,
    visits_by_id: dict[int, tuple],
    ids: dict[str, int],
    max_tokens: int = MAX_TOKENS,
    person: tuple | None = None,
) -> tuple[list[int], dict]:
    """Input tokens from events up to 23:59 on cutoff_date, and a report of dropped and truncated events.

    events: (date, tokens, visit_id, domain, condition_status) sorted by date; person adds sex/age tokens after BOS.
    """
    days = []
    current_day = None
    day_tokens = []
    day_events = 0
    retained_events = 0
    excluded_future_date = 0
    excluded_final_diagnosis = 0
    for event_date, event_tokens, visit_id, domain, condition_status in events:
        if event_date > cutoff_date:
            excluded_future_date += 1
            continue
        if domain == "condition" and condition_status == FINAL_DISCHARGE_DIAGNOSIS:
            visit = visits_by_id.get(visit_id) if visit_id is not None else None
            # Drop final diagnoses of visits still open at the cutoff or with end < start.
            if visit is not None and (visit[1] < visit[0] or visit[1] > cutoff_date):
                excluded_final_diagnosis += 1
                continue
        if event_date != current_day:
            if current_day is not None:
                days.append((current_day, day_tokens, day_events))
            current_day = event_date
            day_tokens = []
            day_events = 0
        day_tokens.extend(event_tokens)
        day_events += 1
        retained_events += 1
    if current_day is not None:
        days.append((current_day, day_tokens, day_events))
    reserved = DEMOGRAPHIC_TOKEN_COUNT if person is not None else 0
    chosen = choose_recent_days(days, max_tokens, eos=False, reserved=reserved)
    prefix = flatten_days(chosen, ids, eos=False, person=person) if chosen else []
    # BOS + demographic tokens + one DAY_GAP per day + event tokens, without rebuilding the full sequence.
    full_tokens = 1 + reserved + len(days) + sum(len(day[1]) for day in days) if days else 0
    latest_day_tokens = 2 + reserved + len(days[-1][1]) if days else 0
    report = {
        "pre_cutoff_events": retained_events,
        "excluded_future_date_events": excluded_future_date,
        "excluded_final_diagnosis_events": excluded_final_diagnosis,
        "truncated_days": len(days) - len(chosen),
        "truncated_events": retained_events - sum(day[2] for day in chosen),
        "truncated_tokens": full_tokens - len(prefix),
        # Events exist but the latest day alone exceeds max_tokens.
        "oversized_latest_day": bool(days) and not chosen,
        "latest_day_required_tokens": latest_day_tokens,
    }
    return prefix, report


INPUT_SCHEMA = pa.schema([
    ("person_id", pa.int64()), ("split", pa.string()), ("task", pa.string()),
    ("cutoff_date", pa.date32()), ("index_datetime", pa.string()),
    ("index_visit_id", pa.int64()), ("label", pa.int8()),
    ("input_tokens", pa.list_(pa.int32())), ("input_length", pa.int32()),
    ("pre_cutoff_events", pa.int32()),
    ("excluded_future_date_events", pa.int32()),
    ("excluded_final_diagnosis_events", pa.int32()),
    ("truncated_days", pa.int32()), ("truncated_events", pa.int32()),
    ("truncated_tokens", pa.int32()),
    ("oversized_latest_day", pa.bool_()),
    ("latest_day_required_tokens", pa.int32()),
])
EXCEPTION_SCHEMA = pa.schema([
    ("person_id", pa.int64()), ("split", pa.string()), ("task", pa.string()),
    ("cutoff_date", pa.date32()), ("index_visit_id", pa.int64()), ("label", pa.int8()),
    ("reason", pa.string()), ("max_context", pa.int32()),
    ("latest_day_required_tokens", pa.int32()),
])


def load_candidates(root: Path, bucket: int) -> pl.DataFrame:
    candidates = (
        pl.scan_parquet(root / "cohorts" / "candidates.parquet")
        .filter((pl.col("person_id") % BUCKET_COUNT == bucket) & pl.col("label").is_not_null())
        .collect()
    )
    return candidates.sort("person_id", "task", "cutoff_date", "index_datetime", "index_visit_id")


def process_input_bucket(config: dict, bucket: int, ids: dict[str, int]) -> tuple[Path, dict]:
    """Write model inputs for one bucket; returns (inputs path, audit by task and split)."""
    root: Path = config["paths"]["output_root"]
    max_tokens = int(config.get("model", {}).get("max_context", MAX_TOKENS))
    source = root / "tokens" / "events" / f"bucket={bucket}"
    event_files = list(source.glob("*.parquet"))
    if not event_files and not (root / "tokens" / "event_summary.json").exists():
        raise FileNotFoundError("event_summary.json not found; run tokens events (tokens joint-events in a model folder) first.")
    destination = output_file(config, "cohorts", "inputs", f"bucket={bucket:03d}.parquet")
    if destination.exists():
        raise FileExistsError(f"task inputs already exist: {destination}")

    visits = (
        pl.scan_parquet(root / "cohorts" / "visits.parquet")
        .filter(pl.col("person_id") % BUCKET_COUNT == bucket)
        .select("visit_occurrence_id", "visit_start_datetime", "visit_end_datetime")
        .collect()
    )
    visits_by_id = {
        visit_id: (date.fromisoformat(start[:10]), date.fromisoformat(end[:10]))
        for visit_id, start, end in visits.iter_rows()
    }
    by_person = defaultdict(list)
    for row in load_candidates(root, bucket).iter_rows(named=True):
        by_person[row["person_id"]].append(row)
    demographics = load_demographics(config, bucket) if uses_demographic_tokens(config) else None

    event_columns = [
        "person_id", "event_date", "domain_order", "event_id", "event_tokens",
        "visit_occurrence_id", "domain", "condition_status_concept_id",
    ]
    if event_files:
        events = (
            pl.read_parquet(event_files, columns=event_columns)
            .sort("person_id", "event_date", "domain_order", "event_id")
            .select("person_id", "event_date", "event_tokens", "visit_occurrence_id",
                    "domain", "condition_status_concept_id")
        )
    else:
        events = pl.DataFrame(schema={
            "person_id": pl.Int64, "event_date": pl.Date,
            "event_tokens": pl.List(pl.Int32), "visit_occurrence_id": pl.Int64,
            "domain": pl.String, "condition_status_concept_id": pl.Int64,
        })

    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = pq.ParquetWriter(temporary, INPUT_SCHEMA, compression="zstd")
    exception_path = output_file(config, "cohorts", "input_exceptions", f"bucket={bucket:03d}.parquet")
    exception_temp = exception_path.with_name(exception_path.name + ".tmp")
    exception_writer = pq.ParquetWriter(exception_temp, EXCEPTION_SCHEMA, compression="zstd")
    batch = []
    exception_batch = []
    split_audit = defaultdict(lambda: defaultdict(int))
    processed = set()
    person_id = None
    person_events = []

    def finish_patient() -> None:
        nonlocal batch, exception_batch, person_events
        if person_id is None:
            return
        processed.add(person_id)
        person = None
        if demographics is not None:
            person = demographics.get(person_id, (None, None, None))
        # Samples on the same cutoff date share one input across tasks.
        prefixes = {}
        for option in by_person.get(person_id, []):
            counts = split_audit[(option["task"], option["split"])]
            counts["labeled_samples"] += 1
            if option["cutoff_date"] not in prefixes:
                prefixes[option["cutoff_date"]] = make_prefix(
                    person_events, option["cutoff_date"], visits_by_id, ids, max_tokens, person,
                )
            prefix, report = prefixes[option["cutoff_date"]]
            if report["oversized_latest_day"]:
                counts["oversized_latest_day_samples"] += 1
                exception_batch.append({
                    "person_id": person_id, "split": option["split"], "task": option["task"],
                    "cutoff_date": option["cutoff_date"], "index_visit_id": option["index_visit_id"],
                    "label": option["label"], "reason": "latest_day_exceeds_context",
                    "max_context": max_tokens,
                    "latest_day_required_tokens": report["latest_day_required_tokens"],
                })
                continue
            if not prefix:
                counts["empty_history_samples"] += 1
                continue
            counts["selected_samples"] += 1
            batch.append({
                "person_id": person_id,
                "split": option["split"],
                "task": option["task"],
                "cutoff_date": option["cutoff_date"],
                "index_datetime": option["index_datetime"],
                "index_visit_id": option["index_visit_id"],
                "label": option["label"],
                "input_tokens": prefix,
                "input_length": len(prefix),
                **report,
            })
        person_events = []
        if len(batch) >= 256:
            writer.write_table(pa.Table.from_pylist(batch, schema=INPUT_SCHEMA))
            batch = []
        if len(exception_batch) >= 256:
            exception_writer.write_table(pa.Table.from_pylist(exception_batch, schema=EXCEPTION_SCHEMA))
            exception_batch = []

    for row in events.iter_rows():
        if row[0] != person_id:
            finish_patient()
            person_id = row[0]
        person_events.append(row[1:])
    finish_patient()
    # Patients without events are not visited above; count their empty-history samples here.
    for candidate_patient in sorted(by_person):
        if candidate_patient not in processed:
            person_id = candidate_patient
            person_events = []
            finish_patient()
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=INPUT_SCHEMA))
    if exception_batch:
        exception_writer.write_table(pa.Table.from_pylist(exception_batch, schema=EXCEPTION_SCHEMA))
    writer.close()
    exception_writer.close()
    temporary.replace(destination)
    exception_temp.replace(exception_path)
    print(f"bucket {bucket:03d}: saved inputs and exception audit", flush=True)
    return destination, {
        "by_task_split": [
            {"task": task, "split": split, **dict(counts)}
            for (task, split), counts in sorted(split_audit.items())
        ],
        "exception_path": str(exception_path),
    }


def summarize_inputs(config: dict, split_audits: list[dict], exception_paths: list[str]) -> None:
    """Save counts per task and split; raise if any sample has a single day longer than the context."""
    root = config["paths"]["output_root"] / "cohorts" / "inputs"
    data = pl.scan_parquet(str(root / "bucket=*.parquet"))
    by_task = (
        data.group_by("task", "split")
        .agg(
            pl.len().alias("samples"),
            pl.col("person_id").n_unique().alias("patients"),
            pl.col("label").cast(pl.Int64).sum().alias("positives"),
            pl.sum("excluded_final_diagnosis_events").alias("excluded_final_diagnosis_events"),
            pl.sum("truncated_days").alias("truncated_days"),
            pl.sum("truncated_events").alias("truncated_events"),
            pl.sum("truncated_tokens").alias("truncated_tokens"),
            pl.col("input_length").median().alias("median_input_length"),
            pl.col("input_length").max().alias("max_input_length"),
        )
        .sort("task", "split")
        .collect()
        .to_dicts()
    )
    pending = sum(row.get("oversized_latest_day_samples", 0) for row in split_audits)
    write_json(
        output_file(config, "cohorts", "inputs_summary.json"),
        {
            "cutoff": "events up to 23:59 on the index date (same time rule as EHRSHOT)",
            "excluded_before_cutoff": "final discharge diagnoses of visits not yet ended",
            "prefix_has_synthetic_eos": False,
            "demographic_tokens": uses_demographic_tokens(config),
            "sampling_unit": "one sample per index point",
            "audit_by_task_split": split_audits,
            "by_task_split": by_task,
            "ready_for_embedding": pending == 0,
            "pending_oversized_day_samples": pending,
            "input_exception_paths": exception_paths,
        },
    )
    print("task input summary written", flush=True)
    if pending:
        raise ValueError(
            f"{pending} samples exceed the max input length with a single day of records. "
            "Check cohorts/inputs_summary.json and input_exceptions/, then decide a handling rule first."
        )


def run_inputs(config_path: str) -> None:
    config = load_config(config_path)
    # The CLI enforces the 16,384 limit; tests with small contexts call process_input_bucket directly.
    context_limit(config)
    labels_summary = output_file(config, "cohorts", "labels_summary.json")
    if labels_summary.exists():
        labels_audit = json.loads(labels_summary.read_text(encoding="utf-8"))
        if labels_audit.get("ready_for_cohort_inputs") is False:
            raise ValueError("cohort labels have unresolved policies; check labels_summary.json.")
    ids = special_ids(output_file(config, "tokens", "vocab.parquet"))
    split_totals = defaultdict(lambda: defaultdict(int))
    exception_paths = []
    for bucket in range(BUCKET_COUNT):
        _path, bucket_audit = process_input_bucket(config, bucket, ids)
        for row in bucket_audit["by_task_split"]:
            if row.get("oversized_latest_day_samples", 0):
                exception_paths.append(bucket_audit["exception_path"])
            for reason, count in row.items():
                if reason not in {"task", "split"}:
                    split_totals[(row["task"], row["split"])][reason] += count
    split_rows = [
        {"task": task, "split": split, **dict(counts)}
        for (task, split), counts in sorted(split_totals.items())
    ]
    summarize_inputs(config, split_rows, sorted(set(exception_paths)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    common_args = config_parent()
    stages.add_parser("labels", parents=[common_args], help="index points and labels (base preprocessing folder)")
    stages.add_parser("inputs", parents=[common_args], help="model inputs per index point (model folder)")
    args = parser.parse_args()
    if args.stage == "labels":
        run_labels(args.config)
    else:
        run_inputs(args.config)


if __name__ == "__main__":
    main()
