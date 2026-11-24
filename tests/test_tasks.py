"""Index-sample inputs, empty histories and oversized-day exceptions on synthetic patients A/B."""

import json
from datetime import date, timedelta

import polars as pl
import pytest

from ausom_ehr import tasks


PATIENT_A = 1
PATIENT_B = 2
SYNTHETIC_DAY_ZERO = date(2000, 1, 1)
TOKEN_IDS = {
    "[BOS]": 1, "[DAY_GAP_0]": 2, "[DAY_GAP_1]": 3,
    "[SEX_M]": 4, "[SEX_F]": 5, "[SEX_UNKNOWN]": 6, "[AGE_30_34]": 7, "[AGE_UNKNOWN]": 8,
}


def day(offset):
    return SYNTHETIC_DAY_ZERO + timedelta(days=offset)


def make_candidate(patient, split, visit_id, offset=0, label=0, hour=8, task="los7"):
    return {
        "person_id": patient, "split": split, "task": task,
        "cutoff_date": day(offset),
        "index_datetime": f"{day(offset).isoformat()}T{hour:02d}:00:00",
        "index_visit_id": visit_id, "label": label,
    }


def make_bucket(tmp_path, candidates, events, max_context=16_384):
    output_root = tmp_path / "synthetic_output"
    cohort_root = output_root / "cohorts"
    cohort_root.mkdir(parents=True)
    candidate_schema = {
        "person_id": pl.Int64, "split": pl.String, "task": pl.String,
        "cutoff_date": pl.Date, "index_datetime": pl.String,
        "index_visit_id": pl.Int64, "label": pl.Int8,
    }
    pl.DataFrame(candidates, schema=candidate_schema).write_parquet(cohort_root / "candidates.parquet")
    pl.DataFrame(schema={
        "person_id": pl.Int64, "visit_occurrence_id": pl.Int64,
        "visit_start_datetime": pl.String, "visit_end_datetime": pl.String,
    }).write_parquet(cohort_root / "visits.parquet")
    token_root = output_root / "tokens"
    token_root.mkdir()
    (token_root / "event_summary.json").write_text(json.dumps({"synthetic": True}))
    if events:
        event_root = token_root / "events" / "bucket=0"
        event_root.mkdir(parents=True)
        rows = []
        for event_id, (patient, offset, tokens) in enumerate(events, start=1):
            rows.append({
                "person_id": patient, "event_date": day(offset),
                "domain_order": 0, "event_id": event_id,
                "event_tokens": tokens, "visit_occurrence_id": None,
                "domain": "condition", "condition_status_concept_id": None,
            })
        pl.DataFrame(rows, schema={
            "person_id": pl.Int64, "event_date": pl.Date, "domain_order": pl.Int32,
            "event_id": pl.Int64, "event_tokens": pl.List(pl.Int32),
            "visit_occurrence_id": pl.Int64, "domain": pl.String,
            "condition_status_concept_id": pl.Int64,
        }).write_parquet(event_root / "condition.parquet")
    return {"paths": {"output_root": output_root}, "model": {"max_context": max_context}}


def audit_by_split(audit):
    return {(row["task"], row["split"]): row for row in audit["by_task_split"]}


def test_oversized_latest_day_reports_all_discarded_tokens():
    events = [(day(0), [10, 11, 12, 13], None, "condition", None)]
    prefix, report = tasks.make_prefix(events, day(0), {}, TOKEN_IDS, max_tokens=5)

    assert prefix == []
    assert report["oversized_latest_day"] is True
    assert report["latest_day_required_tokens"] == 6
    assert report["truncated_days"] == 1
    assert report["truncated_events"] == 1
    assert report["truncated_tokens"] == 6


def test_prefix_starts_with_sex_and_age_at_first_kept_day():
    events = [(day(0), [10], None, "condition", None), (day(1), [11], None, "condition", None)]
    person = (8507, 1970, 1)  # male, age 30 in January 2000
    prefix, _report = tasks.make_prefix(events, day(1), {}, TOKEN_IDS, person=person)

    assert prefix == [1, 4, 7, 2, 10, 3, 11]


def test_every_labeled_sample_of_a_patient_gets_an_input(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "BUCKET_COUNT", 1)
    config = make_bucket(
        tmp_path,
        [make_candidate(PATIENT_A, "train", 10), make_candidate(PATIENT_A, "train", 11, offset=1, label=1)],
        [(PATIENT_A, 0, [10]), (PATIENT_A, 1, [11])],
    )
    path, audit = tasks.process_input_bucket(config, 0, TOKEN_IDS)

    rows = pl.read_parquet(path)
    assert rows["index_visit_id"].to_list() == [10, 11]
    assert rows["label"].to_list() == [0, 1]
    assert audit_by_split(audit)[("los7", "train")]["selected_samples"] == 2


def test_empty_history_sample_is_skipped_but_later_sample_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "BUCKET_COUNT", 1)
    config = make_bucket(
        tmp_path,
        [make_candidate(PATIENT_A, "train", 10), make_candidate(PATIENT_A, "train", 11, offset=1)],
        [(PATIENT_A, 1, [10])],
    )
    path, audit = tasks.process_input_bucket(config, 0, TOKEN_IDS)

    assert pl.read_parquet(path)["index_visit_id"].to_list() == [11]
    counts = audit_by_split(audit)[("los7", "train")]
    assert counts["empty_history_samples"] == 1
    assert counts["selected_samples"] == 1


def test_patient_absent_from_events_and_unlabeled_rows_are_handled(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "BUCKET_COUNT", 1)
    config = make_bucket(
        tmp_path,
        [
            make_candidate(PATIENT_A, "train", 10),
            make_candidate(PATIENT_B, "test", 20),
            make_candidate(PATIENT_B, "test", 21, offset=2, label=None),
        ],
        [(PATIENT_A, 0, [10])],
    )
    path, audit = tasks.process_input_bucket(config, 0, TOKEN_IDS)

    assert pl.read_parquet(path)["person_id"].to_list() == [PATIENT_A]
    counts = audit_by_split(audit)
    assert counts[("los7", "test")]["empty_history_samples"] == 1
    # Unlabeled (censored) rows get no input.
    assert counts[("los7", "test")]["labeled_samples"] == 1


def test_oversized_sample_is_preserved_and_blocks_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "BUCKET_COUNT", 1)
    config = make_bucket(
        tmp_path,
        [make_candidate(PATIENT_A, "train", 10), make_candidate(PATIENT_A, "train", 11, offset=1)],
        [(PATIENT_A, 0, [10, 11, 12, 13]), (PATIENT_A, 1, [10])],
        max_context=5,
    )
    path, audit = tasks.process_input_bucket(config, 0, TOKEN_IDS)

    # Day 0 alone exceeds the context; the day-1 sample keeps only its latest day.
    assert pl.read_parquet(path)["index_visit_id"].to_list() == [11]
    pending = pl.read_parquet(audit["exception_path"])
    assert pending["index_visit_id"].to_list() == [10]
    assert pending["latest_day_required_tokens"].to_list() == [6]
    with pytest.raises(ValueError, match="decide a handling rule first"):
        tasks.summarize_inputs(config, audit["by_task_split"], [audit["exception_path"]])
    summary = json.loads((config["paths"]["output_root"] / "cohorts" / "inputs_summary.json").read_text())
    assert summary["ready_for_embedding"] is False
    assert summary["pending_oversized_day_samples"] == 1


def test_main_finishes_every_bucket_before_oversized_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "BUCKET_COUNT", 2)
    # The CLI enforces the real 16,384 limit, so fill one day with enough 3-token events.
    oversized_day = [(PATIENT_B, 0, [10, 11, 12]) for _ in range(16_384 // 3 + 1)]
    config = make_bucket(
        tmp_path,
        [make_candidate(PATIENT_B, "train", 20), make_candidate(PATIENT_A, "test", 10)],
        oversized_day,
    )
    monkeypatch.setattr(tasks, "load_config", lambda path: config)
    monkeypatch.setattr(tasks, "special_ids", lambda path: TOKEN_IDS)
    monkeypatch.setattr("sys.argv", ["tasks", "inputs", "--config", "synthetic.toml"])

    with pytest.raises(ValueError, match="decide a handling rule first"):
        tasks.main()

    root = config["paths"]["output_root"] / "cohorts"
    assert (root / "inputs" / "bucket=000.parquet").exists()
    assert (root / "inputs" / "bucket=001.parquet").exists()
    summary = json.loads((root / "inputs_summary.json").read_text())
    split_rows = {(row["task"], row["split"]): row for row in summary["audit_by_task_split"]}
    assert summary["ready_for_embedding"] is False
    assert split_rows[("los7", "train")]["oversized_latest_day_samples"] == 1
    assert split_rows[("los7", "test")]["empty_history_samples"] == 1


def test_main_blocks_unresolved_label_policies_before_input_creation(tmp_path, monkeypatch):
    config = make_bucket(tmp_path, [make_candidate(PATIENT_A, "train", 10)], [])
    labels_summary = config["paths"]["output_root"] / "cohorts" / "labels_summary.json"
    labels_summary.write_text(json.dumps({"ready_for_cohort_inputs": False}))
    monkeypatch.setattr(tasks, "load_config", lambda path: config)
    monkeypatch.setattr("sys.argv", ["tasks", "inputs", "--config", "synthetic.toml"])

    with pytest.raises(ValueError, match="labels have unresolved policies"):
        tasks.main()
    assert not (config["paths"]["output_root"] / "cohorts" / "inputs").exists()
