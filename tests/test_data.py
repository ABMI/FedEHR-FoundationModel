"""Patient split, event extraction and exception audits on synthetic patients and relative dates."""

import json
from datetime import date, timedelta

import polars as pl
import pytest

from ausom_ehr import data as data_stage


PATIENT_A = 1
PATIENT_B = 2
OTHER_PATIENT = 3
SYNTHETIC_DAY_ZERO = date(2000, 1, 1)


def relative_date(days: int) -> str:
    return (SYNTHETIC_DAY_ZERO + timedelta(days=days)).isoformat()


def make_config(tmp_path):
    raw_root = tmp_path / "synthetic_raw"
    output_root = tmp_path / "synthetic_output"
    raw_root.mkdir()
    output_root.mkdir()
    marker = {"finished_at": relative_date(9) + "T00:00:00"}
    (raw_root / "_COMPLETE.json").write_text(json.dumps(marker))
    selected_path = output_root / "sample" / "selected_patients.parquet"
    selected_path.parent.mkdir()
    pl.DataFrame({
        "person_id": [PATIENT_A, PATIENT_B],
        "split": ["train", "test"],
    }).write_parquet(selected_path)
    return {"paths": {"raw_root": raw_root, "output_root": output_root}}, selected_path


def write_source(config, domain, person_ids, concepts, event_days, datetime_days):
    table, id_column, concept_column, date_column, datetime_column = data_stage.DOMAINS[domain]
    source_dir = config["paths"]["raw_root"] / "dbo" / table
    batch_dir = source_dir / "batch-synthetic"
    batch_dir.mkdir(parents=True)
    columns = {
        "person_id": person_ids,
        id_column: list(range(1, len(person_ids) + 1)),
        concept_column: concepts,
        date_column: [relative_date(day) if day is not None else None for day in event_days],
    }
    if datetime_days is not None:
        columns[datetime_column] = [
            relative_date(day) + " 12:00:00" if day is not None else None
            for day in datetime_days
        ]
    pl.DataFrame(columns).write_parquet(batch_dir / "part.parquet")
    (source_dir / "_COMPLETE.json").write_text(json.dumps({"rows_written": len(person_ids)}))


def test_sampling_is_reproducible_and_has_no_patient_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(data_stage, "SAMPLE_SIZE", 20)
    monkeypatch.setattr(data_stage, "TRAIN_SIZE", 14)
    monkeypatch.setattr(data_stage, "VALIDATION_SIZE", 3)
    source = tmp_path / "synthetic_person.parquet"
    pl.DataFrame({"person_id": list(range(1, 31)) + [PATIENT_A, None]}).write_parquet(source)

    first, population = data_stage.choose_patients(str(source))
    second, _ = data_stage.choose_patients(str(source))
    audit = data_stage.validate_patient_split(first.lazy())

    assert first.equals(second)
    assert population == 30
    assert first["person_id"].n_unique() == 20
    assert audit["split_counts"] == {"train": 14, "validation": 3, "test": 3}
    patient_splits = first.group_by("person_id").agg(pl.col("split").n_unique())
    assert patient_splits["split"].max() == 1


def test_patient_assigned_to_two_splits_is_rejected():
    selected = pl.DataFrame({
        "person_id": [PATIENT_A, PATIENT_A],
        "split": ["train", "test"],
    }).lazy()
    with pytest.raises(ValueError, match="Duplicate patient IDs"):
        data_stage.validate_patient_split(selected)


def test_extraction_keeps_repeated_and_missing_result_events_without_visit(tmp_path):
    config, selected_path = make_config(tmp_path)
    write_source(
        config, "measurement",
        [PATIENT_A, PATIENT_A, PATIENT_B, OTHER_PATIENT],
        [101, 101, 0, 101], [0, 0, None, 0], [0, 0, None, 0],
    )
    data = data_stage.build_common_scan(
        config["paths"]["raw_root"], selected_path, "measurement"
    ).collect(engine="streaming")

    assert data.height == 3
    assert data.filter(pl.col("concept_id") == 101).height == 2
    assert data["visit_occurrence_id"].null_count() == 3
    assert data["value_as_number"].null_count() == 3
    assert data["range_low"].null_count() == 3
    assert data.filter(pl.col("person_id") == PATIENT_A)["split"].to_list() == ["train", "train"]
    assert "age" not in data.columns
    assert "gender" not in data.columns


def test_missing_datetime_is_counted_before_policy_is_requested(tmp_path):
    config, selected_path = make_config(tmp_path)
    write_source(config, "measurement", [PATIENT_A], [101], [0], None)
    data = data_stage.build_common_scan(
        config["paths"]["raw_root"], selected_path, "measurement"
    ).collect(engine="streaming")
    destination = config["paths"]["output_root"] / "measurement.parquet"
    data.write_parquet(destination)
    summary = data_stage.summarize_output(
        destination, "MEASUREMENT", config["paths"]["raw_root"], "measurement"
    )

    assert summary["pending_decisions"] == {"missing_or_bad_datetime_rows": 1}
    assert summary["pending_event_rows"] == 1
    assert summary["model_event_candidate_rows"] == 0
    assert summary["status"] == "needs_user_decision"


def test_all_domain_audits_are_saved_before_date_exceptions_raise(tmp_path):
    config, _ = make_config(tmp_path)
    for domain in data_stage.DOMAINS:
        if domain == "condition":
            event_days, datetime_days = [0], [1]
        elif domain == "drug":
            event_days, datetime_days = [0], None
        elif domain == "procedure":
            event_days, datetime_days = [10], [10]
        elif domain == "observation":
            event_days, datetime_days = [10], [10]
        else:
            event_days, datetime_days = [0], [0]
        write_source(config, domain, [PATIENT_A], [101], event_days, datetime_days)
        data_stage.extract_one(config, domain, resume=False)

    with pytest.raises(ValueError, match="decide a handling rule first"):
        data_stage.check_event_exceptions(config)

    root = config["paths"]["output_root"]
    combined = json.loads((root / "events" / "exceptions_summary.json").read_text())
    domains = {row["domain"]: row for row in combined["domain_audits"]}
    assert set(domains) == set(data_stage.DOMAINS)
    assert domains["condition"]["pending_decisions"] == {"date_mismatch_rows": 1}
    assert domains["drug"]["pending_decisions"] == {"missing_or_bad_datetime_rows": 1}
    assert domains["procedure"]["pending_decisions"] == {"unconfirmed_future_rows": 1}
    assert domains["observation"]["pending_decisions"] == {}
    assert domains["observation"]["held_after_export_rows"] == 1
    observed = pl.read_parquet(root / "events" / "observation.parquet")
    assert observed["event_id"].to_list() == [1]
    assert observed["source_date"].to_list() == [relative_date(10)]
    assert observed["event_hold_reason"].to_list() == ["observation_after_export"]


def test_unmapped_date_exceptions_are_audited_without_new_policy(tmp_path):
    config, selected_path = make_config(tmp_path)
    write_source(config, "measurement", [PATIENT_A], [0], [None], None)
    data = data_stage.build_common_scan(
        config["paths"]["raw_root"], selected_path, "measurement"
    ).collect(engine="streaming")
    destination = config["paths"]["output_root"] / "unmapped.parquet"
    data.write_parquet(destination)
    summary = data_stage.summarize_output(
        destination, "MEASUREMENT", config["paths"]["raw_root"], "measurement"
    )

    assert summary["unmapped_rows"] == 1
    assert summary["missing_or_bad_date_rows"] == 1
    assert summary["missing_or_bad_datetime_rows"] == 1
    assert summary["pending_decisions"] == {}
    assert summary["model_event_candidate_rows"] == 0


def test_duplicates_ignore_source_code_for_mapped_concepts_only():
    from ausom_ehr.data import clean_event_rows

    rows = pl.DataFrame({
        "event_id": [1, 2, 3, 4],
        "person_id": [1, 1, 1, 1],
        "concept_id": [101, 101, 0, 0],
        # One value stored twice under two source codes, and two unmapped rows with different codes
        "source_code": ["M|a", "M|b", "U|x", "U|y"],
        "event_date": ["2020-01-01"] * 4,
        "source_datetime": ["2020-01-01T08:00"] * 4,
        "value_as_number": [48.0, 48.0, None, None],
    }).lazy()
    kept = clean_event_rows(rows, "measurement").collect().sort("event_id")
    assert kept["event_id"].to_list() == [1, 3, 4]
    assert "_unmapped_source_code" not in kept.columns


def write_raw_table(raw_root, table, frame):
    folder = raw_root / "dbo" / table / "batch-0"
    folder.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(folder / "part.parquet")


def test_discharge_codes_move_to_visit_end_only_for_multi_day_visits(tmp_path):
    raw_root = tmp_path / "synthetic_raw"
    # Visit 10: inpatient, days 0-9; visit 20: same-day outpatient
    write_raw_table(raw_root, "VISIT_OCCURRENCE", pl.DataFrame({
        "visit_occurrence_id": [10, 20],
        "visit_end_date": [relative_date(9), relative_date(30)],
        "visit_end_datetime": [f"{relative_date(9)}T11:00:00", f"{relative_date(30)}T09:00:00"],
    }))
    write_raw_table(raw_root, "CONCEPT", pl.DataFrame({
        "concept_id": [900, 901], "vocabulary_id": ["KCD7", "EDI"],
    }))
    day0 = SYNTHETIC_DAY_ZERO
    day30 = SYNTHETIC_DAY_ZERO + timedelta(days=30)
    conditions = pl.DataFrame({
        "person_id": [PATIENT_A] * 4,
        # final dx (inpatient), admission dx (inpatient), final dx (same-day outpatient), final dx without visit
        "condition_status_concept_id": [4230359, 4203942, 4230359, 4230359],
        "visit_occurrence_id": [10, 10, 20, None],
        "event_date": [day0, day0, day30, day0],
        "datetime_date": [day0, day0, day30, day0],
        "source_datetime": [f"{relative_date(0)}T00:00:00"] * 2 + [f"{relative_date(30)}T00:00:00", None],
    }, schema_overrides={"visit_occurrence_id": pl.Int64})
    moved = data_stage.move_codes_to_visit_end(conditions.lazy(), raw_root, "condition").collect()
    assert moved["moved_to_visit_end"].to_list() == [True, False, False, False]
    assert moved["event_date"].to_list() == [SYNTHETIC_DAY_ZERO + timedelta(days=9), day0, day30, day0]
    assert moved["datetime_date"][0] == SYNTHETIC_DAY_ZERO + timedelta(days=9)
    assert moved["source_datetime"][0] == f"{relative_date(9)}T11:00:00"
    assert moved["original_event_date"][0] == day0

    # Only observation rows from KCD diagnosis codes move.
    observations = pl.DataFrame({
        "person_id": [PATIENT_A] * 2,
        "source_concept_id": [900, 901],
        "visit_occurrence_id": [10, 10],
        "event_date": [day0, day0],
        "datetime_date": [day0, day0],
        "source_datetime": [f"{relative_date(0)}T00:00:00"] * 2,
    }, schema_overrides={"visit_occurrence_id": pl.Int64, "source_concept_id": pl.Int64})
    moved = data_stage.move_codes_to_visit_end(observations.lazy(), raw_root, "observation").collect()
    assert moved["moved_to_visit_end"].to_list() == [True, False]
    assert "_diagnosis_source" not in moved.columns


def test_visit_end_move_is_off_unless_configured():
    assert data_stage.moves_codes_to_visit_end({}) is False
    assert data_stage.moves_codes_to_visit_end({"events": {"move_discharge_codes_to_visit_end": True}}) is True
