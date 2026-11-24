"""Token representation, joint tokens, day limits and empty/oversized patients on synthetic events."""

import json
from datetime import date

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ausom_ehr import tokens as token_stage
from ausom_ehr.tokens import (
    context_limit, process_sequence_bucket, record_for_patient, special_ids, summarize_sequences,
    value_keys,
)


def token_ids():
    return {token: index for index, token in enumerate(token_stage.SPECIAL_TOKENS)}


def numeric_events(domain, values, concepts, qualities, range_usable, lows, highs, unit_group="mg/dL"):
    count = len(values)
    statuses = []
    for value, low, high in zip(values, lows, highs):
        if value is None or low is None or high is None:
            statuses.append(None)
        else:
            statuses.append("LOW" if value < low else "HIGH" if value > high else "NORMAL")
    return pl.DataFrame({
        "person_id": [1] * count,
        "split": ["train"] * count,
        "domain": [domain] * count,
        "concept_id": concepts,
        "event_date": [date(2000, 1, 1)] * count,
        "datetime_date": [date(2000, 1, 1)] * count,
        "date_mismatch": [False] * count,
        "source_datetime": [None] * count,
        "event_id": list(range(1, count + 1)),
        "visit_occurrence_id": [None] * count,
        "condition_status_concept_id": [None] * count,
        "value_as_number": values,
        "value_as_concept_id": [None] * count,
        "result_text": [None] * count,
        "unit_source_value": [None] * count,
        "range_low": lows,
        "range_high": highs,
        "numeric_value": values,
        "qual_result": qualities,
        "unit_group": [unit_group] * count,
        "range_usable": range_usable,
        "range_status": statuses,
        "range_low_converted": lows,
        "range_high_converted": highs,
    }, schema_overrides={
        "concept_id": pl.Int64, "numeric_value": pl.Float64, "value_as_number": pl.Float64,
        "value_as_concept_id": pl.Int64, "result_text": pl.String, "unit_source_value": pl.String,
        "range_low": pl.Float64, "range_high": pl.Float64,
        "range_low_converted": pl.Float64, "range_high_converted": pl.Float64,
        "source_datetime": pl.String, "range_usable": pl.Boolean, "range_status": pl.String,
    })


def boundary_table(rows):
    """Decile boundary table; rows are (domain, concept_id, cut points)."""
    records = []
    for domain, concept, cutpoints in rows:
        padded = list(cutpoints) + [None] * (9 - len(cutpoints))
        records.append({
            "domain": domain, "concept_id": concept, "unit_group": "mg/dL", "numeric_group_seen": True,
            **{f"d{i}": padded[i - 1] for i in range(1, 10)},
        })
    schema = {
        "domain": pl.String, "concept_id": pl.Int64, "unit_group": pl.String, "numeric_group_seen": pl.Boolean,
        **{f"d{i}": pl.Float64 for i in range(1, 10)},
    }
    return pl.DataFrame(records, schema=schema).lazy()


def test_range_priority_three_tokens_missing_and_unseen_numbers(tmp_path):
    ids = token_ids()
    config = {"paths": {"output_root": tmp_path}}
    (tmp_path / "units").mkdir()
    events = numeric_events(
        "measurement",
        [1, 2, 4, 5, 2, None, 2, 2, 2],
        [101, 101, 101, 101, 101, 101, 102, 101, 0],
        ["POS", None, None, None, "NEG", None, None, None, None],
        # Row 8 has a range, but its scale does not match the value, so it is not usable.
        [True, True, True, True, False, False, False, False, False],
        [2, 2, 2, 2, None, None, None, 1, None],
        [4, 4, 4, 4, None, None, None, 3, None],
    )
    events.write_parquet(tmp_path / "units" / "measurement.parquet")
    vocab = pl.DataFrame({
        "domain": ["measurement", "measurement"],
        "concept_id": [101, 102], "token_id": [100, 101],
    }, schema_overrides={"concept_id": pl.Int64, "token_id": pl.Int32}).lazy()
    boundaries = boundary_table([("measurement", 101, [2.0])])
    result = token_stage.build_domain_tokens(
        config, "measurement", vocab, boundaries, ids, "2000-01-02"
    ).collect().sort("event_id")

    assert result["event_tokens"].to_list() == [
        [100, ids["[LOW]"], ids["[POS]"]],
        [100], [100], [100, ids["[HIGH]"]],
        [100, ids["[Q2]"], ids["[NEG]"]], [100],
        [101, ids["[NUMERIC_UNSEEN]"]], [100, ids["[Q2]"]],
    ]
    assert "[NORMAL]" not in ids
    assert result["numeric_representation"].to_list() == [
        "range_low", "range_normal", "range_normal", "range_high",
        "decile", "no_number", "numeric_unseen", "decile",
    ]

    bucket = tmp_path / "tokens" / "events" / "bucket=1"
    bucket.mkdir(parents=True)
    result.write_parquet(bucket / "part.parquet")
    report = token_stage.summarize_tokens(config, ids)[0]
    assert report["events"] == 8
    assert report["three_token_events"] == 2
    assert report["concept_only_fraction"] == 3 / 8
    quantiles = pl.read_parquet(tmp_path / "tokens" / "quantile_counts.parquet")
    known = quantiles.filter(pl.col("concept_id") == 101)
    assert known["q2_events"].item() == 2
    assert known["q10_events"].item() == 0


def test_observation_without_ranges_keeps_number_and_neg(tmp_path):
    ids = token_ids()
    (tmp_path / "units").mkdir()
    events = numeric_events("observation", [5], [101], ["NEG"], [False], [None], [None])
    events.write_parquet(tmp_path / "units" / "observation.parquet")
    vocab = pl.DataFrame({"domain": ["observation"], "concept_id": [101], "token_id": [100]}).lazy()
    boundaries = boundary_table([("observation", 101, [float(i) for i in range(1, 10)])])
    result = token_stage.build_domain_tokens(
        {"paths": {"output_root": tmp_path}}, "observation", vocab, boundaries, ids, "2000-01-02"
    ).collect()
    assert result["event_tokens"].item().to_list() == [100, ids["[Q6]"], ids["[NEG]"]]


def test_validation_uses_range_before_unseen_numeric_fallback(tmp_path):
    """A concept without train numbers still uses a valid range first and UNSEEN last."""
    ids = token_ids()
    (tmp_path / "units").mkdir()
    events = numeric_events(
        "measurement", [1, 2, 3, 4, 5, 3], [102] * 6, [None] * 6,
        [True] * 5 + [False], [2, 2, 2, 2, 2, None], [4, 4, 4, 4, 4, None],
    ).with_columns(pl.lit("validation").alias("split"))
    events.write_parquet(tmp_path / "units" / "measurement.parquet")
    vocab = pl.DataFrame({
        "domain": ["measurement"], "concept_id": [102], "token_id": [100],
    }, schema_overrides={"concept_id": pl.Int64, "token_id": pl.Int32}).lazy()
    # No usable train numbers, so this concept has no decile row.
    boundaries = boundary_table([])
    result = token_stage.build_domain_tokens(
        {"paths": {"output_root": tmp_path}}, "measurement", vocab, boundaries, ids, "2000-01-02",
    ).collect().sort("event_id")
    assert result["split"].unique().to_list() == ["validation"]
    assert result["event_tokens"].to_list() == [
        [100, ids["[LOW]"]], [100], [100], [100], [100, ids["[HIGH]"]],
        [100, ids["[NUMERIC_UNSEEN]"]],
    ]
    assert result["numeric_representation"].to_list() == [
        "range_low", "range_normal", "range_normal", "range_normal", "range_high", "numeric_unseen",
    ]


def test_exact_duplicates_and_echo_headers_are_removed_before_tokens(tmp_path):
    ids = token_ids()
    (tmp_path / "events").mkdir()
    rows = numeric_events("condition", [None] * 3, [201, 201, 201], [None] * 3, [False] * 3, [None] * 3, [None] * 3)
    # Rows 1 and 2 are exact duplicates; row 3 has a different time.
    rows = rows.with_columns(pl.Series("source_datetime", ["2000-01-01T08:00", "2000-01-01T08:00", "2000-01-01T09:00"]))
    rows.write_parquet(tmp_path / "events" / "condition.parquet")
    vocab = pl.DataFrame({"domain": ["condition"], "concept_id": [201], "token_id": [100]},
                         schema_overrides={"concept_id": pl.Int64, "token_id": pl.Int32}).lazy()
    result = token_stage.build_domain_tokens(
        {"paths": {"output_root": tmp_path}}, "condition", vocab, boundary_table([]), ids, "2000-01-02",
    ).collect().sort("event_id")
    assert result["event_id"].to_list() == [1, 3]

    (tmp_path / "units").mkdir()
    header = numeric_events("observation", [None, None], [3046965, 4000], [None] * 2, [False] * 2, [None] * 2, [None] * 2)
    header.write_parquet(tmp_path / "units" / "observation.parquet")
    vocab = pl.DataFrame({"domain": ["observation"] * 2, "concept_id": [3046965, 4000], "token_id": [101, 102]},
                         schema_overrides={"concept_id": pl.Int64, "token_id": pl.Int32}).lazy()
    result = token_stage.build_domain_tokens(
        {"paths": {"output_root": tmp_path}}, "observation", vocab, boundary_table([]), ids, "2000-01-02",
    ).collect()
    # Echo report header Sex (3046965) is excluded.
    assert result["concept_id"].to_list() == [4000]


def test_demographic_tokens_follow_bos_and_use_first_kept_day():
    ids = token_ids()
    days = [(date(2000, 1, 1), [100], 1), (date(2010, 6, 1), [101], 1)]
    record = record_for_patient(1, "train", days, ids, max_tokens=7, person=(8532, 1950, 3))
    # Full history: age at 2000 (49); input: age at the only kept day in 2010 (60).
    assert record["full_tokens"][:3] == [ids["[BOS]"], ids["[SEX_F]"], ids["[AGE_45_49]"]]
    assert record["input_tokens"] == [
        ids["[BOS]"], ids["[SEX_F]"], ids["[AGE_60_64]"], ids["[DAY_GAP_0]"], 101, ids["[EOS]"],
    ]
    unknown = record_for_patient(2, "test", days[:1], ids, person=(None, None, None))
    assert unknown["input_tokens"][:3] == [ids["[BOS]"], ids["[SEX_UNKNOWN]"], ids["[AGE_UNKNOWN]"]]


def test_vocab_keeps_domain_identity_and_validation_test_concepts(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    (output / "events").mkdir(parents=True)
    concepts_folder = raw / "dbo" / "CONCEPT" / "batch-0"
    concepts_folder.mkdir(parents=True)
    (raw / "_COMPLETE.json").write_text(json.dumps({"finished_at": "2000-01-02T00:00:00"}))
    pl.DataFrame({
        "concept_id": [101, 102], "concept_name": ["합성 검사", ""], "vocabulary_id": ["TEST", "TEST"],
    }).write_parquet(concepts_folder / "part.parquet")
    for domain in token_stage.DOMAINS:
        numeric_events(
            domain, [None, None, None], [101, 102, 103], [None] * 3,
            [False] * 3, [None] * 3, [None] * 3,
        ).with_columns(pl.Series("split", ["train", "validation", "test"])).write_parquet(
            output / "events" / f"{domain}.parquet"
        )
    config = {"paths": {"raw_root": raw, "output_root": output}}
    monkeypatch.setattr(token_stage, "load_config", lambda _: config)
    monkeypatch.setattr(token_stage, "check_event_exceptions", lambda _: None)
    token_stage.build_vocab("synthetic-config")
    vocabulary = pl.read_parquet(output / "tokens" / "vocab.parquet")
    assert vocabulary["token_id"].to_list() == list(range(vocabulary.height))
    assert vocabulary.filter(pl.col("concept_id") == 103).height == 5
    assert vocabulary.filter(pl.col("concept_id") == 101)["token_id"].n_unique() == 5
    names = pl.read_parquet(output / "tokens" / "concept_names.parquet")
    assert names.filter(pl.col("concept_id") == 103)["concept_name"].null_count() == 5


def test_tokenization_writes_all_domains_to_patient_bucket(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "_COMPLETE.json").write_text(json.dumps({"finished_at": "2000-01-02T00:00:00"}))
    config = {"paths": {"raw_root": raw, "output_root": tmp_path}}
    (tmp_path / "tokens").mkdir()
    (tmp_path / "deciles").mkdir()
    records = [{"token_id": value, "token": token, "domain": None, "concept_id": None}
               for token, value in token_ids().items()]
    for order, domain in enumerate(token_stage.DOMAINS):
        records.append({"token_id": 100 + order, "token": f"{domain}:101", "domain": domain, "concept_id": 101})
        folder = tmp_path / ("units" if domain in ["measurement", "observation"] else "events")
        folder.mkdir(exist_ok=True)
        numeric_events(domain, [None], [101], [None], [False], [None], [None]).write_parquet(
            folder / f"{domain}.parquet"
        )
    pl.DataFrame(records, schema_overrides={"token_id": pl.Int32, "concept_id": pl.Int64}).write_parquet(
        tmp_path / "tokens" / "vocab.parquet"
    )
    boundary_table([]).collect().write_parquet(tmp_path / "deciles" / "boundaries.parquet")
    monkeypatch.setattr(token_stage, "load_config", lambda _: config)
    for checker in ["check_event_exceptions", "check_parsing_exceptions", "check_unit_exceptions"]:
        monkeypatch.setattr(token_stage, checker, lambda _: None)
    token_stage.tokenize_events("synthetic-config")
    bucket = pl.read_parquet(str(tmp_path / "tokens" / "events" / "bucket=1" / "*.parquet"))
    assert set(bucket["domain"].to_list()) == set(token_stage.DOMAINS)
    assert sorted(bucket["event_tokens"].to_list()) == [[100], [101], [102], [103], [104]]
    report = json.loads((tmp_path / "tokens" / "event_summary.json").read_text())
    assert report["total_events"] == 5
    assert sum(row["concept_only_events"] for row in report["by_split_domain"]) == 5


def test_context_is_fixed_at_16384():
    assert context_limit({}) == 16_384
    with pytest.raises(ValueError, match="16,384"):
        context_limit({"model": {"max_context": 32_768}})


def test_bucket_keeps_same_day_domain_pk_order_and_repeated_events(tmp_path):
    ids = token_ids()
    config = {"paths": {"output_root": tmp_path}}
    tokens_folder = tmp_path / "tokens"
    bucket = tokens_folder / "events" / "bucket=1"
    bucket.mkdir(parents=True)
    (tokens_folder / "event_summary.json").write_text("{}")
    pl.DataFrame({
        "person_id": [1, 1, 1], "split": ["train"] * 3,
        "event_date": [date(2000, 1, 1)] * 3,
        "domain_order": [1, 0, 0], "event_id": [1, 2, 1],
        "event_tokens": [[12], [10], [10]],
    }).write_parquet(bucket / "part.parquet")
    destination = process_sequence_bucket(config, 1, ids)
    row = pl.read_parquet(destination).row(0, named=True)
    assert row["input_tokens"] == [ids["[BOS]"], ids["[DAY_GAP_0]"], 10, 10, 12, ids["[EOS]"]]
    assert row["full_events"] == 3
    empty = process_sequence_bucket(config, 2, ids)
    assert pl.read_parquet(empty).is_empty()


def test_summary_reports_empty_patients_and_stops_for_oversized_latest_day(tmp_path):
    ids = token_ids()
    config = {"paths": {"output_root": tmp_path}}
    normal = record_for_patient(1, "train", [(date(2000, 1, 1), [100], 1)], ids)
    too_long = record_for_patient(2, "test", [(date(2000, 1, 1), [100] * 16_382, 1)], ids)
    assert too_long["input_length"] == 0
    (tmp_path / "sequences").mkdir()
    pq.write_table(pa.Table.from_pylist([normal, too_long]), tmp_path / "sequences" / "bucket=001.parquet")
    (tmp_path / "sample").mkdir()
    pl.DataFrame({
        "person_id": [1, 2, 3, 4], "split": ["train", "test", "train", "validation"],
    }).write_parquet(tmp_path / "sample" / "selected_patients.parquet")
    with pytest.raises(ValueError, match="handling rule"):
        summarize_sequences(config)
    summary = json.loads((tmp_path / "sequences" / "summary.json").read_text())
    assert summary["initial_patients"] == 4
    assert summary["excluded_empty_patients"] == 2
    assert summary["patients_without_input"] == 1
    assert summary["ready_for_training"] is False
    assert summary["excluded_events_fraction"] == 0.5
    assert pl.read_parquet(tmp_path / "sequences" / "lengths.parquet").height == 2


def test_joint_keys_distinguish_numeric_and_qualitative_states():
    events = pl.DataFrame({
        "event_tokens": [[10, 3], [10, 4], [10, 3, 5]],
        "numeric_token_id": [3, 4, 3],
        "qual_token_id": [None, None, 5],
    }).lazy()
    keys = value_keys(events).select("base_id", "number_id", "quality_id").collect()
    assert keys.to_dicts() == [
        {"base_id": 10, "number_id": 3, "quality_id": -1},
        {"base_id": 10, "number_id": 4, "quality_id": -1},
        {"base_id": 10, "number_id": 3, "quality_id": 5},
    ]


def test_joint_vocab_and_events_collapse_existing_value_pair(tmp_path):
    """Only events with a value become one joint token; source-code events stay as they are."""
    from ausom_ehr.tokens import build_joint_vocab, convert_joint_events

    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "tokens").mkdir(parents=True)
    vocab = pl.DataFrame({
        "token_id": [0, 1, 2, 3],
        "token": ["[PAD]", "[Q1]", "MEASUREMENT:123", "PROCEDURE_SOURCE:U|ABC"],
        "domain": [None, None, "measurement", "procedure"],
        "concept_id": [None, None, 123, None],
        "source_code": [None, None, None, "U|ABC"],
    }, schema_overrides={"token_id": pl.Int32, "concept_id": pl.Int64})
    vocab.write_parquet(source / "tokens" / "vocab.parquet")
    original_events = pl.DataFrame({
        "event_tokens": [[2, 1], [3]],
        "numeric_token_id": [1, None],
        "qual_token_id": [None, None],
        "event_token_count": [2, 1],
    }, schema_overrides={
        "event_tokens": pl.List(pl.Int32), "numeric_token_id": pl.Int32,
        "qual_token_id": pl.Int32, "event_token_count": pl.Int8,
    })
    for bucket in range(128):
        folder = source / "tokens" / "events" / f"bucket={bucket}"
        folder.mkdir(parents=True)
        original_events.write_parquet(folder / "part.parquet")
    config = {"paths": {"source_variant_root": str(source), "output_root": target}}
    build_joint_vocab(config)
    joint_vocab = pl.read_parquet(target / "tokens" / "vocab.parquet")
    assert joint_vocab.height == 5
    assert joint_vocab.filter(pl.col("base_id") == 2)["token"].item() == "MEASUREMENT:123+[Q1]"
    assert special_ids(target / "tokens" / "vocab.parquet") == {"[PAD]": 0, "[Q1]": 1}
    convert_joint_events(config)
    converted = pl.read_parquet(target / "tokens" / "events" / "bucket=0" / "part-0.parquet")
    assert converted["event_tokens"].to_list() == [[4], [3]]
    assert converted["event_token_count"].to_list() == [1, 1]


def test_32768_context_keeps_more_whole_days_and_one_patient_batch():
    from datetime import date

    import pytest
    from ausom_ehr.tokens import record_for_patient
    from ausom_ehr.train import make_microbatches

    ids = {"[BOS]": 1, "[EOS]": 2, "[DAY_GAP_0]": 3, "[DAY_GAP_1]": 4}
    days = [(date(2020, 1, 1), [10] * 15_000, 1),
            (date(2020, 1, 2), [11] * 15_000, 1)]
    old = record_for_patient(1, "train", days, ids)
    longer = record_for_patient(1, "train", days, ids, 32_768)
    assert old["input_days"] == 1
    assert longer["input_days"] == 2
    assert longer["input_length"] == 30_004
    row = {"person_id": 1, "input_length": 32_768}
    with pytest.raises(ValueError):
        make_microbatches([row], shuffle=False)
    assert make_microbatches([row], shuffle=False, padded_limit=32_768) == [[row]]
