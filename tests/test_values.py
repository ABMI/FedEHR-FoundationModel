"""Synthetic checks for result parsing, unit conversion and train-only deciles."""

import json
from datetime import date

import polars as pl
import pytest

from ausom_ehr.values import (
    SPELLING_TO_UNIT,
    add_parsed_results,
    apply_unit_conversions,
    build_conversion_rows,
    build_qualitative_dictionary,
    calculate_decile_boundaries,
    check_parsing_exceptions,
    check_unit_exceptions,
    clean_vital_values,
    count_quantile_frequencies,
    load_train_numbers,
    normalize_unit_text,
    parse_result_domain,
    write_converted_domain_summary,
)


def test_numeric_and_qualitative_results_survive_together_and_exceptions_are_visible():
    source = pl.DataFrame({
        "result_text": ["POSITIVE", "10", ">1,000", "2,5", "<NaN", "1e309"],
        "value_as_number": [7.0, 12.0, None, None, None, None],
        "value_as_concept_id": [None] * 6,
    })
    parsed = add_parsed_results(source.lazy()).collect()

    assert parsed["result_text"].to_list() == source["result_text"].to_list()
    assert parsed["numeric_value_raw"][:5].to_list() == [7.0, 12.0, 1000.0, None, None]
    assert parsed["qual_result"][0] == "POS"
    assert parsed["structured_raw_number_conflict"][1]
    assert parsed["malformed_numeric_text"][3]
    assert parsed["nonfinite_text"][4]
    assert parsed["nonfinite_number"][5]


def test_korean_comparators_semiquantitative_grades_and_free_text():
    source = pl.DataFrame({
        "result_text": ["7 이하", "50 이상", "1+", "4+", "Trace", "1. Normal sized cardiac chambers", ".", "1:40"],
        "value_as_number": [None] * 8,
        "value_as_concept_id": [None] * 8,
    })
    parsed = add_parsed_results(source.lazy()).collect()
    assert parsed["numeric_value_raw"].to_list() == [7.0, 50.0, None, None, None, None, None, None]
    assert parsed["qual_result"].to_list() == [None, None, "POS", "POS", None, None, None, None]
    # Free text and '.' are concept only, not malformed; '1:40' is malformed.
    assert parsed["malformed_numeric_text"].to_list() == [False] * 7 + [True]


def test_titers_ratios_and_clock_times_keep_concept_only_without_stopping(tmp_path):
    config = {"paths": {"output_root": tmp_path}}
    (tmp_path / "events").mkdir()
    events = pl.DataFrame({
        "event_id": [1, 2, 3, 4, 5],
        "person_id": [1, 1, 1, 1, 1],
        "concept_id": [3029945, 3013338, 3011625, 0, 3000963],
        "result_text": ["1:16", "1.7:1", "2:30", "2..34", "12.5"],
        "value_as_number": [None] * 5,
        "value_as_concept_id": [None] * 5,
    }, schema_overrides={"value_as_number": pl.Float64, "value_as_concept_id": pl.Int64})
    events.write_parquet(tmp_path / "events" / "measurement.parquet")
    events.head(0).write_parquet(tmp_path / "events" / "observation.parquet")
    for domain in ["measurement", "observation"]:
        parse_result_domain(config, domain, resume=False)
    check_parsing_exceptions(config)
    parsed = pl.read_parquet(tmp_path / "results" / "measurement.parquet").sort("event_id")
    assert parsed["numeric_value_raw"].to_list() == [None, None, None, None, 12.5]
    summary = json.loads((tmp_path / "results" / "measurement_summary.json").read_text())
    assert summary["malformed_numeric_text_rows"] == 4
    audit = pl.read_parquet(tmp_path / "results" / "measurement_concept_only_values.parquet")
    assert sorted(audit["result_text"].to_list()) == ["1.7:1", "1:16", "2..34", "2:30"]
    assert pl.read_parquet(tmp_path / "results" / "measurement_exceptions.parquet").height == 0


def test_unicode_comparators_are_read_and_their_direction_is_kept():
    source = pl.DataFrame({
        "result_text": ["≥60.00", "≤5.0", "＜5.0", "7 이하", "50 이상", "<1.0", "12", "≥P1:1280"],
        "value_as_number": [None, None, None, None, None, 1.0, None, None],
        "value_as_concept_id": [None] * 8,
        "operator_concept_id": [None, None, None, None, None, 4171756, None, None],
    }, schema_overrides={"value_as_number": pl.Float64, "value_as_concept_id": pl.Int64, "operator_concept_id": pl.Int64})
    parsed = add_parsed_results(source.lazy()).collect()

    assert parsed["numeric_value_raw"].to_list() == [60.0, 5.0, 5.0, 7.0, 50.0, 1.0, 12.0, None]
    assert parsed["numeric_censoring"].to_list() == [
        "greater", "less", "less", "less", "greater", "less", None, None,
    ]
    assert not parsed["censoring_conflict"].any()
    # A titer after a comparator is concept only, not a malformed number.
    assert not parsed["malformed_operator_number"][7]


def write_range_case(tmp_path, rows):
    """Rows are (concept, unit text, value, range low, range high, censoring)."""
    (tmp_path / "results").mkdir()
    (tmp_path / "units").mkdir()
    pl.DataFrame({
        "domain": ["measurement"] * len(rows),
        "concept_id": [row[0] for row in rows],
        "numeric_value_raw": [row[2] for row in rows],
        "unit_concept_id": [None] * len(rows),
        "unit_source_value": [row[1] for row in rows],
        "range_low": [row[3] for row in rows],
        "range_high": [row[4] for row in rows],
        "numeric_censoring": [row[5] for row in rows],
    }, schema_overrides={
        "unit_concept_id": pl.Int64, "unit_source_value": pl.String, "numeric_censoring": pl.String,
    }).write_parquet(tmp_path / "results" / "measurement.parquet")
    signatures = sorted({(row[0], row[1]) for row in rows})
    pl.DataFrame({
        "domain": ["measurement"] * len(signatures),
        "concept_id": [concept for concept, _ in signatures],
        "unit_concept_id": [0] * len(signatures),
        "unit_source_value": [unit for _, unit in signatures],
        "canonical_unit": [unit for _, unit in signatures],
        "target_unit": [unit for _, unit in signatures],
        "unit_group": [unit for _, unit in signatures],
        "conversion_factor": [1.0] * len(signatures),
        "unit_status": ["same_unit"] * len(signatures),
    }).write_parquet(tmp_path / "units" / "conversions.parquet")
    return {"paths": {"output_root": tmp_path}}


def test_less_than_results_are_never_high_and_greater_than_results_are_never_low(tmp_path):
    config = write_range_case(tmp_path, [
        (301, "ng/mL", 0.05, 0.0, 0.02, "less"),     # troponin '<0.05': below detection limit
        (301, "ng/mL", 0.30, 0.0, 0.02, None),
        (301, "ng/mL", 0.01, 0.0, 0.02, None),
        (302, "mL/min", 60.0, 90.0, 120.0, "greater"),  # '>=60' is never LOW
        (302, "mL/min", 40.0, 90.0, 120.0, None),
        (302, "mL/min", 100.0, 90.0, 120.0, None),
    ])
    apply_unit_conversions(config, "measurement")
    converted = pl.read_parquet(tmp_path / "units" / "measurement.parquet").sort("concept_id", "numeric_value_raw")
    assert converted["range_status"].to_list() == ["NORMAL", "NORMAL", "HIGH", "LOW", "NORMAL", "NORMAL"]
    summary = json.loads((tmp_path / "units" / "measurement_summary.json").read_text())
    assert summary["censored_kept_normal_rows"] == 2


def test_range_is_rejected_per_unit_spelling_when_results_sit_on_one_side(tmp_path):
    # Only the 'ng/ml' spelling has its range on another scale (values 600 vs upper 0.5).
    rows = [(401, "ng/ml", 600.0 + i, 0.0, 0.5, None) for i in range(20)]
    rows += [(401, "ug/mL", value, 0.0, 0.5, None) for value in [0.2, 0.3, 0.4, 0.8, 1.2] * 4]
    # Under 20 rows, one-sidedness alone does not reject the range.
    rows += [(402, "U/L", value, 0.0, 1.5, None) for value in [2.0, 3.0, 4.0]]
    # Negative lower bounds (base excess) skip the scale check.
    rows += [(403, "mmol/L", value, -2.5, 3.0, None) for value in [-4.0, -1.0, -0.6, 1.0, 4.0] * 4]
    config = write_range_case(tmp_path, rows)
    apply_unit_conversions(config, "measurement")
    converted = pl.read_parquet(tmp_path / "units" / "measurement.parquet")
    usable = dict(converted.group_by("unit_source_value").agg(pl.col("range_usable").all()).iter_rows())
    assert usable == {"ng/ml": False, "ug/mL": True, "U/L": True, "mmol/L": True}
    checks = pl.read_parquet(tmp_path / "units" / "measurement_range_checks.parquet")
    one_sided = dict(zip(checks["unit_source_value"], checks["range_one_sided"]))
    assert one_sided == {"ng/ml": True, "ug/mL": False, "U/L": False, "mmol/L": False}


def test_unit_text_that_contradicts_values_is_not_converted():
    signatures = [
        # Ionized magnesium: 'mg/dl' values are on the mmol/L scale.
        {"domain": "measurement", "concept_id": 3021197, "unit_concept_id": 0, "unit_source_value": "mmol/L",
         "numeric_rows": 14770, "median_value": 0.51},
        {"domain": "measurement", "concept_id": 3021197, "unit_concept_id": 0, "unit_source_value": "mg/dl",
         "numeric_rows": 3808, "median_value": 0.54},
        # IgG subclass: mg/dL x 10 matches mg/L, so it is converted.
        {"domain": "measurement", "concept_id": 3027135, "unit_concept_id": 0, "unit_source_value": "mg/L",
         "numeric_rows": 488, "median_value": 7055.0},
        {"domain": "measurement", "concept_id": 3027135, "unit_concept_id": 0, "unit_source_value": "mg/dL",
         "numeric_rows": 439, "median_value": 603.8},
    ]
    rows = {(row["concept_id"], row["unit_source_value"]): row for row in build_conversion_rows(signatures, {})}
    assert rows[(3021197, "mg/dl")]["unit_status"] == "unit_text_mismatch"
    assert rows[(3021197, "mg/dl")]["conversion_factor"] == 1.0
    assert rows[(3021197, "mg/dl")]["unit_group"] == "mmol/L"
    assert rows[(3027135, "mg/dL")]["unit_status"] == "converted"
    assert rows[(3027135, "mg/dL")]["conversion_factor"] == pytest.approx(10.0)


def test_unit_spellings_collapse_to_one_unit():
    expected = {
        "X10³/㎕": "10^3/uL", "×10³/㎕": "10^3/uL", "10*3/ul": "10^3/uL",
        "mg/dl": "mg/dL", "mg/dL": "mg/dL", "％": "%", "℃": "Cel", "Cel": "Cel",
        "mMol/L": "mmol/L", "uIu/ml": "uIU/mL", "ulU/ml": "uIU/mL", "mcg/L": "ug/L",
        "ml/min/1.73m²": "mL/min/1.73m2",
    }
    assert {text: SPELLING_TO_UNIT[normalize_unit_text(text)] for text in expected} == expected


def test_conversion_rows_unify_spelling_convert_scale_and_separate_assay_units():
    signatures = [
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 0, "unit_source_value": "mg/dL", "numeric_rows": 50},
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 0, "unit_source_value": "mg/dl", "numeric_rows": 40},
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 0, "unit_source_value": "mg/L", "numeric_rows": 5},
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 0, "unit_source_value": "CU", "numeric_rows": 3},
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 0, "unit_source_value": "", "numeric_rows": 2},
        {"domain": "measurement", "concept_id": 101, "unit_concept_id": 8840, "unit_source_value": "", "numeric_rows": 1},
        {"domain": "measurement", "concept_id": 3021197, "unit_concept_id": 0, "unit_source_value": "mmol/L", "numeric_rows": 9},
        {"domain": "measurement", "concept_id": 3021197, "unit_concept_id": 0, "unit_source_value": "mg/dl", "numeric_rows": 2},
    ]
    rows = build_conversion_rows(signatures, {8840: "mg/dL"})
    by_text = {(row["concept_id"], row["unit_concept_id"], row["unit_source_value"]): row for row in rows}

    assert by_text[(101, 0, "mg/dl")]["unit_status"] == "same_unit"
    assert by_text[(101, 0, "mg/L")]["unit_status"] == "converted"
    assert by_text[(101, 0, "mg/L")]["conversion_factor"] == pytest.approx(0.1)
    assert by_text[(101, 0, "CU")]["unit_status"] == "separate_unit_group"
    assert by_text[(101, 0, "CU")]["unit_group"] == "CU"
    assert by_text[(101, 0, "")]["unit_status"] == "unit_missing"
    assert by_text[(101, 0, "")]["unit_group"] == "mg/dL"
    assert by_text[(101, 8840, "")]["unit_status"] == "same_unit"
    assert by_text[(3021197, 0, "mg/dl")]["conversion_factor"] == pytest.approx(10 / 24.305)


def test_unit_conversion_changes_value_and_range_together_and_checks_range_scale(tmp_path):
    config = {"paths": {"output_root": tmp_path}}
    (tmp_path / "results").mkdir()
    (tmp_path / "units").mkdir()
    source = pl.DataFrame({
        "domain": ["measurement"] * 3,
        "concept_id": [101, 101, 102],
        "numeric_value_raw": [2.0, 3.0, 500.0],
        "unit_concept_id": [None, None, None],
        "unit_source_value": [None, "mg/L", "mg/dL"],
        "range_low": [1.0, 10.0, 1.0],
        "range_high": [4.0, 40.0, 4.0],
    }, schema_overrides={"unit_concept_id": pl.Int64, "unit_source_value": pl.String})
    source.write_parquet(tmp_path / "results" / "measurement.parquet")
    pl.DataFrame({
        "domain": ["measurement"] * 3,
        "concept_id": [101, 101, 102],
        "unit_concept_id": [0, 0, 0],
        "unit_source_value": ["", "mg/L", "mg/dL"],
        "canonical_unit": [None, "mg/L", "mg/dL"],
        "target_unit": ["mg/dL", "mg/dL", "mg/dL"],
        "unit_group": ["mg/dL", "mg/dL", "mg/dL"],
        "conversion_factor": [1.0, 0.1, 1.0],
        "unit_status": ["unit_missing", "converted", "same_unit"],
    }).write_parquet(tmp_path / "units" / "conversions.parquet")

    apply_unit_conversions(config, "measurement")
    converted = pl.read_parquet(tmp_path / "units" / "measurement.parquet").sort("concept_id", "numeric_value_raw")
    assert converted["numeric_value_raw"].to_list() == [2.0, 3.0, 500.0]
    assert converted["numeric_value"].to_list() == pytest.approx([2.0, 0.3, 500.0])
    assert converted["range_low_converted"].to_list() == pytest.approx([1.0, 1.0, 1.0])
    assert converted["range_high_converted"].to_list() == pytest.approx([4.0, 4.0, 4.0])
    # A group whose values all exceed 10x the upper bound gets no range.
    assert converted["range_usable"].to_list() == [True, True, False]


def test_vital_placeholders_and_mimic_extract_ranges():
    data = pl.DataFrame({
        "concept_id": [3004249, 3004249, 3004249, 3004249, 3020891, 3020891, 3020891, 3016723],
        "numeric_value": [-1.0, 1.0, 120.0, 400.0, 0.0, 20.0, 37.0, -1.0],
    }).lazy()
    cleaned = clean_vital_values(data).collect()
    # BP -1/1 are placeholders, 400 an outlier; temperature 0 is an outlier, 20 clips to 26; labs untouched.
    assert cleaned["numeric_value"].to_list() == [None, None, 120.0, None, None, 26.0, 37.0, -1.0]
    assert cleaned["vital_cleaning"].to_list() == [
        "placeholder_removed", "placeholder_removed", None, "outlier_removed",
        "outlier_removed", "clipped_low", None, None,
    ]


def test_deciles_use_train_only_and_give_each_value_its_own_bin_when_values_are_few(tmp_path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    raw_root.mkdir()
    (output_root / "units").mkdir(parents=True)
    (raw_root / "_COMPLETE.json").write_text(json.dumps({"finished_at": "2000-01-10T00:00:00"}))
    values = [0.0] * 20 + [1.0, 2.0, 3.0]
    rows = len(values)
    source = pl.DataFrame({
        "domain": ["measurement"] * (rows + 2),
        "concept_id": [101] * (rows + 2),
        "unit_group": ["class"] * (rows + 2),
        "person_id": list(range(rows)) + [900, 901],
        "split": ["train"] * rows + ["validation", "test"],
        "event_date": [date(2000, 1, 1)] * (rows + 2),
        "datetime_date": [date(2000, 1, 1)] * (rows + 2),
        "date_mismatch": [False] * (rows + 2),
        "numeric_value": values + [99.0, -99.0],
        "range_usable": [False] * (rows + 2),
    })
    source.write_parquet(output_root / "units" / "measurement.parquet")
    source.head(0).with_columns(pl.lit("observation").alias("domain")).write_parquet(
        output_root / "units" / "observation.parquet"
    )
    config = {"paths": {"raw_root": raw_root, "output_root": output_root}}

    train = load_train_numbers(config)
    boundaries = calculate_decile_boundaries(train)
    assert boundaries["numeric_rows"].to_list() == [rows]
    assert boundaries["cut_rule"].to_list() == ["each_value"]
    assert [boundaries[f"d{i}"].item() for i in range(1, 10)] == [1.0, 2.0, 3.0] + [None] * 6

    frequencies = count_quantile_frequencies(train, boundaries.lazy()).collect()
    # 0 is [Q1]; 1, 2, 3 are [Q2], [Q3], [Q4].
    assert frequencies["numeric_rows"].to_list() == [20, 1, 1, 1] + [0] * 6


def test_many_value_deciles_merge_repeated_boundaries_and_keep_minimum_in_q1(tmp_path):
    values = pl.DataFrame({
        "domain": ["measurement"] * 100,
        "concept_id": [101] * 100,
        "unit_group": ["mg/dL"] * 100,
        "person_id": list(range(100)),
        "numeric_value": [0.0] * 85 + [float(value) for value in range(1, 16)],
        "range_usable": [False] * 100,
    }).lazy()
    boundaries = calculate_decile_boundaries(values)
    assert boundaries["cut_rule"].item() == "deduplicated_quantiles"
    # 85% zeros: 8 of 9 candidates are 0; cutpoints equal to the minimum are dropped, repeats merged.
    assert [boundaries[f"d{i}"].item() for i in range(1, 10)] == [6.0] + [None] * 8


def test_qualitative_dictionary_uses_observed_concept_names_instead_of_assuming_ids(tmp_path):
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "output"
    concept_folder = raw_root / "dbo" / "CONCEPT" / "batch-test"
    concept_folder.mkdir(parents=True)
    (output_root / "events").mkdir(parents=True)
    pl.DataFrame({
        "concept_id": [123, 9189],
        "concept_name": ["Negative", "An unrelated result"],
        "concept_code": ["A", "B"],
        "vocabulary_id": ["Synthetic", "Synthetic"],
    }).write_parquet(concept_folder / "part.parquet")
    events = pl.DataFrame({
        "domain": ["measurement"] * 2,
        "value_as_concept_id": [123, 9189],
        "result_text": [None, None],
    }, schema_overrides={"result_text": pl.String})
    events.write_parquet(output_root / "events" / "measurement.parquet")
    events.head(0).with_columns(pl.lit("observation").alias("domain")).write_parquet(
        output_root / "events" / "observation.parquet"
    )
    config = {"paths": {"raw_root": raw_root, "output_root": output_root}}
    assert build_qualitative_dictionary(config) == {123: "NEG"}


def test_unit_guard_blocks_nonfinite_values_but_keeps_missing_and_separate_units(tmp_path):
    config = {"paths": {"output_root": tmp_path}}
    (tmp_path / "units").mkdir()
    preserved = pl.DataFrame({
        "numeric_value_raw": [1.0, 2.0, 3.0],
        "numeric_value": [1.0, 2.0, 3.0],
        "range_low_converted": [None] * 3,
        "range_high_converted": [None] * 3,
        "unit_status": ["unit_missing", "unknown_unit_text", "separate_unit_group"],
        "vital_cleaning": [None] * 3,
        "concept_id": [1, 1, 1],
        "range_usable": [False] * 3,
        "range_status": [None] * 3,
        "numeric_censoring": [None] * 3,
    }, schema_overrides={
        "range_low_converted": pl.Float64, "range_high_converted": pl.Float64, "vital_cleaning": pl.String,
        "range_status": pl.String, "numeric_censoring": pl.String,
    })
    measurement_path = tmp_path / "units" / "measurement.parquet"
    observation_path = tmp_path / "units" / "observation.parquet"
    preserved.write_parquet(measurement_path)
    preserved.head(0).write_parquet(observation_path)
    for domain, path in [("measurement", measurement_path), ("observation", observation_path)]:
        write_converted_domain_summary(config, domain, path)
    check_unit_exceptions(config)

    broken = preserved.with_columns(pl.Series("numeric_value", [1.0, float("inf"), 3.0]))
    broken.write_parquet(measurement_path)
    write_converted_domain_summary(config, "measurement", measurement_path)
    with pytest.raises(ValueError, match="nonfinite_numeric_value_rows"):
        check_unit_exceptions(config)


def test_fullwidth_digits_are_read_as_numbers():
    import polars as pl
    from ausom_ehr import values as value_stage
    text = pl.DataFrame({"t": ["５", "１２．５", "≥６０"]})
    expression = pl.col("t")
    for spelling, plain in (value_stage.COMPARATOR_SPELLINGS | value_stage.FULLWIDTH_DIGITS).items():
        expression = expression.str.replace_all(spelling, plain, literal=True)
    assert text.select(expression)["t"].to_list() == ["5", "12.5", ">=60"]
