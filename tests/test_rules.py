"""Synthetic checks for day boundaries and outcome-window leakage."""

from datetime import date, datetime

import polars as pl

from ausom_ehr.tasks import make_prefix, merge_admissions, outcome_label
from ausom_ehr.tokens import choose_recent_days, flatten_days, gap_token
from ausom_ehr.values import add_parsed_results


def test_day_gap_boundaries_and_whole_day_truncation():
    ids = {"[BOS]": 1, "[EOS]": 2, "[DAY_GAP_0]": 3, "[DAY_GAP_1]": 4}
    ids.update({
        "[DAY_GAP_30]": 5,
        "[DAY_GAP_31_60]": 6,
        "[DAY_GAP_61_90]": 7,
        "[DAY_GAP_91_180]": 8,
        "[DAY_GAP_181_365]": 9,
        "[DAY_GAP_LT]": 10,
    })
    assert [gap_token(n, ids) for n in [0, 1, 30, 31, 60, 61, 90, 91, 180, 181, 365, 366]] == [
        3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10,
    ]
    days = [
        (date(2020, 1, 1), [11, 12, 13], 1),
        (date(2020, 1, 2), [14, 15, 16], 1),
    ]
    chosen = choose_recent_days(days, limit=6, eos=True)
    assert chosen == days[1:]
    # After old days are dropped, the first gap becomes DAY_GAP_0.
    assert flatten_days(chosen, ids, eos=True) == [1, 3, 14, 15, 16, 2]


def test_exact_touch_merges_but_same_day_gap_does_not():
    visits = [
        {
            "type": 9201, "start": datetime(2020, 1, 1, 8),
            "end": datetime(2020, 1, 3, 12), "visit_id": 1,
        },
        {
            "type": 262, "start": datetime(2020, 1, 3, 12),
            "end": datetime(2020, 1, 5, 9), "visit_id": 2,
        },
        {
            "type": 9201, "start": datetime(2020, 1, 5, 10),
            "end": datetime(2020, 1, 8, 9), "visit_id": 3,
        },
    ]
    episodes = merge_admissions(visits)
    assert len(episodes) == 2
    assert episodes[0]["member_ids"] == [1, 2]
    assert episodes[1]["member_ids"] == [3]


def test_death_before_readmission_censors_but_later_death_does_not():
    index = date(2020, 1, 1)
    long_record = date(2021, 1, 1)
    assert outcome_label(index, date(2020, 1, 10), date(2020, 1, 9), long_record, 30)[0] is None
    assert outcome_label(index, date(2020, 1, 10), date(2020, 1, 11), long_record, 30)[0] == 1
    assert outcome_label(index, date(2020, 1, 10), date(2020, 1, 10), long_record, 30)[0] is None
    assert outcome_label(index, None, None, long_record, 30)[0] == 0


def test_negative_needs_record_through_the_window_like_ehrshot():
    index = date(2020, 1, 1)
    # No admission and the record ends before the window end (Jan 31): no label.
    assert outcome_label(index, None, None, date(2020, 1, 30), 30) == (None, "record_ends_before_window")
    assert outcome_label(index, None, None, date(2020, 1, 31), 30) == (0, "no_admission_in_window")
    # A positive stays positive regardless of record length.
    assert outcome_label(index, date(2020, 1, 5), None, date(2020, 1, 5), 30)[0] == 1


def test_cutoff_keeps_same_date_events_but_excludes_ongoing_final_diagnosis():
    ids = {"[BOS]": 1, "[DAY_GAP_0]": 2}
    events = [
        (date(2020, 1, 1), [11], 1, "condition", 4230359),
        (date(2020, 1, 1), [12], 1, "measurement", None),
        (date(2020, 1, 1), [13], 2, "drug", None),
        (date(2020, 1, 2), [14], 1, "procedure", None),
    ]
    visit_dates = {
        1: (date(2020, 1, 1), date(2020, 1, 4)),
        2: (date(2020, 1, 2), date(2020, 1, 2)),
    }
    tokens, report = make_prefix(events, date(2020, 1, 1), visit_dates, ids)
    # Events up to the prediction time (Jan 1 23:59) are used even if their visit starts later.
    assert tokens == [1, 2, 12, 13]
    assert report["excluded_final_diagnosis_events"] == 1
    assert report["excluded_future_date_events"] == 1


def test_exact_thousands_comparator_only():
    data = pl.DataFrame({
        "result_text": [">1,000", "<12,345", "<1:2", "NEGATIVE", "2,5"],
        "value_as_number": [None] * 5,
        "value_as_concept_id": [None] * 5,
    }).lazy()
    parsed = add_parsed_results(data).collect()
    assert parsed["numeric_value_raw"].to_list() == [1000.0, 12345.0, None, None, None]
    assert parsed["malformed_operator_number"].to_list() == [False, False, True, False, False]
    assert parsed["qual_result"].to_list() == [None, None, None, "NEG", None]


def test_invalid_visit_interval_final_diagnosis_is_excluded():
    ids = {"[BOS]": 1, "[DAY_GAP_0]": 2}
    events = [
        (date(2020, 1, 3), [11], 1, "condition", 4230359),
        (date(2020, 1, 3), [12], 1, "measurement", None),
    ]
    visits = {1: (date(2020, 1, 3), date(2020, 1, 2))}
    tokens, report = make_prefix(events, date(2020, 1, 3), visits, ids)
    assert tokens == [1, 2, 12]
    assert report["excluded_final_diagnosis_events"] == 1


def test_every_admission_and_outpatient_visit_is_a_sample_like_ehrshot():
    from ausom_ehr.tasks import candidates_for_patient

    def visit(visit_id, visit_type, start, end):
        return {"person_id": 1, "split": "train", "visit_id": visit_id, "type": visit_type,
                "start": start, "end": end}

    visits = [
        visit(1, 9202, datetime(2019, 12, 20, 9), datetime(2019, 12, 20, 10)),   # outpatient, admitted within 90 days
        visit(2, 9201, datetime(2020, 1, 1, 8), datetime(2020, 1, 10, 10)),      # 9-day admission
        visit(3, 32037, datetime(2020, 1, 20, 8), datetime(2020, 1, 22, 8)),     # ICU-only stay
        visit(4, 9202, datetime(2020, 3, 1, 7), datetime(2020, 3, 1, 7, 30)),    # outpatient on an admission day
        visit(5, 9201, datetime(2020, 3, 1, 9), datetime(2020, 3, 1, 18)),       # same-day discharge
        visit(6, 9202, datetime(2020, 6, 1, 9), datetime(2020, 6, 1, 10)),       # outpatient, no later admission
    ]
    rows, audit = candidates_for_patient(visits, death_date=None, last_record=date(2020, 12, 31))
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append((row["index_visit_id"], row["label"]))

    assert by_task["los7"] == [(2, 1), (3, 0)]
    # Jan 10 discharge -> Jan 20 ICU (positive); Jan 22 discharge -> Mar 1, 39 days later (negative).
    assert by_task["readmit30"] == [(2, 1), (3, 0), (5, 0)]
    assert by_task["outpatient90"] == [(1, 1), (6, 0)]
    assert audit["los7_same_day_discharge"] == 1
    assert audit["outpatient90_same_day_admission"] == 1
