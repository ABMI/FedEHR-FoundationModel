"""Subgroup analysis: D3 FM minus other FMs on D5 test by age 65+ x cancer (patient and index-date definitions).

Reads the paired_ehrshot15.py probability cache; patient-level bootstrap, seed 42; writes analysis/subgroup_d5.json.
Usage: python scripts/analysis/subgroup_d5.py
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "outputs" / "sites_disjoint"
sys.path.insert(0, str(ROOT / "src"))
from ausom_ehr.ehrshot_tasks import RAW_ROOT, TASKS as NEW_TASKS  # noqa: E402

TASKS = ["los7", "readmit30", "outpatient90"] + NEW_TASKS
CACHE = SD / "analysis" / "probs_cache"
MODELS = ["D3_cancer", "D2_er", "A", "D0_random", "D4_female", "D1_pediatric", "fl6", "fl7", "local"]
OTHERS = ["D2_er", "A", "D0_random", "D4_female", "D1_pediatric", "fl6", "local"]
CANCER = 443392  # SNOMED malignant neoplastic disease; descendants count
CELLS = ["65+/cancer", "65+/no-cancer", "<65/cancer", "<65/no-cancer"]
MARGINS = {"all": None, "cancer": ("cancer", True), "no-cancer": ("cancer", False), "65+": ("old", True), "<65": ("old", False)}


def groups() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Patient-level profile flags, and each patient's first cancer diagnosis date for index-date flags."""
    profile = pl.read_parquet(SD.parent / "sites" / "patient_profile.parquet",
                              columns=["person_id", "year_of_birth", "age_65_plus", "dx_cancer"])
    descendants = (
        pl.scan_parquet(RAW_ROOT / "dbo" / "CONCEPT_ANCESTOR" / "**" / "*.parquet")
        .filter(pl.col("ancestor_concept_id").cast(pl.Int64) == CANCER)
        .select(pl.col("descendant_concept_id").cast(pl.Int64).alias("concept_id")).collect()
    )
    first_cancer = (
        pl.scan_parquet(SD / "D5_elderly" / "base" / "events" / "condition.parquet")
        .filter(pl.col("event_hold_reason").is_null())
        .select("person_id", "concept_id", "event_date").collect()
        .join(descendants, on="concept_id", how="inner")
        .group_by("person_id").agg(pl.col("event_date").min().alias("first_cancer"))
    )
    return profile, first_cancer


def label_rows(frame: pl.DataFrame, profile: pl.DataFrame, first_cancer: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.join(profile, on="person_id", how="left").join(first_cancer, on="person_id", how="left")
        .with_columns(
            pl.col("age_65_plus").alias("old_patient"), pl.col("dx_cancer").alias("cancer_patient"),
            ((pl.col("cutoff_date").dt.year() - pl.col("year_of_birth")) >= 65).alias("old_index"),
            (pl.col("first_cancer").is_not_null() & (pl.col("first_cancer") <= pl.col("cutoff_date"))).alias("cancer_index"),
        )
    )


def masks(old: np.ndarray, cancer: np.ndarray) -> dict:
    out = {"65+/cancer": old & cancer, "65+/no-cancer": old & ~cancer, "<65/cancer": ~old & cancer, "<65/no-cancer": ~old & ~cancer}
    for name, rule in MARGINS.items():
        if rule is None:
            out[name] = np.ones_like(old)
        else:
            column, value = rule
            base = cancer if column == "cancer" else old
            out[name] = base == value
    return out


def auc(y, p):
    return np.nan if y.min() == y.max() else roc_auc_score(y, p)


def delta(y, a, b, m):
    if m.sum() == 0:
        return np.nan
    yy = y[m]
    if yy.min() == yy.max():
        return np.nan
    return roc_auc_score(yy, a[m]) - roc_auc_score(yy, b[m])


def job(item):
    task, other, y, pa, pb, inverse, definitions = item
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(counts))]
    rng = np.random.default_rng(42)
    names = CELLS + list(MARGINS)
    draws = {d: {n: [] for n in names + ["cancer−no-cancer", "65+−<65"]} for d in definitions}
    for _ in range(1000):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        yy, aa, bb = y[idx], pa[idx], pb[idx]
        for d, (old, cancer) in definitions.items():
            ms = masks(old[idx], cancer[idx])
            values = {n: delta(yy, aa, bb, ms[n]) for n in names}
            for n in names:
                draws[d][n].append(values[n])
            draws[d]["cancer−no-cancer"].append(values["cancer"] - values["no-cancer"])
            draws[d]["65+−<65"].append(values["65+"] - values["<65"])
    out = []
    for d, (old, cancer) in definitions.items():
        ms = masks(old, cancer)
        point = {n: delta(y, pa, pb, ms[n]) for n in names}
        point["cancer−no-cancer"] = point["cancer"] - point["no-cancer"]
        point["65+−<65"] = point["65+"] - point["<65"]
        for n in names + ["cancer−no-cancer", "65+−<65"]:
            v = np.asarray(draws[d][n], dtype=float)
            v = v[~np.isnan(v)]
            record = {"definition": d, "task": task, "a": "D3_cancer", "b": other, "group": n,
                      "d_auroc": float(point[n]) if not np.isnan(point[n]) else None, "rounds": int(len(v))}
            if n in ms:
                m = ms[n]
                record.update(n=int(m.sum()), positives=int(y[m].sum()), patients=int(len(np.unique(inverse[m]))),
                              positive_patients=int(len(np.unique(inverse[m & (y == 1)]))),
                              auroc_a=float(auc(y[m], pa[m])) if m.sum() else None,
                              auroc_b=float(auc(y[m], pb[m])) if m.sum() else None)
            if len(v) >= 100:
                record.update(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])],
                              p_two_sided=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
            out.append(record)
    return out


def main() -> None:
    profile, first_cancer = groups()
    items, cell_auroc, sizes = [], [], {}
    for task in TASKS:
        frames = {m: pl.read_parquet(CACHE / f"{m}@D5_elderly__{task}.parquet") for m in MODELS}
        key = ["person_id", "cutoff_date", "index_visit_id"]
        base = frames["D3_cancer"].rename({"p": "p_D3_cancer"})
        for m in MODELS[1:]:
            base = base.join(frames[m].select(*key, pl.col("p").alias(f"p_{m}")), on=key, how="inner", nulls_equal=True)
        assert base.height == frames["D3_cancer"].height, task
        base = label_rows(base, profile, first_cancer)
        assert base["old_patient"].null_count() == 0 and base["year_of_birth"].null_count() == 0, task
        y = base["label"].to_numpy()
        _, inverse = np.unique(base["person_id"].to_numpy(), return_inverse=True)
        definitions = {
            "patient": (base["old_patient"].to_numpy(), base["cancer_patient"].to_numpy()),
            "index": (base["old_index"].to_numpy(), base["cancer_index"].to_numpy()),
        }
        for d, (old, cancer) in definitions.items():
            for n, m in masks(old, cancer).items():
                sizes[(d, task, n)] = (int(m.sum()), int(y[m].sum()), int(len(np.unique(inverse[m & (y == 1)]))))
                cell_auroc.append({"definition": d, "task": task, "group": n, "n": int(m.sum()), "positives": int(y[m].sum()),
                                   "positive_patients": int(len(np.unique(inverse[m & (y == 1)]))),
                                   **{m_: (float(auc(y[m], base[f"p_{m_}"].to_numpy()[m])) if m.sum() else None) for m_ in MODELS}})
        for other in OTHERS:
            items.append((task, other, y, base["p_D3_cancer"].to_numpy(), base[f"p_{other}"].to_numpy(), inverse, definitions))
    # patient counts over the whole D5 test split, not one task
    test = pl.read_parquet(SD / "D5_elderly.parquet").filter(pl.col("split") == "test").join(profile, on="person_id", how="left")
    patients = test.group_by("age_65_plus", "dx_cancer").agg(pl.len().alias("patients")).sort("age_65_plus", "dx_cancer").to_dicts()
    print(f"{len(items)} jobs", flush=True)
    with mp.get_context("spawn").Pool(min(48, os.cpu_count())) as pool:
        results = [r for chunk in pool.map(job, items, chunksize=1) for r in chunk]
    out = SD / "analysis" / "subgroup_d5.json"
    out.write_text(json.dumps({"test_patients_by_patient_definition": patients, "cell_auroc": cell_auroc,
                               "paired": results}, indent=1, ensure_ascii=False))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
