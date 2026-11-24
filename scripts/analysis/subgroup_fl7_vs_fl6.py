"""Subgroup analysis of D5's participation effect (FedAvg-7 minus FedAvg-6 without D5) on A-D4 test.

Same groups as subgroup_d5_12cells, per site and pooled over 6 sites; writes analysis/subgroup_fl7_vs_fl6.json.
Usage: python scripts/analysis/subgroup_fl7_vs_fl6.py
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subgroup_d5 import CACHE, SD, TASKS, auc, delta  # noqa: E402
from subgroup_d5_12cells import CELLS, MIN_POSITIVE_PATIENTS, cell_masks, contrasts  # noqa: E402

ROOT = SD.parents[1]
SITES = ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female"]
OLD = ["los7", "readmit30", "outpatient90"]
MARGINS = ["all", "65+", "<65", "cancer", "no-cancer", "T1", "T2", "T3"]


def input_lengths(site: str) -> pl.DataFrame:
    population = ROOT / "outputs" / "m0" if site == "A" else SD / site / "joint"
    columns = ["person_id", "cutoff_date", "input_length"]
    old = (pl.scan_parquet(population / "cohorts" / "inputs" / "*.parquet")
           .filter((pl.col("split") == "test") & pl.col("task").is_in(OLD)).select(columns))
    new = (pl.scan_parquet(SD / "tasks_ehrshot" / site / "cohorts" / "inputs" / "*.parquet")
           .filter(pl.col("split") == "test").select(columns))
    both = pl.concat([old, new]).unique().collect()
    duplicated = both.group_by("person_id", "cutoff_date").len().filter(pl.col("len") > 1).height
    assert duplicated == 0, f"{site}: input length differs for the same (person, cutoff date): {duplicated}"
    return both


def all_masks(old: np.ndarray, cancer: np.ndarray, level: np.ndarray) -> dict:
    out = cell_masks(old, cancer, level)
    out.update({"all": np.ones_like(old, dtype=bool), "65+": old, "<65": ~old, "cancer": cancer, "no-cancer": ~cancer,
                "T1": level == 0, "T2": level == 1, "T3": level == 2})
    return out


def job(item):
    site, task, y, pa, pb, inverse, old, cancer, level = item
    full = all_masks(old, cancer, level)
    positive_patients = {n: int(len(np.unique(inverse[m & (y == 1)]))) for n, m in full.items()}
    usable = {name: [pair for pair in pairs if min(positive_patients[pair[0]], positive_patients[pair[1]]) >= MIN_POSITIVE_PATIENTS]
              for name, pairs in contrasts().items()}

    def evaluate(yy, aa, bb, masks):
        values = {n: delta(yy, aa, bb, m) for n, m in masks.items()}
        for name, pairs in usable.items():
            values[name] = float(np.mean([values[h] - values[l] for h, l in pairs])) if pairs else np.nan
        return values

    point = evaluate(y, pa, pb, full)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(counts))]
    rng = np.random.default_rng(42)
    draws = {n: np.full(1000, np.nan) for n in point}
    for r in range(1000):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        for n, v in evaluate(y[idx], pa[idx], pb[idx], all_masks(old[idx], cancer[idx], level[idx])).items():
            draws[n][r] = v

    records = []
    for n in CELLS + MARGINS:
        m = full[n]
        records.append({"site": site, "task": task, "kind": "cell" if n in CELLS else "margin", "group": n,
                        "n": int(m.sum()), "positives": int(y[m].sum()), "patients": int(len(np.unique(inverse[m]))),
                        "positive_patients": positive_patients[n],
                        "auroc_fl7": float(auc(y[m], pa[m])) if m.sum() else None,
                        "auroc_fl6": float(auc(y[m], pb[m])) if m.sum() else None,
                        **interval(draws[n], point[n])})
    for name in usable:
        records.append({"site": site, "task": task, "kind": "effect", "group": name,
                        "strata_used": [f"{h} vs {l}" for h, l in usable[name]], "strata_total": len(contrasts()[name]),
                        **interval(draws[name], point[name])})
    return site, task, records, point, draws


def interval(draws: np.ndarray, value: float) -> dict:
    v = draws[~np.isnan(draws)]
    record = {"d_auroc": None if np.isnan(value) else float(value), "rounds": int(len(v))}
    if len(v) >= 100:
        record.update(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])],
                      p_two_sided=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
    return record


def pooled(results: list) -> list:
    """Equal-weight mean over sites per task; sites without a value (too few strata) are skipped."""
    out = []
    for task in TASKS:
        rows = [(site, point, draws) for site, t, _, point, draws in results if t == task]
        for name in MARGINS + list(contrasts()):
            sites = [s for s, point, _ in rows if not np.isnan(point[name])]
            if not sites:
                continue
            value = float(np.mean([point[name] for s, point, _ in rows if s in sites]))
            stacked = np.vstack([draws[name] for s, _, draws in rows if s in sites])
            out.append({"task": task, "group": name, "sites": sites, **interval(stacked.mean(axis=0), value)})
    return out


def main() -> None:
    profile = pl.read_parquet(SD.parent / "sites" / "patient_profile.parquet", columns=["person_id", "age_65_plus", "dx_cancer"])
    reference = {(r["site"], r["task"]): r["d_auroc"] for r in json.loads((SD / "analysis" / "paired_ehrshot15.json").read_text())
                 if r["a"] == "fl7" and r["b"] == "fl6"}
    key = ["person_id", "cutoff_date", "index_visit_id"]
    items, cell_auroc = [], []
    for site in SITES:
        lengths = input_lengths(site)
        for task in TASKS:
            base = pl.read_parquet(CACHE / f"fl7@{site}__{task}.parquet").rename({"p": "p_fl7"})
            base = base.join(pl.read_parquet(CACHE / f"fl6@{site}__{task}.parquet").select(*key, pl.col("p").alias("p_fl6")),
                             on=key, how="inner", nulls_equal=True)
            rows = base.height
            base = base.join(lengths, on=["person_id", "cutoff_date"], how="left").join(profile, on="person_id", how="left")
            assert base.height == rows and base["input_length"].null_count() == 0 and base["dx_cancer"].null_count() == 0, (site, task)
            y, pa, pb = base["label"].to_numpy(), base["p_fl7"].to_numpy(), base["p_fl6"].to_numpy()
            check = delta(y, pa, pb, np.ones(rows, dtype=bool))
            assert abs(check - reference[(site, task)]) < 1e-9, (site, task, check, reference[(site, task)])
            length = base["input_length"].to_numpy()
            cuts = np.quantile(length, [1 / 3, 2 / 3])
            level = np.digitize(length, cuts, right=True)
            old, cancer = base["age_65_plus"].to_numpy(), base["dx_cancer"].to_numpy()
            _, inverse = np.unique(base["person_id"].to_numpy(), return_inverse=True)
            for n, m in all_masks(old, cancer, level).items():
                cell_auroc.append({"site": site, "task": task, "cuts": [float(c) for c in cuts], "group": n, "n": int(m.sum()),
                                   "positives": int(y[m].sum()), "positive_patients": int(len(np.unique(inverse[m & (y == 1)]))),
                                   "median_length": float(np.median(length[m])) if m.sum() else None})
            items.append((site, task, y, pa, pb, inverse, old, cancer, level))
        print(f"{site} ready (overall delta matches paired_ehrshot15.json)", flush=True)
    print(f"{len(items)} jobs", flush=True)
    with mp.get_context("spawn").Pool(40) as pool:
        results = pool.map(job, items, chunksize=1)
    out = SD / "analysis" / "subgroup_fl7_vs_fl6.json"
    out.write_text(json.dumps({"cell_sizes": cell_auroc, "paired": [r for _, _, records, _, _ in results for r in records],
                               "pooled": pooled(results)}, indent=1, ensure_ascii=False))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
