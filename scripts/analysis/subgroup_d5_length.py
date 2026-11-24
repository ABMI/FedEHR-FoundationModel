"""Follow-up to subgroup_d5: is D3's edge on D5 cancer patients cancer knowledge or longer records?

D3 FM minus other FMs by input-length tertile (model input tokens) x cancer; writes analysis/subgroup_d5_length.json.
Usage: python scripts/analysis/subgroup_d5_length.py
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subgroup_d5 import CACHE, MODELS, OTHERS, SD, TASKS, auc, delta  # noqa: E402

OLD = ["los7", "readmit30", "outpatient90"]
LEVELS = ["T1", "T2", "T3"]


def input_lengths() -> pl.DataFrame:
    columns = ["person_id", "cutoff_date", "split", "input_length", "truncated_tokens"]
    old = (pl.scan_parquet(SD / "D5_elderly" / "joint" / "cohorts" / "inputs" / "*.parquet")
           .filter((pl.col("split") == "test") & pl.col("task").is_in(OLD)).select(columns))
    new = (pl.scan_parquet(SD / "tasks_ehrshot" / "D5_elderly" / "cohorts" / "inputs" / "*.parquet")
           .filter(pl.col("split") == "test").select(columns))
    both = pl.concat([old, new]).unique(["person_id", "cutoff_date", "input_length"]).collect()
    duplicated = both.group_by("person_id", "cutoff_date").len().filter(pl.col("len") > 1).height
    assert duplicated == 0, f"input length differs for the same (person, cutoff date): {duplicated}"
    return both.drop("split")


def masks(level: np.ndarray, cancer: np.ndarray) -> dict:
    out = {"all": np.ones_like(cancer), "cancer": cancer, "no-cancer": ~cancer}
    for k, name in enumerate(LEVELS):
        out[name] = level == k
        out[f"{name}/cancer"] = (level == k) & cancer
        out[f"{name}/no-cancer"] = (level == k) & ~cancer
    return out


def summarize(values: dict) -> dict:
    v = dict(values)
    for name in LEVELS:
        v[f"{name} cancer−no-cancer"] = v[f"{name}/cancer"] - v[f"{name}/no-cancer"]
    v["cancer−no-cancer"] = v["cancer"] - v["no-cancer"]
    v["length-adjusted cancer−no-cancer"] = float(np.mean([v[f"{name} cancer−no-cancer"] for name in LEVELS]))
    v["T3−T1"] = v["T3"] - v["T1"]
    v["T3−T1 (no-cancer)"] = v["T3/no-cancer"] - v["T1/no-cancer"]
    v["T3−T1 (cancer)"] = v["T3/cancer"] - v["T1/cancer"]
    v["cancer-adjusted T3−T1"] = float(np.mean([v["T3−T1 (cancer)"], v["T3−T1 (no-cancer)"]]))  # length effect averaged within cancer strata
    return v


def job(item):
    task, other, y, pa, pb, inverse, level, cancer = item
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(counts))]
    rng = np.random.default_rng(42)
    full = masks(level, cancer)
    point = summarize({n: delta(y, pa, pb, m) for n, m in full.items()})
    draws = {n: [] for n in point}
    for _ in range(1000):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        ms = masks(level[idx], cancer[idx])
        values = summarize({n: delta(y[idx], pa[idx], pb[idx], m) for n, m in ms.items()})
        for n in draws:
            draws[n].append(values[n])
    out = []
    for n in point:
        v = np.asarray(draws[n], dtype=float)
        v = v[~np.isnan(v)]
        record = {"task": task, "a": "D3_cancer", "b": other, "group": n,
                  "d_auroc": None if np.isnan(point[n]) else float(point[n]), "rounds": int(len(v))}
        if n in full:
            m = full[n]
            record.update(n=int(m.sum()), positives=int(y[m].sum()),
                          positive_patients=int(len(np.unique(inverse[m & (y == 1)]))))
        if len(v) >= 100:
            record.update(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])],
                          p_two_sided=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
        out.append(record)
    return out


def main() -> None:
    lengths = input_lengths()
    profile = pl.read_parquet(SD.parent / "sites" / "patient_profile.parquet", columns=["person_id", "dx_cancer"])
    items, describe = [], []
    key = ["person_id", "cutoff_date", "index_visit_id"]
    for task in TASKS:
        base = pl.read_parquet(CACHE / f"D3_cancer@D5_elderly__{task}.parquet").rename({"p": "p_D3_cancer"})
        for m in MODELS[1:]:
            base = base.join(pl.read_parquet(CACHE / f"{m}@D5_elderly__{task}.parquet").select(*key, pl.col("p").alias(f"p_{m}")),
                             on=key, how="inner", nulls_equal=True)
        rows = base.height
        base = base.join(lengths, on=["person_id", "cutoff_date"], how="left").join(profile, on="person_id", how="left")
        assert base.height == rows and base["input_length"].null_count() == 0 and base["dx_cancer"].null_count() == 0, task
        cuts = np.quantile(base["input_length"].to_numpy(), [1 / 3, 2 / 3])
        level = np.digitize(base["input_length"].to_numpy(), cuts, right=True)
        cancer = base["dx_cancer"].to_numpy()
        y = base["label"].to_numpy()
        _, inverse = np.unique(base["person_id"].to_numpy(), return_inverse=True)
        length = base["input_length"].to_numpy()
        info = {"task": task, "cuts": [float(c) for c in cuts], "samples": rows,
                "truncated_share": float((base["truncated_tokens"].to_numpy() > 0).mean()),
                "median_length_cancer": float(np.median(length[cancer])), "median_length_no_cancer": float(np.median(length[~cancer])),
                "cancer_share_by_level": [float(cancer[level == k].mean()) for k in range(3)],
                "auroc_by_level": {name: {m: float(auc(y[level == k], base[f"p_{m}"].to_numpy()[level == k])) for m in MODELS}
                                   for k, name in enumerate(LEVELS)}}
        describe.append(info)
        for other in OTHERS:
            items.append((task, other, y, base["p_D3_cancer"].to_numpy(), base[f"p_{other}"].to_numpy(), inverse, level, cancer))
    print(f"{len(items)} jobs", flush=True)
    with mp.get_context("spawn").Pool(min(48, os.cpu_count())) as pool:
        results = [r for chunk in pool.map(job, items, chunksize=1) for r in chunk]
    out = SD / "analysis" / "subgroup_d5_length.json"
    out.write_text(json.dumps({"describe": describe, "paired": results}, indent=1, ensure_ascii=False))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
