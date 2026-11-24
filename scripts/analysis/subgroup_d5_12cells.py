"""D3 FM minus other FMs on D5 test in 12 groups: age 65+ x cancer x input-length tertile.

Age, cancer and length effects average over strata where both groups have >= 30 positive patients.
Usage: python scripts/analysis/subgroup_d5_12cells.py   (writes analysis/subgroup_d5_12cells.json)
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
from subgroup_d5_length import input_lengths  # noqa: E402

AGES = ["65+", "<65"]
CANCERS = ["cancer", "no-cancer"]
LEVELS = ["T1", "T2", "T3"]
CELLS = [f"{a}/{c}/{l}" for a in AGES for c in CANCERS for l in LEVELS]  # groups 1-12 in this order
MIN_POSITIVE_PATIENTS = 30


def cell_masks(old: np.ndarray, cancer: np.ndarray, level: np.ndarray) -> dict:
    out = {}
    for a, is_old in zip(AGES, [True, False]):
        for c, has in zip(CANCERS, [True, False]):
            for k, l in enumerate(LEVELS):
                out[f"{a}/{c}/{l}"] = (old == is_old) & (cancer == has) & (level == k)
    return out


def contrasts() -> dict:
    """Effect name -> [(higher group, lower group), ...], one pair per fixed stratum."""
    return {
        "age effect (65+ − <65)": [(f"65+/{c}/{l}", f"<65/{c}/{l}") for c in CANCERS for l in LEVELS],
        "cancer effect (cancer − no-cancer)": [(f"{a}/cancer/{l}", f"{a}/no-cancer/{l}") for a in AGES for l in LEVELS],
        "length effect (T3 − T1)": [(f"{a}/{c}/T3", f"{a}/{c}/T1") for a in AGES for c in CANCERS],
    }


def job(item):
    task, other, y, pa, pb, inverse, old, cancer, level = item
    full = cell_masks(old, cancer, level)
    positive_patients = {n: int(len(np.unique(inverse[m & (y == 1)]))) for n, m in full.items()}
    usable = {name: [pair for pair in pairs if min(positive_patients[pair[0]], positive_patients[pair[1]]) >= MIN_POSITIVE_PATIENTS]
              for name, pairs in contrasts().items()}

    def evaluate(yy, aa, bb, masks):
        cells = {n: delta(yy, aa, bb, m) for n, m in masks.items()}
        effects = {name: (float(np.mean([cells[h] - cells[l] for h, l in pairs])) if pairs else np.nan)
                   for name, pairs in usable.items()}
        return cells, effects

    point_cells, point_effects = evaluate(y, pa, pb, full)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(counts))]
    rng = np.random.default_rng(42)
    draws = {n: [] for n in list(point_cells) + list(point_effects)}
    for _ in range(1000):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        cells, effects = evaluate(y[idx], pa[idx], pb[idx], cell_masks(old[idx], cancer[idx], level[idx]))
        for n, v in {**cells, **effects}.items():
            draws[n].append(v)

    def interval(name, value):
        v = np.asarray(draws[name], dtype=float)
        v = v[~np.isnan(v)]
        record = {"d_auroc": None if np.isnan(value) else float(value), "rounds": int(len(v))}
        if len(v) >= 100:
            record.update(ci=[float(x) for x in np.percentile(v, [2.5, 97.5])],
                          p_two_sided=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
        return record

    out = []
    for number, n in enumerate(CELLS, start=1):
        m = full[n]
        out.append({"task": task, "a": "D3_cancer", "b": other, "kind": "cell", "group": n, "group_number": number,
                    "n": int(m.sum()), "positives": int(y[m].sum()), "patients": int(len(np.unique(inverse[m]))),
                    "positive_patients": positive_patients[n],
                    "auroc_a": float(auc(y[m], pa[m])) if m.sum() else None,
                    "auroc_b": float(auc(y[m], pb[m])) if m.sum() else None,
                    **interval(n, point_cells[n])})
    for name in point_effects:
        out.append({"task": task, "a": "D3_cancer", "b": other, "kind": "effect", "group": name,
                    "strata_used": [f"{h} vs {l}" for h, l in usable[name]], "strata_total": len(contrasts()[name]),
                    **interval(name, point_effects[name])})
    return out


def main() -> None:
    lengths = input_lengths()
    profile = pl.read_parquet(SD.parent / "sites" / "patient_profile.parquet", columns=["person_id", "age_65_plus", "dx_cancer"])
    key = ["person_id", "cutoff_date", "index_visit_id"]
    items, cell_auroc = [], []
    for task in TASKS:
        base = pl.read_parquet(CACHE / f"D3_cancer@D5_elderly__{task}.parquet").rename({"p": "p_D3_cancer"})
        for m in MODELS[1:]:
            base = base.join(pl.read_parquet(CACHE / f"{m}@D5_elderly__{task}.parquet").select(*key, pl.col("p").alias(f"p_{m}")),
                             on=key, how="inner", nulls_equal=True)
        rows = base.height
        base = base.join(lengths, on=["person_id", "cutoff_date"], how="left").join(profile, on="person_id", how="left")
        assert base.height == rows and base["input_length"].null_count() == 0 and base["dx_cancer"].null_count() == 0, task
        length = base["input_length"].to_numpy()
        cuts = np.quantile(length, [1 / 3, 2 / 3])
        level = np.digitize(length, cuts, right=True)
        old, cancer, y = base["age_65_plus"].to_numpy(), base["dx_cancer"].to_numpy(), base["label"].to_numpy()
        _, inverse = np.unique(base["person_id"].to_numpy(), return_inverse=True)
        for number, (n, m) in enumerate(cell_masks(old, cancer, level).items(), start=1):
            cell_auroc.append({"task": task, "cuts": [float(c) for c in cuts], "group": n, "group_number": number,
                               "n": int(m.sum()), "positives": int(y[m].sum()),
                               "positive_patients": int(len(np.unique(inverse[m & (y == 1)]))),
                               "median_length": float(np.median(length[m])) if m.sum() else None,
                               **{f"auroc_{m_}": (float(auc(y[m], base[f"p_{m_}"].to_numpy()[m])) if m.sum() else None) for m_ in MODELS}})
        for other in OTHERS:
            items.append((task, other, y, base["p_D3_cancer"].to_numpy(), base[f"p_{other}"].to_numpy(), inverse, old, cancer, level))
    print(f"{len(items)} jobs", flush=True)
    with mp.get_context("spawn").Pool(min(48, os.cpu_count())) as pool:
        results = [r for chunk in pool.map(job, items, chunksize=1) for r in chunk]
    out = SD / "analysis" / "subgroup_d5_12cells.json"
    out.write_text(json.dumps({"cell_auroc": cell_auroc, "paired": results}, indent=1, ensure_ascii=False))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
