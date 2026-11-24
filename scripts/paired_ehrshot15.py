"""Paired test AUROC comparisons on 15 tasks (patient-level bootstrap, 1,000 rounds, seed 42).

fl7_vs_local, fl7_vs_fl6 (FedAvg-6 minus D5) and fl6_vs_x@D5; writes analysis/paired_ehrshot15.json.
Usage: python scripts/paired_ehrshot15.py [--fl6 SITE]   (--fl6: FedAvg-6 minus SITE vs FL7 and local only)
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "outputs" / "sites_disjoint"
sys.path.insert(0, str(ROOT / "src"))
from ausom_ehr.ehrshot_tasks import TASKS as NEW_TASKS  # noqa: E402

OLD_TASKS = ["los7", "readmit30", "outpatient90"]
TASKS = OLD_TASKS + NEW_TASKS
SITES = ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female", "D5_elderly"]
CACHE = SD / "analysis" / "probs_cache"
EXCLUDE = sys.argv[sys.argv.index("--fl6") + 1] if "--fl6" in sys.argv else "D5_elderly"
FL6 = f"fedavg6_minus_{EXCLUDE}_round20"


def old_folder(model: str, site: str) -> Path:
    if model == "local":
        return SD / "eval" / "A_epoch20" if site == "A" else SD / site / "joint"
    if model == "fl7":
        return SD / "eval" / f"fedavg7_round20@{site}"
    if model == "fl6":
        return SD / "eval" / f"{FL6}@{site}"
    return SD / "eval" / f"{model}_epoch20@{site}"  # cross-site: model is the source site


def new_folder(model: str, site: str) -> Path:
    name = {"local": f"{site}_epoch20", "fl7": "fedavg7_round20", "fl6": FL6}.get(model, f"{model}_epoch20")
    return SD / "eval_ehrshot" / f"{name}@{site}"


def probabilities(model: str, site: str, task: str) -> pl.DataFrame:
    """Test rows (person_id, cutoff_date, index_visit_id, label, p), cached; AUROC must match results.json."""
    key = f"fl6_minus_{EXCLUDE}" if model == "fl6" and EXCLUDE != "D5_elderly" else model
    cache = CACHE / f"{key}@{site}__{task}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    if task in OLD_TASKS:
        folder = old_folder(model, site)
        data = pl.read_parquet(folder / "embeddings/continued/epoch20" / f"{task}.parquet").filter(pl.col("split") == "test")
        lr = joblib.load(folder / "evaluation/continued/epoch20" / f"{task}_model.joblib")
        lr = lr["model"] if isinstance(lr, dict) else lr
        p = lr.predict_proba(np.asarray(data["embedding"].to_list(), dtype=np.float32))[:, 1]
        frame = data.select("person_id", "cutoff_date", "index_visit_id", "label").with_columns(pl.Series("p", p))
        reference = json.loads((folder / "evaluation/continued/epoch20/results.json").read_text())
    else:
        folder = new_folder(model, site)
        frame = (pl.read_parquet(folder / "evaluation/continued/epoch20/test_predictions.parquet")
                 .filter(pl.col("task") == task).select("person_id", "cutoff_date", "index_visit_id", "label", pl.col("probability").alias("p")))
        reference = json.loads((folder / "evaluation/continued/epoch20/results.json").read_text())
    expected = next(t for t in reference["tasks"] if t["task"] == task)["test"]["auroc"]
    got = roc_auc_score(frame["label"].to_numpy(), frame["p"].to_numpy())
    assert abs(got - expected) < 1e-6, (model, site, task, got, expected)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache)
    return frame


def bootstrap(item):
    meta, y, pa, pb, inverse = item
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.r_[0, np.cumsum(counts)]
    rows = [order[starts[i]:starts[i + 1]] for i in range(len(counts))]
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(1000):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    diffs = np.asarray(diffs)
    return {**meta, "n": int(len(y)), "positives": int(y.sum()), "patients": int(len(counts)),
            "positive_patients": int(len(np.unique(inverse[y == 1]))),
            "auroc_a": float(roc_auc_score(y, pa)), "auroc_b": float(roc_auc_score(y, pb)),
            "auprc_a": float(average_precision_score(y, pa)), "auprc_b": float(average_precision_score(y, pb)),
            "d_auroc": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "ci": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
            "se": float(diffs.std(ddof=1)), "rounds": int(len(diffs)),
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def pair(comparison: str, site: str, task: str, a: str, b: str):
    left, right = probabilities(a, site, task), probabilities(b, site, task)
    key = ["person_id", "cutoff_date", "index_visit_id"]
    joined = left.join(right.select(*key, pl.col("p").alias("q"), pl.col("label").alias("label_b")), on=key, how="inner", nulls_equal=True)
    assert joined.height == left.height == right.height, (comparison, site, task, joined.height, left.height, right.height)
    assert (joined["label"] == joined["label_b"]).all()
    _, inverse = np.unique(joined["person_id"].to_numpy(), return_inverse=True)
    meta = {"comparison": comparison, "site": site, "task": task, "a": a, "b": b}
    return meta, joined["label"].to_numpy(), joined["p"].to_numpy(), joined["q"].to_numpy(), inverse


def main() -> None:
    specs = []
    for task in TASKS:
        if EXCLUDE != "D5_elderly":
            specs += [("fl7_vs_fl6", site, task, "fl7", "fl6") for site in SITES]
            specs.append(("fl6_vs_local", EXCLUDE, task, "fl6", "local"))
            continue
        for site in SITES:
            specs.append(("fl7_vs_local", site, task, "fl7", "local"))
            specs.append(("fl7_vs_fl6", site, task, "fl7", "fl6"))
        for source in ["A", "D0_random", "D1_pediatric", "D2_er", "D3_cancer", "D4_female"]:
            specs.append(("fl6_vs_x@D5", "D5_elderly", task, "fl6", source))
        specs.append(("fl6_vs_x@D5", "D5_elderly", task, "fl6", "local"))
    items = [pair(*spec) for spec in specs]
    print(f"{len(items)} pairs ready (FedAvg-6 = minus {EXCLUDE})", flush=True)
    with mp.get_context("spawn").Pool(min(48, os.cpu_count())) as pool:
        results = pool.map(bootstrap, items, chunksize=1)
    for r in results:
        r["fl6_excluded"] = EXCLUDE
    out = SD / "analysis" / ("paired_ehrshot15.json" if EXCLUDE == "D5_elderly" else f"paired_fl6_minus_{EXCLUDE}.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
