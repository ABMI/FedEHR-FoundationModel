"""Paired transfer comparisons on evaluation site T (15 tasks, LR trained on T, patient-level bootstrap).

x_vs_local@T: other site FM - T FM; fl6_vs_x@T: FedAvg-6 (minus T) - each FM. Writes paired_transfer_to_{T}.json.
Usage: python scripts/paired_transfer.py --fl6 T
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_ehrshot15 import EXCLUDE, OLD_TASKS, SD, SITES, TASKS, bootstrap, new_folder, old_folder, pair  # noqa: E402

TARGET = EXCLUDE
SOURCES = [s for s in SITES if s != TARGET]


def ready(task: str) -> bool:
    """True once every source FM has finished this task on T; rerun later to fill in the rest."""
    if task in OLD_TASKS:
        return all((old_folder(s, TARGET) / "evaluation/continued/epoch20/results.json").exists() for s in SOURCES)
    return all((new_folder(s, TARGET) / "evaluation/continued/epoch20/test_predictions.parquet").exists() for s in SOURCES)


def main() -> None:
    assert "--fl6" in sys.argv, "pass --fl6 EVAL_SITE"
    tasks = [t for t in TASKS if ready(t)]
    print(f"{len(tasks)}/{len(TASKS)} tasks ready", flush=True)
    specs = [(f"x_vs_local@{TARGET}", task, source, "local") for task in tasks for source in SOURCES]
    specs += [(f"fl6_vs_x@{TARGET}", task, "fl6", x) for task in tasks for x in SOURCES + ["local"]]
    items = [pair(c, TARGET, task, a, b) for c, task, a, b in specs]
    print(f"{len(items)} pairs ready (evaluation site {TARGET})", flush=True)
    with mp.get_context("spawn").Pool(min(24, os.cpu_count())) as pool:
        results = pool.map(bootstrap, items, chunksize=1)
    out = SD / "analysis" / f"paired_transfer_to_{TARGET}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
