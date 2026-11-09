"""
ENG401 - Classification of the SDOF SHM benchmark acceleration data into one CSV per scenario

Usage:
    python src/sdof_tasnif.py                 # all scenarios in SCENARIOS
    python src/sdof_tasnif.py D1 D2_1         # only the selected scenarios
    python src/sdof_tasnif.py D1 --limit 5    # trial run: first 5 hours of each scenario

Output:  SDOF/<scenario>.csv   (D1.csv, D2_1.csv, ..., D5.csv)

Same layout as the Rt345 CSVs: numbers only, no header row and no time column.
    rows   : time samples, 100 Hz
    column : single sensor (1 column)
The hourly records (acc<Index>-<k>.h5, 18000 samples = 3 min each) are appended in Index
order: row r -> record r // 18000, time within the record t = (r % 18000) / 100 s.
Time is not continuous between records (only 3 min of each hour are recorded); the hour of
each record follows the order of the file names in dataset/output/output/acceleration/<scenario>/.

The file is written chunk by chunk; only --chunk hours are kept in memory at a time.

Reading example (without loading everything at once, each chunk = 1 hour):
    for x in pd.read_csv("SDOF/D2_2.csv", header=None, chunksize=18000): ...

Required packages:  pip install h5py numpy pandas
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SDOF_OUT = PROJECT / "dataset" / "output" / "output"
TARGET = PROJECT / "SDOF"

SCENARIOS = ["D1", "D2_1", "D2_2", "D2_3", "D3_1", "D3_2", "D3_3", #"D4",
              "D5"]
N_SAMPLES = 18000


def _read(path):
    with h5py.File(path, "r") as f:
        x = f["acc"][()].ravel()  # (1, 18000) -> (18000,)
    assert x.size == N_SAMPLES, f"{path.name}: {x.size} samples"
    return x


def scenario_csv(scenario, limit=None, chunk=500):
    # file names are acc<5-digit Index>-<k>.h5: name order = time order
    files = sorted((SDOF_OUT / "acceleration" / scenario).glob("acc*.h5"))[:limit]
    target = TARGET / f"{scenario}.csv"
    t0 = time.time()
    with open(target, "w", newline="", encoding="utf-8") as out:
        for start in range(0, len(files), chunk):
            x = np.concatenate([_read(p) for p in files[start:start + chunk]])
            # no float_format: float32 values are written in their shortest lossless form
            pd.DataFrame(x).to_csv(out, index=False, header=False)
            n, N = min(start + chunk, len(files)), len(files)
            elapsed = time.time() - t0
            print(f"  {scenario}: {n}/{N} hours  {100 * n / N:.0f}%  "
                  f"elapsed {elapsed / 60:.1f} min, remaining ~{elapsed / n * (N - n) / 60:.1f} min", flush=True)
    print(f"  -> {target.relative_to(PROJECT)} written ({len(files) * N_SAMPLES} rows)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=SCENARIOS)
    ap.add_argument("--limit", type=int, help="first N hours of each scenario (for a trial run)")
    ap.add_argument("--chunk", type=int, default=500, help="number of hours kept in memory at a time")
    a = ap.parse_args()
    TARGET.mkdir(exist_ok=True)
    t0 = time.time()
    for i, sc in enumerate(a.scenarios, 1):
        print(f"[{i}/{len(a.scenarios)}] {sc}", flush=True)
        scenario_csv(sc, a.limit, a.chunk)
    print(f"Done: {len(a.scenarios)} files, {(time.time() - t0) / 60:.1f} min -> {TARGET}")
