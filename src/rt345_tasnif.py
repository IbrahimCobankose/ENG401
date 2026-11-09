"""
ENG401 - Classification of the Route 345 Bridge acceleration data into folders

Usage :  python src/rt345_tasnif.py
Output:  into the Rt345Bridge.h5/ folder at the project root
    Undamaged/<baseline_k>/<test_k>/undamaged_<baselinek>_<testk>_<y|z>.csv
    Damaged/<damage_k>/<test_k>/damaged_<damagek>_<testk>_<y|z>.csv

Each CSV is one direction of one test: numbers only, no header row and no time column.
    rows    : samples, 128 Hz ('sample_rate' attribute in the h5 file) -> row i, t = i/128 s
    columns : sensors A01..A30 in this order (30 columns), unit m/s^2

Required packages:  pip install h5py numpy pandas
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
# Source: Zenodo v2 (doi:10.5281/zenodo.15497673); v1 is kept under _eski_tasnif/dataset/
RT345_H5 = PROJECT / "dataset" / "Rt345Bridge.h5.v2" / "Rt345Bridge.h5"

TARGET = PROJECT / "Rt345Bridge.h5"
TOP_FOLDER = {"baseline": "Undamaged", "damage": "Damaged"}
SENSORS = [f"A{i:02d}" for i in range(1, 31)]  # column order


def classify():
    with h5py.File(RT345_H5, "r") as f:
        for scenario in f:
            kind = scenario.split("_")[0]  # baseline | damage
            top = TOP_FOLDER[kind]
            for test, T in f[scenario]["acceleration"].items():
                assert sorted(T.keys()) == SENSORS, f"{scenario}/{test}: unexpected sensor list"
                target = TARGET / top / scenario / test
                target.mkdir(parents=True, exist_ok=True)
                for axis in ("y", "z"):
                    X = np.column_stack([T[s][axis][()].ravel() for s in SENSORS])
                    name = f"{top.lower()}_{scenario.replace('_', '')}_{test.replace('_', '')}_{axis}.csv"
                    pd.DataFrame(X).to_csv(target / name, index=False, header=False)
                    print(target.relative_to(PROJECT) / name, X.shape, flush=True)


if __name__ == "__main__":
    classify()
