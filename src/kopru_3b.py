"""
ENG401 - Phase 2: 3D visualization of the undamaged mode shapes of the Rt345 bridge

Usage :  python src/kopru_3b.py
Output:  2.Aşama/bridge_modes_3d.html  (opens in a browser, single page)

Data sources:
- Mode shapes     : 2.Aşama/Hasarsız Modlar_son/Hasarsız Modlar_son/Baseline<b>/TEST <t>/*_<y|z>.xlsx
                    (row i = sensor A<i>, columns Mode 1..8)
- Sensor positions: dataset/Rt345Bridge.h5.v2/Rt345Bridge.h5, 'position' attribute of A01..A30
                    (the mode shape files contain no positions)

Required packages:  pip install h5py numpy
"""
import json
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MODE_FOLDER = PROJECT / "2.Aşama" / "Hasarsız Modlar_son" / "Hasarsız Modlar_son"
RT345_H5 = PROJECT / "dataset" / "Rt345Bridge.h5.v2" / "Rt345Bridge.h5"
OUTPUT = PROJECT / "2.Aşama" / "bridge_modes_3d.html"
TEMPLATE = Path(__file__).with_name("kopru_3b_sablon.html")

SENSORS = [f"A{i:02d}" for i in range(1, 31)]


def read_xlsx(path):
    """Numeric rows of a single-sheet xlsx (a header row, if any, is skipped).
    The XML inside the xlsx is read directly so that openpyxl is not required."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("xl/worksheets/sheet1.xml").decode()
    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>", xml):
        if 't="s"' in r:  # text cell -> header row ("Mod 1" ...)
            continue
        v = re.findall(r"<v>(.*?)</v>", r)
        if v:
            rows.append([float(a) for a in v])
    X = np.array(rows)
    assert X.shape == (30, 8), f"{path}: expected 30x8, got {X.shape}"
    return X


def sensor_positions():
    with h5py.File(RT345_H5, "r") as f:
        T = f["baseline_1/acceleration/test_1"]
        return {s: np.ravel(T[s].attrs["position"]).round(3).tolist() for s in SENSORS}


def build_data():
    scenarios = []
    for b in (1, 2):
        for t in (1, 2, 3):
            folder = MODE_FOLDER / f"Baseline{b}" / f"TEST {t}"
            y = read_xlsx(next(folder.glob("*_y.xlsx")))
            z = read_xlsx(next(folder.glob("*_z.xlsx")))
            scenarios.append({"name": f"Baseline {b} · Test {t}",
                              "y": y.T.tolist(),   # [mode][sensor], xlsx values without rounding
                              "z": z.T.tolist()})
    return {"sensors": SENSORS, "positions": sensor_positions(), "scenarios": scenarios}


if __name__ == "__main__":
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(build_data()))
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"{OUTPUT.relative_to(PROJECT)} written")
