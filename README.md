# ENG401 – Uncertainty-Aware Structural Anomaly Detection Using Open Vibration Data

This repository contains the software developed within the ENG401 interdisciplinary project
(Civil Engineering and Computer Engineering). The aim of the project is to learn the normal
dynamic behaviour of a structure from healthy vibration records and to evaluate new records as
**normal, uncertain or anomalous**.

The current phase covers:

1. **Data classification** – converting the acceleration records of two open datasets into a common
   CSV layout (rows = time samples, columns = sensors).
2. **3D visualization** – an interactive page that displays the undamaged mode shapes of the
   Route 345 bridge at the sensor locations.

## Datasets

The datasets are not stored in this repository. Download them and place them as shown below.

| Dataset | Source | Expected location |
|---|---|---|
| Route 345 Bridge (version 2) | Whelan, Gangone, Janoyan – [doi:10.5281/zenodo.15497673](https://doi.org/10.5281/zenodo.15497673) | `dataset/Rt345Bridge.h5.v2/Rt345Bridge.h5` |
| SDOF SHM benchmark | Marafini – [doi:10.5281/zenodo.17900302](https://doi.org/10.5281/zenodo.17900302) | `dataset/output/output/` (contents of `output.rar`) |

## Installation

Python 3.12 is used.

```bash
pip install -r requirements.txt
```

## Usage

All scripts are run from the project root.

### Route 345 classification

```bash
python src/rt345_tasnif.py
```

Writes one CSV file per test and direction:

```
Rt345Bridge.h5/
├── Undamaged/baseline_1/test_1/undamaged_baseline1_test1_y.csv
│                              /undamaged_baseline1_test1_z.csv
│   …
└── Damaged/damage_1/test_1/damaged_damage1_test1_y.csv
    …
```

Each file has no header row and no time column: 30 columns are the sensors A01–A30 in this order,
row *i* corresponds to *t = i / 128* s.

### SDOF classification

```bash
python src/sdof_tasnif.py                 # all scenarios
python src/sdof_tasnif.py D1 D2_1         # selected scenarios
python src/sdof_tasnif.py D1 --limit 5    # trial run: first 5 hours of each scenario
```

Writes one single-column CSV file per scenario (`SDOF/D1.csv`, `SDOF/D2_1.csv`, …, `SDOF/D5.csv`).
The hourly records (18 000 samples, 3 min at 100 Hz) are appended in time order: row *r* belongs to
record *r div 18000* at *t = (r mod 18000) / 100* s. The sensor-fault scenario D4 is not included.

### 3D visualization of mode shapes

```bash
python src/kopru_3b.py
```

Reads the mode shape files (`2.Aşama/Hasarsız Modlar_son/…`, not included in this repository) and the
sensor positions from the Route 345 dataset, and writes `2.Aşama/bridge_modes_3d.html`.
A generated copy of this page is included in the repository and can be opened directly in a browser
(an internet connection is needed to load three.js).

## Repository structure

```
src/
├── rt345_tasnif.py          Route 345 classification
├── sdof_tasnif.py           SDOF classification
├── kopru_3b.py              builds the 3D visualization page
└── kopru_3b_sablon.html     page template (three.js)
2.Aşama/bridge_modes_3d.html generated 3D visualization
ENG401_Ara_Rapor_Bilgisayar_Muhendisligi.docx   interim report
requirements.txt
```

## Libraries

Python 3.12, NumPy, pandas, h5py (HDF5) and three.js (r160, loaded from a CDN).
