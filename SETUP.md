# SETUP — Running Locally in VS Code (Windows)

This project was built and validated in a Linux container with the package
versions pinned in `requirements.txt`. These steps reproduce that
environment on Windows as closely as possible.

## 1. Confirm your folder matches this layout

```
IITG-WiFi-Hackathon/
├── generate_data.py
├── locations_synthetic.csv
├── wifi_telemetry_synthetic.parquet
├── requirements.txt
├── README.md
├── SETUP.md
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── anomaly_detector.py
│   └── diagnosis.py
├── models/
│   ├── categorical_encoder.joblib
│   ├── degradation_model.joblib
│   ├── feature_columns.json
│   ├── historical_baseline_lookup.joblib
│   ├── location_diagnosis_baselines.joblib
│   └── threshold.json
└── reports/
    ├── anomaly_detector_thresholds.csv
    ├── congestion_example_early_warning_table.csv
    ├── diagnosis_distribution.csv
    ├── feature_importance.csv
    ├── feature_importance_grouped.csv
    ├── lead_time_metrics.json
    ├── model_metrics.json
    ├── subgroup_metrics.csv
    └── threshold_table_validation.csv
```

**Note on paths:** the four scripts in `src/` were originally written
assuming the data files lived under a `data/raw/` subfolder. Since your
tree keeps them at the project root instead, the path constants in
`train_model.py`, `anomaly_detector.py`, and `diagnosis.py` (and
`generate_data.py`'s `--out-dir` default) were adjusted to point one
level up from `src/` to the root — e.g.
`DATA_PATH = "../wifi_telemetry_synthetic.parquet"`. Nothing else in
any file was changed: no target definition, no features, no model
parameters, no thresholds. If you `Ctrl+F` for `../` in any of the
four scripts, every match is one of these path constants.

## 2. Open the folder in VS Code

`File > Open Folder...` → select `C:\Users\Priyanshu Naik\Desktop\IITG-WiFi-Hackathon`.

## 3. Create and activate a virtual environment

In VS Code's integrated terminal (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution-policy error,
run this once (in an admin PowerShell, or just for your user):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry activation. You should see `(.venv)` appear at the start of
your prompt. In VS Code, also select this interpreter: `Ctrl+Shift+P` →
"Python: Select Interpreter" → the one inside `.venv`.

## 4. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**If `pandas==3.0.2` fails to install** (pandas 3.x is very recent and a
Windows wheel may not be available for your Python version yet), install
without the pin and note the installed version instead:

```powershell
pip install pandas numpy scikit-learn xgboost joblib pyarrow
pip freeze > requirements.installed.txt
```

The pipeline doesn't rely on anything pandas-3-specific as far as this
project's code goes, but if you hit a behavior difference, that's the
first place to look.

## 5. Verify the existing artifacts load correctly (no retraining)

From the project root:

```powershell
cd src
python -c "import joblib, json; m = joblib.load('../models/degradation_model.joblib'); print(type(m).__name__); print(json.load(open('../models/threshold.json')))"
```

Expected output:
```
XGBClassifier
{'threshold': 0.3, 'criterion': 'max recall subject to precision>=0.75 on VAL'}
```

This confirms the model file, and the path from `src/` to `models/`,
are both intact — without touching the dataset or retraining anything.

## 6. (Optional) Re-run the full pipeline to reproduce results

Only do this if you want to confirm the metrics reproduce end-to-end.
It will overwrite `models/` and `reports/` with freshly computed
(but identical, since the seed is fixed) versions.

```powershell
cd src
python train_model.py
```

This takes under a minute on typical hardware once XGBoost is
installed. It will print the same baseline/GBM metrics documented in
`README.md`.

```powershell
python anomaly_detector.py
python diagnosis.py
```

## 7. Common Windows gotchas

- **`ModuleNotFoundError: No module named 'xgboost'`** — the venv isn't
  activated, or `pip install -r requirements.txt` didn't complete. Check
  `(.venv)` is showing in your prompt.
- **Long path errors** — Windows has a 260-character path limit by
  default; this project's paths are short enough to not hit it, but if
  you nest it deeper, enable long paths via `git config --system
  core.longpaths true` or the Windows registry setting.
- **Parquet read errors** — almost always means `pyarrow` didn't
  install; re-run `pip install pyarrow`.

## 8. Run the dashboard

From the project root (not `src/`):

```powershell
pip install streamlit plotly
streamlit run app.py
```

This opens in your browser automatically (usually `http://localhost:8501`).
It loads the already-trained artifacts from `models/` and scores the full
dataset once at startup (~10-15 seconds the first time; cached after that
via Streamlit's `@st.cache_data`/`@st.cache_resource`, so switching pages
or moving the time slider afterward is instant). It does **not** retrain
anything.

Five views, selectable from the sidebar:
- **Campus Health** — campus-wide snapshot and map at the selected simulated time
- **Live Alert Feed** — currently-flagged locations with root-cause diagnosis
- **Location Deep Dive** — pick any location, see its metrics over a ±6h window
- **Diagnosis Comparison** — the congestion-example vs. infrastructure-example
  locations side by side, each with a "jump to a representative moment" button
- **Admin Action List** — all 40 locations ranked into Investigate Now / Monitor / Normal

The sidebar's date/time control only lets you scrub through the **TEST**
period (Apr 17 - May 4) — the model's genuinely held-out data, so what
you're seeing really is "the model has never seen this," not a replay of
its own training data.

## What NOT to do yet

Per the plan: don't regenerate the dataset (`generate_data.py`), and
don't retrain unless you're just verifying reproducibility.
