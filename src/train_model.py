"""
train_model.py
===============
Builds the modeling dataset, trains all baselines + the GBM, selects a
decision threshold on VALIDATION ONLY, then evaluates once on TEST.

Run from the src/ directory:
    python train_model.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

from feature_engineering import build_target, build_features, assert_no_leakage

DATA_PATH = "../wifi_telemetry_synthetic.parquet"  # path adjusted for local tree: data file lives at project root, not data/raw/
MODELS_DIR = "../models"
REPORTS_DIR = "../reports"

TRAIN_END_DAY = 84   # Jan 5 - Mar 29
VAL_END_DAY = 102    # Mar 30 - Apr 16
                     # test: Apr 17 - May 4
BOUNDARY_BUFFER_STEPS = 6  # matches horizon; drop rows whose target window crosses a split boundary

CATEGORICAL_FEATURES = ["building_type", "semester_period"]


CACHE_PATH = "../_features_cache.parquet"  # path adjusted for local tree: same root level as the data file above


def load_and_prepare(use_cache: bool = True) -> tuple[pd.DataFrame, list[str]]:
    if use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading cached feature frame from {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
    else:
        df = pd.read_parquet(DATA_PATH)
        df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
        df = build_target(df)
        df, _ = build_features(df, include_current_state_features=False)
        df.to_parquet(CACHE_PATH)

    # Recompute feature_cols the same way build_features derives them, without
    # rebuilding the (expensive) rolling columns themselves.
    from feature_engineering import ROLLING_METRICS, STATIC_FEATURES, CALENDAR_FEATURES
    rolling_feature_cols = [c for c in df.columns if any(c.startswith(m + "_") for m in ROLLING_METRICS)]
    current_state_cols = list(ROLLING_METRICS) + ["connected_users", "active_users", "data_volume_MB"]
    feature_cols = rolling_feature_cols + current_state_cols + STATIC_FEATURES + CALENDAR_FEATURES

    assert_no_leakage(df, feature_cols)
    df["day_index"] = (df["timestamp"] - df["timestamp"].min()).dt.days
    return df, feature_cols


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split with a buffer at each boundary so that no row's
    target window (which looks up to 6 steps / 60 min into the future)
    reaches across into the next split. Concretely: drop the last
    BOUNDARY_BUFFER_STEPS rows of each location within TRAIN and VAL before
    the cutoff, so their forward-looking target label never touches data
    from the following split.
    """
    train_mask = df["day_index"] < TRAIN_END_DAY
    val_mask = (df["day_index"] >= TRAIN_END_DAY) & (df["day_index"] < VAL_END_DAY)
    test_mask = df["day_index"] >= VAL_END_DAY

    def drop_boundary_tail(sub_df: pd.DataFrame) -> pd.DataFrame:
        # drop the last BOUNDARY_BUFFER_STEPS rows per location (chronological tail)
        keep_idx = []
        for _, g in sub_df.groupby("location_id", sort=False):
            g_sorted = g.sort_values("timestamp")
            keep_idx.append(g_sorted.index[:-BOUNDARY_BUFFER_STEPS] if len(g_sorted) > BOUNDARY_BUFFER_STEPS else g_sorted.index)
        return sub_df.loc[np.concatenate([np.asarray(i) for i in keep_idx])]

    train = drop_boundary_tail(df[train_mask])
    val = drop_boundary_tail(df[val_mask])
    test = df[test_mask]  # no following split, no buffer needed at the far end

    return train, val, test


def metrics_row(name: str, y_true, y_pred, y_score=None) -> dict:
    row = {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_score is not None:
        row["roc_auc"] = round(roc_auc_score(y_true, y_score), 4)
        row["pr_auc"] = round(average_precision_score(y_true, y_score), 4)
    return row


def run_baselines(train, val, test) -> tuple[list[dict], pd.Series, pd.Series, dict]:
    """Returns (metrics_rows, val_hist_score, test_hist_score, hist_lookup)."""
    rows = []
    y_val = val["target_sustained_degradation_60m"]
    y_test = test["target_sustained_degradation_60m"]

    # 1. Persistence
    rows.append(metrics_row("1_persistence (VAL)", y_val, val["is_sustained_degraded"].astype(int)))
    rows.append(metrics_row("1_persistence (TEST)", y_test, test["is_sustained_degraded"].astype(int)))

    # 2. Fixed threshold
    rows.append(metrics_row("2_fixed_threshold_load75 (VAL)", y_val, (val["AP_load_pct"] > 75).astype(int)))
    rows.append(metrics_row("2_fixed_threshold_load75 (TEST)", y_test, (test["AP_load_pct"] > 75).astype(int)))

    # 3. Historical location x hour x weekday baseline (fit on TRAIN only)
    train_h = train.copy()
    val_h = val.copy()
    test_h = test.copy()
    for d in (train_h, val_h, test_h):
        d["hour_bucket"] = d["hour"].round(0) % 24
    hist_rate = train_h.groupby(["location_id", "hour_bucket", "day_of_week"])["target_sustained_degradation_60m"].mean()
    global_mean = train_h["target_sustained_degradation_60m"].mean()

    val_score = val_h.set_index(["location_id", "hour_bucket", "day_of_week"]).index.map(hist_rate).fillna(global_mean)
    test_score = test_h.set_index(["location_id", "hour_bucket", "day_of_week"]).index.map(hist_rate).fillna(global_mean)
    val_score = pd.Series(val_score, index=val.index)
    test_score = pd.Series(test_score, index=test.index)

    rows.append(metrics_row("3_historical_baseline (VAL)", y_val, (val_score > 0.5).astype(int), val_score))
    rows.append(metrics_row("3_historical_baseline (TEST)", y_test, (test_score > 0.5).astype(int), test_score))

    return rows, val_score, test_score, {"hist_rate": hist_rate, "global_mean": global_mean}


def encode_categoricals(train, val, test, feature_cols):
    """Ordinal-encode categorical features (fit on TRAIN only), leave numeric/bool as-is."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train[cat_cols])

    def transform(d):
        d = d.copy()
        d[cat_cols] = encoder.transform(d[cat_cols])
        return d

    return transform(train), transform(val), transform(test), encoder


def prep_xy(d, feature_cols):
    X = d[feature_cols].copy()
    # bool -> int, everything else left as numeric (NaNs from lag warm-up allowed;
    # HistGradientBoostingClassifier and XGBoost both handle NaN natively)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    y = d["target_sustained_degradation_60m"]
    return X, y


def run_logistic_regression(train, val, test, feature_cols):
    # LR can't handle NaN or raw categoricals -- use a reduced, complete-case feature set
    simple_features = [
        "AP_load_pct", "channel_utilization_pct", "SNR_dB", "connected_users",
        "packet_loss_pct", "latency_ms", "jitter_ms", "interference_score",
    ]
    simple_features = [f for f in simple_features if f in feature_cols or f in train.columns]

    Xtr, ytr = train[simple_features], train["target_sustained_degradation_60m"]
    Xval, ytest = val[simple_features], test[simple_features]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    val_score = clf.predict_proba(val[simple_features])[:, 1]
    test_score = clf.predict_proba(test[simple_features])[:, 1]

    rows = [
        metrics_row("4_logistic_regression (VAL)", val["target_sustained_degradation_60m"], (val_score > 0.5).astype(int), val_score),
        metrics_row("4_logistic_regression (TEST)", test["target_sustained_degradation_60m"], (test_score > 0.5).astype(int), test_score),
    ]
    return rows, val_score, test_score


def train_gbm(Xtr, ytr, Xval, yval):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            scale_pos_weight=(ytr == 0).sum() / max(1, (ytr == 1).sum()),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        used = "xgboost"
    except Exception as e:
        print(f"XGBoost unavailable/failed ({e}), falling back to HistGradientBoostingClassifier")
        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=42,
        )
        model.fit(Xtr, ytr)
        used = "hist_gbm"
    return model, used


def threshold_table(y_true, y_score) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.1, 0.91, 0.05):
        pred = (y_score >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": round(precision_score(y_true, pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, pred, zero_division=0), 4),
            "flagged_pct": round(pred.mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading data, building target + features...")
    df, feature_cols = load_and_prepare()
    print(f"Feature count: {len(feature_cols)}")

    train, val, test = time_based_split(df)
    print(f"TRAIN: {train['timestamp'].min()} to {train['timestamp'].max()} ({len(train)} rows)")
    print(f"VAL:   {val['timestamp'].min()} to {val['timestamp'].max()} ({len(val)} rows)")
    print(f"TEST:  {test['timestamp'].min()} to {test['timestamp'].max()} ({len(test)} rows)")

    all_metrics = []

    baseline_rows, val_hist_score, test_hist_score, hist_lookup = run_baselines(train, val, test)
    all_metrics += baseline_rows

    lr_rows, val_lr_score, test_lr_score = run_logistic_regression(train, val, test, feature_cols)
    all_metrics += lr_rows

    print("\nEncoding categoricals + preparing GBM matrices...")
    train_e, val_e, test_e, encoder = encode_categoricals(train, val, test, feature_cols)
    Xtr, ytr = prep_xy(train_e, feature_cols)
    Xval, yval = prep_xy(val_e, feature_cols)
    Xtest, ytest = prep_xy(test_e, feature_cols)

    print("Training GBM...")
    model, model_kind = train_gbm(Xtr, ytr, Xval, yval)

    val_gbm_score = model.predict_proba(Xval)[:, 1]
    test_gbm_score = model.predict_proba(Xtest)[:, 1]

    all_metrics.append(metrics_row(f"5_gbm_{model_kind} (VAL, default 0.5)", yval, (val_gbm_score >= 0.5).astype(int), val_gbm_score))
    all_metrics.append(metrics_row(f"5_gbm_{model_kind} (TEST, default 0.5)", ytest, (test_gbm_score >= 0.5).astype(int), test_gbm_score))

    metrics_df = pd.DataFrame(all_metrics)
    print("\n" + metrics_df.to_string(index=False))
    metrics_df.to_json(os.path.join(REPORTS_DIR, "model_metrics.json"), orient="records", indent=2)

    # --- Threshold selection on VALIDATION only ---
    thresh_df = threshold_table(yval, val_gbm_score)
    print("\nThreshold table (VALIDATION):\n", thresh_df.to_string(index=False))
    thresh_df.to_csv(os.path.join(REPORTS_DIR, "threshold_table_validation.csv"), index=False)

    # Operational criterion: maximize recall (missing a real incident costs more
    # than one extra false alert) SUBJECT TO precision staying at or above 0.75.
    # Rationale: an admin who gets a false alarm 1 in 4 times will still trust
    # the feed; a false alarm 1 in 3 times (the precision=0.5 bar we tried first,
    # which selected threshold=0.10 and flagged 30% of all rows) trains them to
    # ignore it. 0.75 is a defensible, stated line -- not the only reasonable
    # one, but a documented one, per the assignment's requirement.
    candidates = thresh_df[thresh_df["precision"] >= 0.75]
    if len(candidates) > 0:
        selected = candidates.sort_values("recall", ascending=False).iloc[0]
    else:
        selected = thresh_df.sort_values("f1", ascending=False).iloc[0]
    selected_threshold = float(selected["threshold"])
    print(f"\nSelected threshold (from VALIDATION only): {selected_threshold}")

    # --- Freeze threshold, evaluate ONCE on TEST ---
    test_pred_frozen = (test_gbm_score >= selected_threshold).astype(int)
    frozen_row = metrics_row(f"5_gbm_{model_kind} (TEST, frozen threshold={selected_threshold})", ytest, test_pred_frozen, test_gbm_score)
    print("Frozen-threshold TEST result:", frozen_row)

    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "r") as f:
        existing = json.load(f)
    existing.append(frozen_row)
    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(existing, f, indent=2)

    # --- Save artifacts ---
    joblib.dump(model, os.path.join(MODELS_DIR, "degradation_model.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "categorical_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": selected_threshold, "criterion": "max recall subject to precision>=0.75 on VAL"}, f, indent=2)

    # Save the historical lookup table too (needed for the baseline comparison in evaluate_model.py)
    joblib.dump(hist_lookup, os.path.join(MODELS_DIR, "historical_baseline_lookup.joblib"))

    # Persist the assembled (features + target) splits for evaluate_model.py to reuse
    train.assign(gbm_score=model.predict_proba(Xtr)[:, 1]).to_parquet(os.path.join(REPORTS_DIR, "_train_scored.parquet"))
    val.assign(gbm_score=val_gbm_score, hist_score=val_hist_score.values, lr_score=val_lr_score).to_parquet(os.path.join(REPORTS_DIR, "_val_scored.parquet"))
    test.assign(gbm_score=test_gbm_score, hist_score=test_hist_score.values, lr_score=test_lr_score).to_parquet(os.path.join(REPORTS_DIR, "_test_scored.parquet"))

    print("\nSaved model artifacts to", MODELS_DIR)
    print("Saved metrics/threshold table to", REPORTS_DIR)
    print("Model type used:", model_kind)


if __name__ == "__main__":
    main()
