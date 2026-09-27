"""
anomaly_detector.py
====================
Per-location, causal (trailing-window only) z-score anomaly detector.

Deliberately NOT a global-threshold detector: what counts as "abnormal" is
defined relative to each location's own recent history, so a naturally busy
hostel and a naturally quiet admin building aren't held to the same bar.

`is_anomaly_injected` is used ONLY to evaluate this detector after the fact.
It is never an input to the detector itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

ROLLING_WINDOW_STEPS = 144  # 24 hours at 10-min sampling
MIN_PERIODS = 36            # require at least 6 hours of history before flagging
METRICS_FOR_ANOMALY = ["latency_ms", "packet_loss_pct"]


def compute_rolling_zscore(df: pd.DataFrame, metrics: list[str] = None) -> pd.Series:
    """Returns a single combined z-score per row: the max absolute z-score
    across `metrics`, each computed against that location's own trailing
    (shifted, i.e. causal) mean/std. Must be called on data sorted by
    [location_id, timestamp]; safe to call on unsorted data too (it sorts
    internally per group and returns results aligned to the input index).
    """
    metrics = metrics or METRICS_FOR_ANOMALY
    df = df.sort_values(["location_id", "timestamp"])

    z_frames = []
    for loc, sub in df.groupby("location_id", sort=False):
        z_cols = {}
        for m in metrics:
            roll_mean = sub[m].rolling(ROLLING_WINDOW_STEPS, min_periods=MIN_PERIODS).mean().shift(1)
            roll_std = sub[m].rolling(ROLLING_WINDOW_STEPS, min_periods=MIN_PERIODS).std().shift(1)
            z_cols[m] = ((sub[m] - roll_mean) / roll_std.replace(0, np.nan)).abs()
        z_frames.append(pd.DataFrame(z_cols, index=sub.index))

    z_all = pd.concat(z_frames).sort_index()
    combined = z_all.max(axis=1)
    return combined


def evaluate_thresholds(df: pd.DataFrame, thresholds=(2.5, 3.0, 3.5, 4.0)) -> pd.DataFrame:
    z = compute_rolling_zscore(df)
    usable = z.notna()
    y = df.loc[usable, "is_anomaly_injected"].astype(int)
    z_usable = z[usable]

    rows = []
    for t in thresholds:
        pred = (z_usable > t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "flagged_pct": round(pred.mean() * 100, 2),
            "n_rows_usable": int(usable.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_parquet("../wifi_telemetry_synthetic.parquet")  # path adjusted for local tree: data file at project root
    result = evaluate_thresholds(df)
    print(result.to_string(index=False))
    result.to_csv("../reports/anomaly_detector_thresholds.csv", index=False)
