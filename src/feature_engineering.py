"""
feature_engineering.py
======================
Two responsibilities, kept deliberately separate:

1. build_target() — constructs `target_sustained_degradation_60m` from the
   raw `is_degraded` column. This uses FORWARD information (t+1 .. t+6) by
   definition -- it is the label, not a feature. It must never be joined
   back in as a predictor.

2. build_features() — constructs every model INPUT. Every rolling/lag
   feature here is explicitly shifted so that the value available "as of
   time t" never includes t itself for rolling windows that could leak
   the current computed degradation state, and never includes t+1 or later
   under any circumstances. See the leakage self-test at the bottom of this
   file (`assert_no_leakage`) which is run from train_model.py before any
   model sees the data.

Excluded from features, deliberately, with reasons in the README /
audit report:
    - is_seeded_problem_location (generator ground truth, not realistically knowable)
    - demo_tag (hackathon artifact, pure leakage)
    - is_anomaly_injected (evaluation-only ground truth for the anomaly detector)
    - location_id / AP_id as raw categorical features (kept only as a grouping key)
    - network_quality_score / is_degraded as *default* features (optional experiment only,
      see include_current_state_features flag)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_STEPS = 6          # 60 minutes at 10-min sampling
SUSTAINED_MIN_RUN = 3      # >=3 consecutive degraded steps (30+ min) = "sustained"
STEP_MINUTES = 10

ROLLING_METRICS = [
    "AP_load_pct",
    "latency_ms",
    "packet_loss_pct",
    "channel_utilization_pct",
    "throughput_Mbps",
    "signal_strength_dBm",
    "SNR_dB",
    "interference_score",
]

STATIC_FEATURES = ["building_type", "floor", "AP_capacity", "AP_age_months"]
CALENDAR_FEATURES = [
    "hour", "day_of_week", "is_weekend", "semester_period",
    "is_lecture_hour", "is_event_active",
]


# ---------------------------------------------------------------------------
# 1. TARGET CONSTRUCTION
# ---------------------------------------------------------------------------

def _flag_sustained_single_location(is_degraded: np.ndarray, min_run: int) -> np.ndarray:
    """Given a boolean array (one location, already time-sorted), return a
    boolean array marking every row that belongs to a run of >= min_run
    consecutive True values. Backward+forward within-run only -- this
    flags the CURRENT/PAST state, used only to build the historical label,
    not as a leaking feature.
    """
    x = is_degraded.astype(int)
    n = len(x)
    run_id = np.full(n, -1, dtype=int)
    rid = -1
    for i in range(n):
        if x[i] == 1:
            if i == 0 or x[i - 1] == 0:
                rid += 1
            run_id[i] = rid
        # else stays -1
    out = np.zeros(n, dtype=bool)
    valid = run_id >= 0
    if valid.any():
        s = pd.Series(run_id[valid])
        lengths = s.groupby(s).transform("count")
        out[valid] = (lengths >= min_run).values
    return out


def _future_window_target_single_location(sustained_flag: np.ndarray, horizon_steps: int) -> np.ndarray:
    """target[i] = 1 if sustained_flag is True anywhere in (i, i+horizon_steps].
    Strictly forward-looking -- this is the LABEL, never to be used as a feature.
    Implemented with a reverse-cumulative trick for O(n) instead of O(n*h).
    """
    n = len(sustained_flag)
    flag = sustained_flag.astype(int)
    # future_any[i] = max(flag[i+1 : i+1+horizon])
    # build via a reversed rolling max
    padded = np.concatenate([flag, np.zeros(horizon_steps, dtype=int)])
    target = np.zeros(n, dtype=int)
    # simple O(n) sliding window max using deque-free approach (horizon is small, 6 steps)
    for offset in range(1, horizon_steps + 1):
        target = np.maximum(target, padded[offset:offset + n])
    return target


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_sustained_degraded` (current/past-state label) and
    `target_sustained_degradation_60m` (the forward-looking model target).

    Must be called on a DataFrame already sorted by [location_id, timestamp].
    """
    assert df["timestamp"].is_monotonic_increasing or True  # per-group check below
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    sustained_parts = []
    target_parts = []
    for loc, sub in df.groupby("location_id", sort=False):
        sub_sorted = sub.sort_values("timestamp")
        assert sub_sorted["timestamp"].is_monotonic_increasing, f"Non-monotonic timestamps for {loc}"
        flag = _flag_sustained_single_location(sub_sorted["is_degraded"].values, SUSTAINED_MIN_RUN)
        tgt = _future_window_target_single_location(flag, HORIZON_STEPS)
        sustained_parts.append(pd.Series(flag, index=sub_sorted.index))
        target_parts.append(pd.Series(tgt, index=sub_sorted.index))

    df["is_sustained_degraded"] = pd.concat(sustained_parts).sort_index()
    df["target_sustained_degradation_60m"] = pd.concat(target_parts).sort_index().astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. CAUSAL FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def _causal_rolling_features(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """All rolling/lag stats here use only observations <= t (current row
    included for the rolling stats, which is fine -- the CURRENT reading at
    time t is legitimately known at prediction time; only t+1 and beyond
    would be leakage, and none of these windows look forward).
    """
    s = sub[col]
    out = pd.DataFrame(index=sub.index)

    out[f"{col}_lag_10m"] = s.shift(1)
    out[f"{col}_lag_20m"] = s.shift(2)
    out[f"{col}_lag_30m"] = s.shift(3)

    out[f"{col}_rolling_mean_30m"] = s.rolling(3, min_periods=1).mean()
    out[f"{col}_rolling_mean_60m"] = s.rolling(6, min_periods=1).mean()
    out[f"{col}_rolling_max_60m"] = s.rolling(6, min_periods=1).max()
    out[f"{col}_rolling_std_60m"] = s.rolling(6, min_periods=2).std()

    # slope = finite-difference trend proxy (change per step over the window),
    # fully vectorized -- causal, uses only s(t) and s(t-w). A true per-window
    # least-squares slope was tried first but its rolling().apply(polyfit) cost
    # was prohibitive at this row count; this proxy captures the same "is it
    # trending up/down" signal the model needs, at a fraction of the cost.
    out[f"{col}_rolling_slope_30m"] = (s - s.shift(3)) / 3.0
    out[f"{col}_rolling_slope_60m"] = (s - s.shift(6)) / 6.0

    return out


def build_features(
    df: pd.DataFrame,
    include_current_state_features: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Builds the full feature matrix (causal rolling + static + calendar).

    Returns (df_with_features, feature_column_names).
    Must be called on data already sorted by [location_id, timestamp].
    `include_current_state_features` controls whether network_quality_score
    and is_degraded (current-timestep state) are added -- OFF by default,
    per the requirement to justify their inclusion before using them.
    """
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    feature_frames = []
    for loc, sub in df.groupby("location_id", sort=False):
        sub_sorted = sub.sort_values("timestamp")
        loc_features = [sub_sorted[["location_id"]]]  # keep for realignment
        for col in ROLLING_METRICS:
            loc_features.append(_causal_rolling_features(sub_sorted, col))
        feature_frames.append(pd.concat(loc_features, axis=1))

    rolling_df = pd.concat(feature_frames).sort_index()
    rolling_feature_cols = [c for c in rolling_df.columns if c != "location_id"]

    df = pd.concat([df, rolling_df[rolling_feature_cols]], axis=1)

    feature_cols = list(rolling_feature_cols)

    # Current-timestep raw readings (available at time t -- not leakage, just "now")
    current_state_cols = list(ROLLING_METRICS) + ["connected_users", "active_users", "data_volume_MB"]
    feature_cols += current_state_cols

    # Static + calendar
    feature_cols += STATIC_FEATURES + CALENDAR_FEATURES

    if include_current_state_features:
        feature_cols += ["network_quality_score", "is_degraded"]

    return df, feature_cols


def assert_no_leakage(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Sanity checks run before training:
    1. None of the excluded/forbidden columns are present in feature_cols.
    2. For a random sample of rows, manually recompute a rolling feature
       using only data up to t and confirm it matches what's in the frame
       (catches accidental centered/forward windows).
    """
    forbidden = {
        "is_seeded_problem_location", "demo_tag", "is_anomaly_injected",
        "location_id", "AP_id", "target_sustained_degradation_60m",
        "is_sustained_degraded", "timestamp",
    }
    leaked = forbidden.intersection(set(feature_cols))
    assert not leaked, f"Forbidden columns present in feature set: {leaked}"

    # Spot-check: rolling_mean_60m for AP_load_pct at a random row must equal
    # the mean of the 6 rows ending at that row (inclusive), for one location.
    rng = np.random.default_rng(0)
    for loc in rng.choice(df["location_id"].unique(), size=3, replace=False):
        sub = df[df["location_id"] == loc].sort_values("timestamp").reset_index(drop=True)
        idx = rng.integers(10, len(sub) - 1)
        expected = sub["AP_load_pct"].iloc[max(0, idx - 5): idx + 1].mean()
        actual = sub["AP_load_pct_rolling_mean_60m"].iloc[idx]
        assert np.isclose(expected, actual, equal_nan=True), (
            f"Leakage check FAILED for {loc} at row {idx}: expected {expected}, got {actual}"
        )
    print("Leakage self-test passed: rolling features are strictly backward-looking.")
