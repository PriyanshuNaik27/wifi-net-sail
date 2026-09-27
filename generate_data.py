"""
generate_data.py
================
Synthetic Wi-Fi / network telemetry generator for the IITG Network
Intelligence hackathon project.

IMPORTANT: All data produced by this script is SYNTHETIC. It is designed to
represent realistic *patterns* of campus Wi-Fi behaviour, not real IITG
measurements. Nothing produced here should be presented as an actual
measurement of IITG's network.

Design principle
-----------------
This generator does NOT draw columns independently at random. Each metric is
built from a causal chain so that relationships in the data mean something:

    building_type + hour + day_of_week + semester_period + event_active
            -> connected_users            (demand)
            -> AP_load_pct                (load)
            -> channel_utilization_pct    (contention)
            -> packet_loss / latency / jitter   (congestion effects)
            -> throughput_Mbps            (experienced quality, inverse of load/loss)

Independently of that chain (this is what makes root-cause diagnosis
possible at all):
    - signal_strength_dBm has a *per-location* baseline. ~12% of locations
      are seeded as "structurally poor" (old AP / bad placement) regardless
      of how busy they are.
    - ~1.5% of (location, timestamp) rows get an injected anomaly: a sudden
      deviation unrelated to the smooth causal chain above. These are the
      ground truth for anomaly-detection evaluation.

Four locations are deliberately tagged for the hackathon demo (see
DEMO_TAGS below) so the live demo doesn't depend on getting lucky with a
random timestamp.

Usage
-----
    python generate_data.py --n-locations 40 --days 120 --freq-minutes 10 \
        --seed 42 --out-dir .

All key parameters are configurable via CLI flags (see `parse_args`).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

BUILDING_TYPES = [
    "hostel",
    "academic",
    "lecture_hall",
    "library",
    "lab",
    "admin",
    "cafeteria",
    "sports",
    "outdoor_common",
]

# Roughly how common each building type is on campus (used to sample locations)
BUILDING_TYPE_WEIGHTS = {
    "hostel": 0.22,
    "academic": 0.16,
    "lecture_hall": 0.14,
    "library": 0.05,
    "lab": 0.14,
    "admin": 0.06,
    "cafeteria": 0.08,
    "sports": 0.06,
    "outdoor_common": 0.09,
}

# Fraction of locations seeded as structurally poor (bad signal/old AP,
# independent of load) -- this is what lets the diagnosis component tell
# "infrastructure problem" apart from "congestion".
SEEDED_PROBLEM_LOCATION_FRACTION = 0.12

# Fraction of (location, timestamp) rows that receive an injected anomaly.
ANOMALY_INJECTION_RATE = 0.015

# Demo scenario tags -- assigned to specific locations after generation so
# the dashboard/demo can reliably pull "the lecture hall that fills up",
# "the congestion example", "the infrastructure example", and
# "the persistent problem location" without hunting through random data.
DEMO_TAGS = {
    "lecture_fill": None,        # assigned to a lecture_hall location
    "congestion_example": None,  # assigned to a high-load, decent-signal location
    "infrastructure_example": None,  # assigned to a seeded problem location, low load
    "persistent_problem": None,  # assigned to a seeded problem location (can overlap above)
}


@dataclass
class BuildingProfile:
    """Defines the demand curve and typical scale for a building type."""

    peak_hours_weekday: list  # hours (0-23) of peak demand on a weekday
    peak_hours_weekend: list
    base_users_range: tuple  # (min, max) typical peak connected users per AP
    weekend_scale: float     # multiplier applied to weekday pattern on weekends
    exam_period_scale: float  # multiplier during exam periods
    ap_density_factor: float  # how much nearby-AP contention typically exists (0-1)


BUILDING_PROFILES = {
    "hostel": BuildingProfile(
        peak_hours_weekday=[7, 8, 20, 21, 22, 23],
        peak_hours_weekend=[10, 11, 20, 21, 22, 23, 0],
        base_users_range=(25, 70),
        weekend_scale=1.15,
        exam_period_scale=1.3,
        ap_density_factor=0.6,
    ),
    "academic": BuildingProfile(
        peak_hours_weekday=[9, 10, 11, 14, 15, 16],
        peak_hours_weekend=[10, 11],
        base_users_range=(15, 45),
        weekend_scale=0.25,
        exam_period_scale=0.7,
        ap_density_factor=0.45,
    ),
    "lecture_hall": BuildingProfile(
        peak_hours_weekday=[9, 10, 11, 14, 15, 16],
        peak_hours_weekend=[],
        base_users_range=(40, 140),
        weekend_scale=0.05,
        exam_period_scale=1.1,  # exams held in lecture halls
        ap_density_factor=0.7,
    ),
    "library": BuildingProfile(
        peak_hours_weekday=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        peak_hours_weekend=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        base_users_range=(30, 90),
        weekend_scale=0.8,
        exam_period_scale=1.8,  # library surges heavily during exams
        ap_density_factor=0.55,
    ),
    "lab": BuildingProfile(
        peak_hours_weekday=[10, 11, 14, 15, 16, 17],
        peak_hours_weekend=[],
        base_users_range=(10, 35),
        weekend_scale=0.1,
        exam_period_scale=0.6,
        ap_density_factor=0.3,
    ),
    "admin": BuildingProfile(
        peak_hours_weekday=[10, 11, 12, 15, 16],
        peak_hours_weekend=[],
        base_users_range=(8, 25),
        weekend_scale=0.05,
        exam_period_scale=0.9,
        ap_density_factor=0.25,
    ),
    "cafeteria": BuildingProfile(
        peak_hours_weekday=[8, 9, 13, 14, 19, 20, 21],
        peak_hours_weekend=[9, 10, 13, 14, 19, 20],
        base_users_range=(20, 60),
        weekend_scale=1.0,
        exam_period_scale=1.0,
        ap_density_factor=0.5,
    ),
    "sports": BuildingProfile(
        peak_hours_weekday=[6, 7, 17, 18, 19],
        peak_hours_weekend=[7, 8, 16, 17, 18],
        base_users_range=(5, 25),
        weekend_scale=1.2,
        exam_period_scale=0.6,
        ap_density_factor=0.2,
    ),
    "outdoor_common": BuildingProfile(
        peak_hours_weekday=[12, 13, 17, 18, 19, 20],
        peak_hours_weekend=[11, 12, 17, 18, 19],
        base_users_range=(10, 40),
        weekend_scale=1.1,
        exam_period_scale=0.8,
        ap_density_factor=0.35,
    ),
}


# ---------------------------------------------------------------------------
# Location generation
# ---------------------------------------------------------------------------

def generate_locations(n_locations: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate location + AP metadata, including seeded problem locations
    and demo-scenario tags.
    """
    types = list(BUILDING_TYPE_WEIGHTS.keys())
    weights = np.array(list(BUILDING_TYPE_WEIGHTS.values()))
    weights = weights / weights.sum()

    building_types = rng.choice(types, size=n_locations, p=weights)

    # Fake campus grid (IITG-style layout, purely illustrative coordinates)
    base_lat, base_lon = 26.1928, 91.6919  # rough IITG campus centroid
    lats = base_lat + rng.uniform(-0.01, 0.01, n_locations)
    lons = base_lon + rng.uniform(-0.012, 0.012, n_locations)

    ap_capacity = rng.integers(60, 150, n_locations)  # rated max concurrent devices
    ap_age_months = rng.integers(2, 90, n_locations)

    n_seeded = max(1, int(round(SEEDED_PROBLEM_LOCATION_FRACTION * n_locations)))
    seeded_idx = rng.choice(n_locations, size=n_seeded, replace=False)
    is_seeded_problem = np.zeros(n_locations, dtype=bool)
    is_seeded_problem[seeded_idx] = True

    # Signal strength baseline: normal locations cluster around -55 dBm (good),
    # seeded problem locations cluster around -78 dBm (poor), independent of load.
    signal_baseline = np.where(
        is_seeded_problem,
        rng.normal(-78, 4, n_locations),
        rng.normal(-55, 5, n_locations),
    )
    signal_baseline = np.clip(signal_baseline, -95, -35)

    floors = rng.integers(0, 6, n_locations)

    location_ids = [f"LOC_{i:03d}" for i in range(n_locations)]
    ap_ids = [f"AP_{i:03d}" for i in range(n_locations)]
    buildings = [
        f"{bt.replace('_', ' ').title()} {chr(65 + (i % 4))}"
        for i, bt in enumerate(building_types)
    ]

    df = pd.DataFrame(
        {
            "location_id": location_ids,
            "AP_id": ap_ids,
            "building": buildings,
            "building_type": building_types,
            "floor": floors,
            "latitude": lats,
            "longitude": lons,
            "AP_capacity": ap_capacity,
            "AP_age_months": ap_age_months,
            "signal_strength_baseline_dBm": signal_baseline,
            "is_seeded_problem_location": is_seeded_problem,
        }
    )

    # --- Assign demo tags deliberately (not left to chance) ---
    demo_tag = pd.Series([""] * n_locations, index=df.index, dtype=object)

    lecture_halls = df.index[df["building_type"] == "lecture_hall"].tolist()
    if lecture_halls:
        demo_tag.iloc[lecture_halls[0]] = "lecture_fill"

    # Congestion example: NOT a seeded-problem location, high capacity building type
    congestion_candidates = df.index[
        (~df["is_seeded_problem_location"])
        & (df["building_type"].isin(["library", "hostel"]))
    ].tolist()
    if congestion_candidates:
        demo_tag.iloc[congestion_candidates[0]] = "congestion_example"

    # Infrastructure example: a seeded problem location with typically LOW load
    infra_candidates = df.index[
        (df["is_seeded_problem_location"])
        & (df["building_type"].isin(["admin", "lab", "outdoor_common"]))
    ].tolist()
    if not infra_candidates:
        infra_candidates = df.index[df["is_seeded_problem_location"]].tolist()
    if infra_candidates:
        demo_tag.iloc[infra_candidates[0]] = "infrastructure_example"
        # Reuse the same location for "persistent problem" (it genuinely is one)
        demo_tag.iloc[infra_candidates[0]] = demo_tag.iloc[infra_candidates[0]] + "|persistent_problem"

    df["demo_tag"] = demo_tag
    return df


# ---------------------------------------------------------------------------
# Time index / calendar features
# ---------------------------------------------------------------------------

def build_time_index(start_date: str, days: int, freq_minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=start_date, periods=int(days * 24 * 60 / freq_minutes), freq=f"{freq_minutes}min"
    )
    hour = timestamps.hour + timestamps.minute / 60.0
    dow = timestamps.dayofweek  # 0=Mon
    is_weekend = dow >= 5

    day_index = (timestamps - timestamps[0]).days
    total_days = day_index.max() + 1

    # Semester structure: SEVERAL shorter exam-like blocks and a single mid-sem
    # break, distributed across the whole timeline -- not one block at the end.
    # This matters for evaluation: a model/baseline trained only on a single
    # trailing exam block would never see exam-period behaviour in TRAIN, and
    # a model evaluated on a single trailing block can't be checked for
    # exam-regime generalization in VAL either. Day targets below are tuned
    # for the default 120-day run (two mid-term-style blocks early/mid, one
    # exam block landing in each of the approved VAL and TEST windows); they
    # scale proportionally if --days changes so the same relative spread
    # holds for a shorter or longer run.
    scale = total_days / 120.0

    def scaled_range(start_day, end_day):
        s = int(round(start_day * scale))
        e = int(round(end_day * scale))
        return s, max(e, s + 1)

    exam_block_targets = [(27, 31), (70, 74), (90, 94), (112, 116)]
    is_exam = np.zeros(len(day_index), dtype=bool)
    for start_day, end_day in exam_block_targets:
        s, e = scaled_range(start_day, end_day)
        is_exam |= (day_index >= s) & (day_index < e)

    break_s, break_e = scaled_range(58, 63)
    is_break = (day_index >= break_s) & (day_index < break_e) & (~is_exam)

    semester_period = np.where(is_exam, "exam", np.where(is_break, "break", "regular"))

    is_lecture_hour = (~is_weekend) & (hour >= 9) & (hour <= 17) & (~is_break)

    # Campus events (fest/seminar/sports-meet style single-day load boosts),
    # spread across the timeline rather than one recurring festival, so every
    # split (TRAIN/VAL/TEST) sees at least a couple of event days. Kept clear
    # of the exam blocks above (an event during exams would confound two
    # separate causal effects in the same days).
    event_day_targets = [10, 36, 54, 88, 97, 108, 116]
    event_days = set()
    for target in event_day_targets:
        d = int(round(target * scale))
        event_days.add(d)
    is_event = np.isin(day_index, list(event_days)) & (~is_exam)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "semester_period": semester_period,
            "is_lecture_hour": is_lecture_hour,
            "is_event_active": is_event,
        }
    )


# ---------------------------------------------------------------------------
# Per-location time series generation (the causal chain)
# ---------------------------------------------------------------------------

def _demand_multiplier(profile: BuildingProfile, hour: np.ndarray, is_weekend: np.ndarray) -> np.ndarray:
    """Smooth demand curve built from Gaussian bumps around each peak hour."""
    mult = np.full(hour.shape, 0.15)  # baseline off-peak floor
    for h in profile.peak_hours_weekday:
        bump = np.exp(-0.5 * ((hour - h) / 1.1) ** 2)
        mult = mult + np.where(~is_weekend, bump, 0.0)
    for h in profile.peak_hours_weekend:
        bump = np.exp(-0.5 * ((hour - h) / 1.3) ** 2)
        mult = mult + np.where(is_weekend, bump * profile.weekend_scale, 0.0)
    if not profile.peak_hours_weekend:
        mult = mult + np.where(is_weekend, 0.0, 0.0)  # explicit no-op, kept for clarity
        mult = np.where(is_weekend, mult * profile.weekend_scale * 0.3, mult)
    return mult


def generate_location_timeseries(
    loc: pd.Series,
    time_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    profile = BUILDING_PROFILES[loc["building_type"]]
    n = len(time_df)
    hour = time_df["hour"].values
    is_weekend = time_df["is_weekend"].values
    is_exam = (time_df["semester_period"] == "exam").values
    is_break = (time_df["semester_period"] == "break").values
    is_event = time_df["is_event_active"].values

    # ---- Demand -> connected_users ----
    demand = _demand_multiplier(profile, hour, is_weekend)
    exam_boost = np.where(is_exam, profile.exam_period_scale, 1.0)
    break_damp = np.where(is_break, 0.3, 1.0)
    event_boost = np.where(is_event, 1.6, 1.0)

    lo, hi = profile.base_users_range
    scale = rng.uniform(lo, hi)
    noise = rng.normal(1.0, 0.08, n)
    connected_users = scale * demand * exam_boost * break_damp * event_boost * noise
    connected_users = np.clip(connected_users, 1, None)

    active_users = connected_users * rng.uniform(0.55, 0.85, n)
    data_volume_MB = active_users * rng.uniform(3.0, 9.0, n) * (1 + 0.3 * np.sin(hour / 24 * 2 * np.pi))
    data_volume_MB = np.clip(data_volume_MB, 0, None)

    # ---- Load ----
    ap_capacity = loc["AP_capacity"]
    AP_load_pct = np.clip((connected_users / ap_capacity) * 100 * rng.normal(1.0, 0.05, n), 0, 130)

    # ---- Channel utilization: load + typical nearby-AP contention for this building type ----
    channel_utilization_pct = np.clip(
        AP_load_pct * (0.55 + 0.35 * profile.ap_density_factor) + rng.normal(0, 4, n),
        0, 100,
    )

    # ---- Signal strength: per-location baseline, near-independent of load ----
    signal_strength_dBm = loc["signal_strength_baseline_dBm"] + rng.normal(0, 2.5, n)

    # Older APs add a small amount of extra interference on top of location baseline
    age_interference = np.clip((loc["AP_age_months"] - 24) / 60, 0, 1) * 10

    interference_score = np.clip(
        0.5 * channel_utilization_pct / 100 * 100
        + age_interference
        + rng.normal(0, 5, n),
        0, 100,
    )

    noise_floor = -95 + interference_score * 0.15
    SNR_dB = np.clip(signal_strength_dBm - noise_floor, 0, 60)

    # ---- Congestion effects, PLUS a load-independent weak-signal effect ----
    # Weak signal forces a lower PHY rate, which drives up retransmissions and
    # effective latency/packet-loss even when very few users are connected.
    # This is what makes "infrastructure problem" distinguishable from
    # "congestion" at low load, rather than only showing up in throughput.
    snr_deficit = np.clip(20 - SNR_dB, 0, None)  # 0 when signal is healthy (SNR >= 20dB)

    packet_loss_pct = np.clip(
        0.02 * channel_utilization_pct
        + 0.15 * interference_score
        + 0.9 * snr_deficit
        + rng.normal(0, 1.2, n),
        0, 40,
    )
    latency_ms = np.clip(
        15
        + 0.9 * AP_load_pct
        + 0.5 * channel_utilization_pct
        + 1.8 * packet_loss_pct
        + 2.5 * snr_deficit
        + rng.normal(0, 4, n),
        5, 500,
    )
    jitter_ms = np.clip(0.12 * latency_ms + rng.normal(0, 2, n), 0.5, 100)

    # ---- Experienced throughput: inverse of load/loss, driven by signal quality ----
    signal_quality = np.clip((SNR_dB / 40), 0.05, 1.2)
    throughput_Mbps = np.clip(
        120 * signal_quality * (1 - AP_load_pct / 140) * (1 - packet_loss_pct / 60)
        + rng.normal(0, 3, n),
        0.5, 300,
    )

    out = pd.DataFrame(
        {
            "location_id": loc["location_id"],
            "AP_id": loc["AP_id"],
            "building": loc["building"],
            "building_type": loc["building_type"],
            "floor": loc["floor"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "AP_capacity": ap_capacity,
            "AP_age_months": loc["AP_age_months"],
            "is_seeded_problem_location": loc["is_seeded_problem_location"],
            "demo_tag": loc["demo_tag"],
            "timestamp": time_df["timestamp"].values,
            "hour": time_df["hour"].values,
            "day_of_week": time_df["day_of_week"].values,
            "is_weekend": time_df["is_weekend"].values,
            "semester_period": time_df["semester_period"].values,
            "is_lecture_hour": time_df["is_lecture_hour"].values,
            "is_event_active": time_df["is_event_active"].values,
            "connected_users": connected_users.round(0),
            "active_users": active_users.round(0),
            "data_volume_MB": data_volume_MB.round(2),
            "signal_strength_dBm": signal_strength_dBm.round(1),
            "SNR_dB": SNR_dB.round(1),
            "channel_utilization_pct": channel_utilization_pct.round(1),
            "interference_score": interference_score.round(1),
            "packet_loss_pct": packet_loss_pct.round(2),
            "latency_ms": latency_ms.round(1),
            "jitter_ms": jitter_ms.round(1),
            "throughput_Mbps": throughput_Mbps.round(1),
            "AP_load_pct": AP_load_pct.round(1),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Anomaly injection (ground truth for anomaly detection evaluation)
# ---------------------------------------------------------------------------

def inject_anomalies(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["is_anomaly_injected"] = False

    n_rows = len(df)
    n_anomalies = int(round(n_rows * rate))
    anomaly_idx = rng.choice(df.index.values, size=n_anomalies, replace=False)

    kinds = rng.choice(
        ["overload", "signal_collapse", "interference_spike"], size=n_anomalies
    )

    for idx, kind in zip(anomaly_idx, kinds):
        if kind == "overload":
            df.loc[idx, "connected_users"] *= rng.uniform(2.2, 3.5)
            df.loc[idx, "AP_load_pct"] = np.clip(df.loc[idx, "AP_load_pct"] * rng.uniform(2.0, 2.8), 0, 200)
            df.loc[idx, "latency_ms"] *= rng.uniform(2.5, 4.0)
            df.loc[idx, "packet_loss_pct"] = np.clip(df.loc[idx, "packet_loss_pct"] * rng.uniform(3, 5), 0, 80)
            df.loc[idx, "throughput_Mbps"] *= rng.uniform(0.2, 0.4)
        elif kind == "signal_collapse":
            df.loc[idx, "signal_strength_dBm"] -= rng.uniform(20, 30)
            df.loc[idx, "SNR_dB"] = np.clip(df.loc[idx, "SNR_dB"] - rng.uniform(15, 25), 0, 60)
            df.loc[idx, "throughput_Mbps"] *= rng.uniform(0.1, 0.3)
            df.loc[idx, "packet_loss_pct"] = np.clip(df.loc[idx, "packet_loss_pct"] + rng.uniform(10, 25), 0, 80)
        else:  # interference_spike
            df.loc[idx, "interference_score"] = np.clip(df.loc[idx, "interference_score"] + rng.uniform(35, 55), 0, 100)
            df.loc[idx, "packet_loss_pct"] = np.clip(df.loc[idx, "packet_loss_pct"] + rng.uniform(8, 20), 0, 80)
            df.loc[idx, "jitter_ms"] *= rng.uniform(2.5, 4)
            df.loc[idx, "latency_ms"] *= rng.uniform(1.8, 2.6)

        df.loc[idx, "is_anomaly_injected"] = True

    return df


# ---------------------------------------------------------------------------
# Derived ground-truth labels (documented, not hidden)
# ---------------------------------------------------------------------------

# Quality-score weights, chosen so no single term dominates. Values are
# normalized against rough "acceptable vs unacceptable" thresholds before
# weighting -- see README for the cited thresholds this leans on.
QUALITY_WEIGHTS = dict(latency=0.28, packet_loss=0.27, jitter=0.15, throughput=0.20, snr=0.10)


def compute_quality_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def norm(series, good, bad, invert=False):
        # good/bad define the scale endpoints; result in [0, 1], higher = worse
        # unless invert=True (for throughput/SNR where higher raw value = better)
        x = (series - good) / (bad - good)
        x = x.clip(0, 1)
        return (1 - x) if invert else x

    latency_pen = norm(df["latency_ms"], good=20, bad=250)
    loss_pen = norm(df["packet_loss_pct"], good=0.5, bad=15)
    jitter_pen = norm(df["jitter_ms"], good=2, bad=40)
    throughput_pen = norm(df["throughput_Mbps"], good=80, bad=5, invert=True)
    snr_pen = norm(df["SNR_dB"], good=35, bad=8, invert=True)

    penalty = (
        QUALITY_WEIGHTS["latency"] * latency_pen
        + QUALITY_WEIGHTS["packet_loss"] * loss_pen
        + QUALITY_WEIGHTS["jitter"] * jitter_pen
        + QUALITY_WEIGHTS["throughput"] * throughput_pen
        + QUALITY_WEIGHTS["snr"] * snr_pen
    )
    df["network_quality_score"] = (100 * (1 - penalty)).round(1)

    # Degradation threshold: quality_score below 55 is where the underlying
    # metrics cross commonly-cited "poor experience" territory (latency
    # approaching 150ms+, packet loss above ~3-5%, throughput collapsing).
    df["is_degraded"] = df["network_quality_score"] < 55

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_dataset(
    n_locations: int,
    start_date: str,
    days: int,
    freq_minutes: int,
    anomaly_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    locations = generate_locations(n_locations, rng)
    time_df = build_time_index(start_date, days, freq_minutes)

    frames = []
    for _, loc in locations.iterrows():
        frames.append(generate_location_timeseries(loc, time_df, rng))
    telemetry = pd.concat(frames, ignore_index=True)

    telemetry = inject_anomalies(telemetry, anomaly_rate, rng)
    telemetry = compute_quality_and_labels(telemetry)

    telemetry = telemetry.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    return telemetry, locations


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic IITG Wi-Fi telemetry.")
    p.add_argument("--n-locations", type=int, default=40)
    p.add_argument("--start-date", type=str, default="2026-01-05")
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--freq-minutes", type=int, default=10)
    p.add_argument("--anomaly-rate", type=float, default=ANOMALY_INJECTION_RATE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=".")  # path adjusted for local tree: this script now lives at project root, output goes alongside it
    p.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet",
                   help="Output format. Parquet is much smaller/faster to load in the dashboard; "
                        "use csv only if you need to eyeball the file directly.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    telemetry, locations = generate_dataset(
        n_locations=args.n_locations,
        start_date=args.start_date,
        days=args.days,
        freq_minutes=args.freq_minutes,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )

    ext = "parquet" if args.format == "parquet" else "csv"
    telemetry_path = os.path.join(args.out_dir, f"wifi_telemetry_synthetic.{ext}")
    locations_path = os.path.join(args.out_dir, "locations_synthetic.csv")  # always small, keep as csv

    if args.format == "parquet":
        telemetry.to_parquet(telemetry_path, index=False)
    else:
        telemetry.to_csv(telemetry_path, index=False)
    locations.to_csv(locations_path, index=False)

    print(f"Generated {len(telemetry):,} telemetry rows across {args.n_locations} locations "
          f"over {args.days} days at {args.freq_minutes}-min intervals.")
    print(f"Injected anomalies: {telemetry['is_anomaly_injected'].sum():,} "
          f"({telemetry['is_anomaly_injected'].mean()*100:.2f}%)")
    print(f"Seeded problem locations: {locations['is_seeded_problem_location'].sum()} / {len(locations)}")
    print(f"Demo-tagged locations:\n{locations[locations['demo_tag'] != ''][['location_id','building','building_type','demo_tag']]}")
    print(f"Saved telemetry to: {telemetry_path}")
    print(f"Saved location metadata to: {locations_path}")


if __name__ == "__main__":
    main()
