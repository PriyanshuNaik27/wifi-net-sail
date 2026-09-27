"""
app.py
======
IITG Wi-Fi Network Intelligence dashboard.

Run from the project root:
    streamlit run app.py

Loads the artifacts already trained and validated in models/ (see README.md
and SETUP.md) -- this file does NOT retrain anything. It scores the full
synthetic dataset once at startup (cached), then lets you scrub through the
TEST period (the model's held-out data -- the honest choice for a "live"
demo, since that's what the model has genuinely never seen) and drill into
individual locations, alerts, diagnoses, and a prioritized admin view.

All data is SYNTHETIC. Every page says so at least once; nothing here should
be read as a real measurement of IITG's network.
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from feature_engineering import build_features  # noqa: E402
from diagnosis import diagnose_row  # noqa: E402
from anomaly_detector import compute_rolling_zscore  # noqa: E402

DATA_PATH = "wifi_telemetry_synthetic.parquet"
LOCATIONS_PATH = "locations_synthetic.csv"
MODELS_DIR = "models"

ANOMALY_Z_THRESHOLD = 3.5  # matches the validated operating point in reports/anomaly_detector_thresholds.csv

st.set_page_config(page_title="IITG Wi-Fi Network Intelligence", layout="wide")


# ---------------------------------------------------------------------------
# Cached loading + scoring (runs once per server session, not per interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "degradation_model.joblib"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "categorical_encoder.joblib"))
    feature_cols = json.load(open(os.path.join(MODELS_DIR, "feature_columns.json")))
    threshold = json.load(open(os.path.join(MODELS_DIR, "threshold.json")))["threshold"]
    hist_lookup = joblib.load(os.path.join(MODELS_DIR, "historical_baseline_lookup.joblib"))
    diag_baselines = joblib.load(os.path.join(MODELS_DIR, "location_diagnosis_baselines.joblib"))
    return model, encoder, feature_cols, threshold, hist_lookup, diag_baselines


@st.cache_data
def load_scored_data():
    model, encoder, feature_cols, threshold, hist_lookup, _ = load_artifacts()

    df = pd.read_parquet(DATA_PATH).sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    df, _ = build_features(df, include_current_state_features=False)

    df_e = df.copy()
    df_e[["building_type", "semester_period"]] = encoder.transform(df_e[["building_type", "semester_period"]])
    X = df_e[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    df["gbm_score"] = model.predict_proba(X)[:, 1]
    df["gbm_pred"] = (df["gbm_score"] >= threshold).astype(int)

    df["hour_bucket"] = df["hour"].round(0) % 24
    hist_rate = hist_lookup["hist_rate"]
    global_mean = hist_lookup["global_mean"]
    idx = df.set_index(["location_id", "hour_bucket", "day_of_week"]).index
    df["hist_score"] = idx.map(hist_rate).fillna(global_mean).values
    df["hist_pred"] = (df["hist_score"] > 0.5).astype(int)

    df["anomaly_z"] = compute_rolling_zscore(df)
    df["is_anomaly_flagged"] = df["anomaly_z"] > ANOMALY_Z_THRESHOLD

    locations = pd.read_csv(LOCATIONS_PATH)
    return df, locations, threshold


df, locations, THRESHOLD = load_scored_data()
_, _, _, _, _, DIAG_BASELINES = load_artifacts()

# Restrict the "live" time-scrubbing UI to the TEST period only: this is the
# model's genuinely held-out data, so demoing on it is an honest "the model
# has never seen this" claim rather than replaying data it trained on.
TEST_START_DAY = 102
df["day_index"] = (df["timestamp"] - df["timestamp"].min()).dt.days
test_df = df[df["day_index"] >= TEST_START_DAY].copy()
test_dates = sorted(test_df["timestamp"].dt.date.unique())


def priority_category(gbm_pred, anomaly_flag, hist_pred):
    if gbm_pred or anomaly_flag:
        return "Investigate Now"
    if hist_pred:
        return "Monitor"
    return "Normal"


def get_diagnosis(row):
    return diagnose_row(row, DIAG_BASELINES)


# ---------------------------------------------------------------------------
# Shared sidebar: the "current simulated time" control used across pages
# ---------------------------------------------------------------------------

st.sidebar.title("IITG Wi-Fi Network Intelligence")
st.sidebar.caption("All data below is **synthetic** — a simulated campus network, not real IITG measurements.")

page = st.sidebar.radio(
    "View",
    ["Campus Health", "Live Alert Feed", "Location Deep Dive", "Diagnosis Comparison", "Admin Action List"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulated current time")
sel_date = st.sidebar.select_slider("Date", options=test_dates, value=test_dates[len(test_dates) // 2])
time_options = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 10, 20, 30, 40, 50)]
sel_time = st.sidebar.select_slider("Time of day", options=time_options, value="09:00")
sel_hour, sel_minute = map(int, sel_time.split(":"))
current_timestamp = pd.Timestamp(sel_date) + pd.Timedelta(hours=sel_hour, minutes=sel_minute)

current_rows = df[df["timestamp"] == current_timestamp]
if len(current_rows) == 0:
    st.sidebar.error("No data at this exact timestamp -- try nudging the slider.")
    st.stop()

st.sidebar.caption(f"Showing campus state at **{current_timestamp}**")


# ---------------------------------------------------------------------------
# PAGE 1 -- Campus Health
# ---------------------------------------------------------------------------

if page == "Campus Health":
    st.title("Campus Health")
    st.caption("Synthetic campus-wide snapshot at the selected simulated time.")

    cats = current_rows.apply(lambda r: priority_category(r["gbm_pred"], r["is_anomaly_flagged"], r["hist_pred"]), axis=1)
    current_rows = current_rows.assign(risk_category=cats)

    n_investigate = (current_rows["risk_category"] == "Investigate Now").sum()
    n_monitor = (current_rows["risk_category"] == "Monitor").sum()
    avg_quality = current_rows["network_quality_score"].mean()
    total_users = int(current_rows["connected_users"].sum())
    n_anomalies = int(current_rows["is_anomaly_flagged"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Locations to investigate", n_investigate)
    c2.metric("Locations to monitor", n_monitor)
    c3.metric("Avg network quality", f"{avg_quality:.0f} / 100")
    c4.metric("Connected users (campus)", f"{total_users:,}")
    c5.metric("Active anomalies", n_anomalies)

    st.markdown("#### Campus map (synthetic coordinates)")
    merged = current_rows  # 'building'/'building_type' are already native columns on df -- no merge needed
    color_map = {"Investigate Now": "#d62728", "Monitor": "#e6b800", "Normal": "#2ca02c"}
    fig = px.scatter(
        merged, x="longitude", y="latitude", color="risk_category",
        color_discrete_map=color_map, size="connected_users", size_max=28,
        hover_name="building", hover_data={"AP_load_pct": ":.0f", "network_quality_score": ":.0f",
                                            "longitude": False, "latitude": False},
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Highest-risk locations right now")
    top = current_rows.sort_values("gbm_score", ascending=False).head(8)
    st.dataframe(
        top[["building", "risk_category", "gbm_score", "AP_load_pct", "network_quality_score", "connected_users"]]
        .rename(columns={"gbm_score": "Risk score", "AP_load_pct": "AP load %", "network_quality_score": "Quality score", "connected_users": "Users"})
        .style.format({"Risk score": "{:.2f}", "AP load %": "{:.0f}", "Quality score": "{:.0f}"}),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# PAGE 2 -- Live Alert Feed
# ---------------------------------------------------------------------------

elif page == "Live Alert Feed":
    st.title("Live Alert Feed")
    st.caption("Locations the model flags as likely to sustain degradation within 60 minutes, with a root-cause read.")

    alerts = current_rows[current_rows["gbm_pred"] == 1].sort_values("gbm_score", ascending=False)

    if len(alerts) == 0:
        st.success("No active alerts at this simulated time. Try a different time of day (peak hours around 9-11am or 8-9pm tend to show more).")
    else:
        for _, row in alerts.iterrows():
            dx, conf, evidence = get_diagnosis(row)
            with st.container(border=True):
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.markdown(f"**{row['building']}**")
                    st.markdown(f"Risk score: **{row['gbm_score']:.2f}**  |  Calendar baseline said: {row['hist_score']:.2f}")
                    st.markdown(f"Estimated affected users: **{int(row['connected_users'])}**")
                    if row["is_anomaly_flagged"]:
                        st.markdown(f"⚠️ Anomaly flag active (z={row['anomaly_z']:.1f})")
                with col2:
                    st.markdown(f"**Diagnosis: {dx}** (confidence: {conf})")
                    st.caption(evidence)


# ---------------------------------------------------------------------------
# PAGE 3 -- Location Deep Dive
# ---------------------------------------------------------------------------

elif page == "Location Deep Dive":
    st.title("Location Deep Dive")

    loc_options = locations.copy()
    loc_options["label"] = loc_options.apply(
        lambda r: f"{r['building']} ({r['location_id']})" + (f" -- demo: {r['demo_tag']}" if isinstance(r["demo_tag"], str) and r["demo_tag"] else ""),
        axis=1,
    )
    sel_label = st.selectbox("Location", loc_options["label"].tolist())
    sel_loc_id = loc_options.loc[loc_options["label"] == sel_label, "location_id"].iloc[0]

    window = df[
        (df["location_id"] == sel_loc_id)
        & (df["timestamp"] >= current_timestamp - pd.Timedelta(hours=6))
        & (df["timestamp"] <= current_timestamp + pd.Timedelta(hours=6))
    ].sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["AP_load_pct"], name="AP load %", yaxis="y1"))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["latency_ms"], name="Latency (ms)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["gbm_score"] * 100, name="Model risk score (x100)", yaxis="y1", line=dict(dash="dot")))
    degraded_pts = window[window["is_degraded"]]
    fig.add_trace(go.Scatter(x=degraded_pts["timestamp"], y=degraded_pts["latency_ms"], mode="markers",
                              name="Actually degraded", marker=dict(color="red", size=8, symbol="x")))
    fig.add_vline(x=current_timestamp, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, xaxis_title="Time", yaxis_title="Value", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    current_loc_row = current_rows[current_rows["location_id"] == sel_loc_id]
    if len(current_loc_row) > 0:
        r = current_loc_row.iloc[0]
        dx, conf, evidence = get_diagnosis(r)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk score", f"{r['gbm_score']:.2f}")
        c2.metric("Quality score", f"{r['network_quality_score']:.0f}")
        c3.metric("SNR", f"{r['SNR_dB']:.1f} dB")
        c4.metric("Anomaly z-score", f"{r['anomaly_z']:.1f}" if pd.notna(r["anomaly_z"]) else "n/a")
        st.info(f"**{dx}** (confidence: {conf}) — {evidence}")


# ---------------------------------------------------------------------------
# PAGE 4 -- Diagnosis Comparison (same symptom, different cause)
# ---------------------------------------------------------------------------

elif page == "Diagnosis Comparison":
    st.title("Diagnosis Comparison")
    st.caption("Two locations, both showing degraded quality -- the diagnosis tells you whether it's the same problem.")

    cong_id = locations.loc[locations["demo_tag"].fillna("").str.contains("congestion_example"), "location_id"].iloc[0]
    infra_id = locations.loc[locations["demo_tag"].fillna("").str.contains("infrastructure_example"), "location_id"].iloc[0]

    # Confirmed via testing: this timestamp reliably produces a clean, high-confidence
    # CONGESTION diagnosis at the congestion-example location (an earlier pick, chosen
    # from the early-warning table for a different reason, landed on OTHER_UNCERTAIN
    # instead -- swapped out for this one after checking).
    cong_example_ts = pd.Timestamp("2026-01-25 22:10:00")
    infra_example_ts = pd.Timestamp("2026-02-28 12:20:00")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Congestion-example location")
        if st.button("Jump to a representative moment", key="jump_cong"):
            st.session_state["cong_override_ts"] = cong_example_ts
    with col2:
        st.subheader("Infrastructure-example location")
        if st.button("Jump to a representative moment", key="jump_infra"):
            st.session_state["infra_override_ts"] = infra_example_ts

    cong_display_ts = st.session_state.get("cong_override_ts", current_timestamp)
    infra_display_ts = st.session_state.get("infra_override_ts", current_timestamp)

    def show_location_card(loc_id, col, display_ts):
        row_df = df[(df["location_id"] == loc_id) & (df["timestamp"] == display_ts)]
        with col:
            st.caption(f"Showing: {display_ts}")
            if len(row_df) == 0:
                st.warning("No data at this exact timestamp for this location.")
                return
            r = row_df.iloc[0]
            dx, conf, evidence = get_diagnosis(r)
            st.metric("AP load", f"{r['AP_load_pct']:.0f}%")
            st.metric("Channel utilization", f"{r['channel_utilization_pct']:.0f}%")
            st.metric("SNR", f"{r['SNR_dB']:.1f} dB")
            st.metric("Quality score", f"{r['network_quality_score']:.0f}")
            st.info(f"**{dx}** (confidence: {conf})\n\n{evidence}")

    show_location_card(cong_id, col1, cong_display_ts)
    show_location_card(infra_id, col2, infra_display_ts)


# ---------------------------------------------------------------------------
# PAGE 5 -- Admin Action List
# ---------------------------------------------------------------------------

elif page == "Admin Action List":
    st.title("Admin Action List")

    with st.expander("How this ranking is calculated"):
        st.markdown(
            "- **Investigate Now**: the predictive model flags likely sustained degradation "
            "in the next 60 minutes (risk score ≥ threshold), OR the location's current reading "
            "is a statistical anomaly relative to its own history (z-score > 3.5).\n"
            "- **Monitor**: the calendar/historical baseline suggests elevated risk for this "
            "location at this hour, but the real-time model does not currently agree.\n"
            "- **Normal**: neither signal is elevated.\n\n"
            "This is a transparent rule over the two already-validated components (the GBM "
            "predictor and the anomaly detector) -- not a separate opaque score."
        )

    ranked = current_rows.copy()  # 'building'/'building_type' already native columns -- no merge needed
    ranked["risk_category"] = ranked.apply(
        lambda r: priority_category(r["gbm_pred"], r["is_anomaly_flagged"], r["hist_pred"]), axis=1
    )
    diagnoses = ranked.apply(get_diagnosis, axis=1)
    ranked["diagnosis"] = [d[0] for d in diagnoses]
    ranked["confidence"] = [d[1] for d in diagnoses]

    order = {"Investigate Now": 0, "Monitor": 1, "Normal": 2}
    ranked["_sort"] = ranked["risk_category"].map(order)
    ranked = ranked.sort_values(["_sort", "gbm_score"], ascending=[True, False])

    st.dataframe(
        ranked[["building", "building_type", "risk_category", "gbm_score", "diagnosis", "confidence", "connected_users"]]
        .rename(columns={"gbm_score": "Risk score", "connected_users": "Users"})
        .style.format({"Risk score": "{:.2f}"}),
        use_container_width=True, hide_index=True, height=600,
    )
