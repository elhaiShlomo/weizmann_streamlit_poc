import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import pyvista as pv

def build_electrode_intervals(df_elec: pd.DataFrame):
    """
    Build activation intervals for each electrode using spike_value as start and next timestamp as end.
    Only rows with spike_value > 0.0 start an interval.
    """
    intervals = {}
    for elec, sub in df_elec.groupby("electrode"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        elec_intervals = []
        for i in range(len(sub) - 1):
            spike = float(sub.loc[i, "spike_value"])
            next_ts = float(sub.loc[i + 1, "timestamp"])
            if spike > 0.0:
                elec_intervals.append((spike, next_ts))
        intervals[elec] = elec_intervals
    return intervals


def is_active(electrode: str, t: float, intervals: dict) -> bool:
    """Return True if time t is inside any activation interval for electrode."""
    for start, end in intervals.get(electrode, []):
        if start <= t < end:
            return True
    return False


def filter_intervals_by_window(intervals: dict, tmin: float, tmax: float) -> dict:
    """Keep only activation intervals that overlap the global time window."""
    filtered = {}
    for elec, spans in intervals.items():
        kept = []
        for start, end in spans:
            if end >= tmin and start <= tmax:
                kept.append((max(start, tmin), min(end, tmax)))
        filtered[elec] = kept
    return filtered


def build_time_frames(tmin: float, tmax: float, fps: int = 20, max_frames: int = 150, min_step: float | None = None):
    """
    Build time points across the current window for playback.
    - If min_step is given, sample at that resolution (capped by max_frames).
    - Else, derive frame count from fps.
    """
    if tmax <= tmin:
        return [tmin]
    if min_step is not None and min_step > 0:
        total_frames = min(max_frames, max(2, int((tmax - tmin) / min_step) + 1))
    else:
        total_frames = min(max_frames, max(2, int((tmax - tmin) * fps) + 1))
    return list(np.linspace(tmin, tmax, total_frames))

def assign_region(electrode_name: str) -> str:
    """
    Assign brain region based on electrode naming patterns.
    Works with names like 'LA1', 'RA2', 'LMH1', 'RMH2', 'LT1', 'LAM1', 'LPFC1', etc.
    """
    if not isinstance(electrode_name, str) or electrode_name.strip() == "":
        return "Other"

    e = electrode_name.upper().split('-')[0]  # Use only first part (before "-")

    # --- Hippocampus ---
    # Covers LA, RA, LMH, RMH, PH, AH, MH, HPC
    if e.startswith(("LA", "RA", "LMH", "RMH", "AH", "PH", "MH", "HPC")):
        return "HPC"

    # --- Amygdala ---
    elif e.startswith(("LAM", "RAM", "AMY", "AMYG")):
        return "AMYG"

    # --- Temporal Cortex ---
    elif e.startswith(("LT", "RT", "TEMP", "MT")):
        return "Temporal"

    # --- Prefrontal Cortex ---
    elif e.startswith(("LPFC", "RPFC", "FP", "PFC", "LPF", "RPF")):
        return "PFC"

    # --- Unknown / other ---
    else:
        return "Other"

REGION_COLORS = {
    "HPC": "#00FFFF",
    "PFC": "#FFD700",
    "Temporal": "#ADFF2F",
    "AMYG": "#FF69B4",
    "Other": "#AAAAAA",
}
FOCUS_COLOR = "#FF0000"

st.set_page_config(page_title="Avg Density Line Chart", layout="wide")
st.title("Weizmann Demo v2")

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_data():
    df_main = pd.read_csv("samples/merged_band_data.csv")
    df_events = pd.read_csv("samples/clinical_events_new.csv")
    df_electrodes = pd.read_csv("samples/all_patients_unified.csv")
    return df_main, df_events, df_electrodes

try:
    df, df_events, df_electrodes = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ✅ assign brain region to each electrode
df_electrodes["region"] = df_electrodes["electrode"].apply(assign_region)

# --- Validation ---
required_cols = {"time_s", "avg_density"}
if not required_cols.issubset(df.columns):
    st.error(f"Missing columns in merged_band_data.csv: {required_cols - set(df.columns)}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Controls")

# =============================
# 🧍 Group 1: Patient Information
# =============================
st.sidebar.markdown("### 🧠 Patient Information")

if "patient_id" in df_events.columns:
    available_patients = sorted(df_events["patient_id"].dropna().unique())
    selected_patient = st.sidebar.selectbox("Select Patient ID:", available_patients)
    df_events = df_events[df_events["patient_id"] == selected_patient]
    if "patient_id" in df_electrodes.columns:
        df_electrodes = df_electrodes[df_electrodes["patient_id"] == selected_patient]

available_admissions = sorted(df_events["admission_id"].dropna().unique())
selected_admission = st.sidebar.selectbox("Select Admission ID:", available_admissions)
df_events = df_events[df_events["admission_id"] == selected_admission]

st.sidebar.markdown("<hr style='border:1px solid gray; margin:10px 0;'>", unsafe_allow_html=True)

# =============================
# ⚙️ Group 2:
# =============================
st.sidebar.markdown("### ⚙️ Group 2")

if "band" in df.columns:
    available_bands = sorted(df["band"].dropna().unique())
    selected_bands = st.sidebar.multiselect(
        "Select EEG Bands:",
        available_bands,
        default=available_bands
    )
    df = df[df["band"].isin(selected_bands)]

# --- Clinical Events Filter ---
if "event_name" in df_events.columns:
    st.sidebar.markdown("### Clinical Events Filter")

    available_events = sorted(df_events["event_name"].dropna().unique())
    selected_events = st.sidebar.multiselect(
        "Select Clinical Events:",
        available_events,
        default=available_events
    )

    df_events = df_events[df_events["event_name"].isin(selected_events)]

# --- Electrodes Filter ---
if "electrode" in df_electrodes.columns:

    st.sidebar.markdown("### Electrodes Filter")
    available_electrodes = sorted(df_electrodes["electrode"].dropna().unique())
    select_all = st.sidebar.checkbox("Select all electrodes", key="select_all_elec", value=True)

    if select_all:
        selected_electrodes = st.sidebar.multiselect(
            "Select Electrodes:",
            options=available_electrodes,
            default=available_electrodes,
            key="elec_multiselect",
        )
    else:
        selected_electrodes = st.sidebar.multiselect(
            "Select Electrodes:",
            options=available_electrodes,
            key="elec_multiselect",
        )

    # Apply filter
    df_electrodes = df_electrodes[df_electrodes["electrode"].isin(selected_electrodes)]


# --- Brain Region Filter ---
if "region" in df_electrodes.columns:
    st.sidebar.markdown("### 🧩 Brain Region Filter")

    available_regions = sorted(df_electrodes["region"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "Select Brain Regions:",
        available_regions,
        default=available_regions
    )

    df_electrodes = df_electrodes[df_electrodes["region"].isin(selected_regions)]

tmin, tmax = st.sidebar.slider(
    "Select Time Range (seconds):",
    float(df["time_s"].min()),
    float(df["time_s"].max()),
    (float(df["time_s"].min()), float(df["time_s"].max()))
)

st.sidebar.markdown("### 📊 Data Processing & Smoothing")
apply_smoothing = st.sidebar.checkbox("Apply Data Smoothing", value=False)

smoothing_method = st.sidebar.selectbox(
    "Smoothing Method:",
    ["Moving Average"],
    index=0
)

if apply_smoothing:
    window_seconds = st.sidebar.slider(
        "Smoothing Window (seconds):",
        min_value=0.5,
        max_value=30.0,
        value=5.0,
        step=0.5
    )

    # Ensure data sorted by time
    df = df.sort_values("time_s").copy()

    # Compute sampling interval (average Δt)
    dt = df["time_s"].diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 0.1  # fallback default if irregular sampling
    window_size = max(3, int(window_seconds / dt))

    # Apply smoothing per band if exists
    if "band" in df.columns:
        smoothed_frames = []
        for band_name, sub_df in df.groupby("band"):
            temp = sub_df.copy()
            temp["avg_density"] = (
                temp["avg_density"]
                .fillna(method="ffill")
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            smoothed_frames.append(temp)
        df = pd.concat(smoothed_frames).sort_values(["band", "time_s"])
    else:
        df["avg_density"] = (
            df["avg_density"]
            .fillna(method="ffill")
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

    st.sidebar.caption(f"Δt ≈ {dt:.3f}s → window = {window_size} samples")

# --- Filtering by time ---
filtered_df = df[(df["time_s"] >= tmin) & (df["time_s"] <= tmax)]
filtered_events = df_events[(df_events["event_time"] >= tmin) & (df_events["event_time"] <= tmax)]
filtered_elec = df_electrodes[(df_electrodes["timestamp"] >= tmin) & (df_electrodes["timestamp"] <= tmax)]
electrode_intervals = build_electrode_intervals(df_electrodes)
window_intervals = filter_intervals_by_window(electrode_intervals, tmin, tmax)
# Show all selected electrodes on the brain (even if no activity in the window)
electrodes = sorted(df_electrodes["electrode"].dropna().unique())

# --- Group events (unchanged) ---
grouped_events = (
    filtered_events.groupby("occurrence_id")
    .agg({
        "admission_id": "first",
        "category_name": "first",
        "event_name": "first",
        "event_time": "first",
        "param_name": list,
        "value_name": list
    })
    .reset_index()
)

# ==============================
# 🔹 Create subplot with spacing
# ==============================
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,  # ✅ more spacing between charts
    row_heights=[0.55, 0.45],
    subplot_titles=(f"EEG Avg Density (Admission {selected_admission})", "Electrode Activity")
)

# ==============================
# 🧠 Row 1: Original EEG chart (UNTOUCHED)
# ==============================
if "band" in filtered_df.columns:
    for band_name, group in filtered_df.groupby("band"):
        fig.add_trace(go.Scatter(
            x=group["time_s"],
            y=group["avg_density"],
            mode="lines",
            name=str(band_name),
            line=dict(width=2)
        ), row=1, col=1)
else:
    fig.add_trace(go.Scatter(
        x=filtered_df["time_s"],
        y=filtered_df["avg_density"],
        mode="lines",
        name="Avg Density",
        line=dict(width=2)
    ), row=1, col=1)

# --- Events (unchanged logic) ---
if not grouped_events.empty:
    emoji_map = {
        "Seizure": "⚡", "Medication": "💊", "Medications": "💊",
        "Sleep": "😴", "Awake": "☀️", "Stimulus": "🔔",
        "Clinical": "🩺", "Test": "🧪", "EEG": "🧠", "Unknown": "❓"
    }
    severity_colors = {"Mild": "lime", "Moderate": "gold", "Severe": "red"}
    y_max = filtered_df["avg_density"].max()
    event_names = sorted(grouped_events["event_name"].dropna().unique())
    offsets = {name: y_max * (1.05 + i * 0.05) for i, name in enumerate(event_names)}
    last_dose_by_drug = {}

    for event_name, group in grouped_events.groupby("event_name"):
        emoji = next((v for k, v in emoji_map.items() if k.lower() in event_name.lower()), "📍")
        y_value = offsets.get(event_name, y_max * 1.05)
        xs, ys, texts, hovers = [], [], [], []
        for _, row in group.iterrows():
            params = dict(zip(row["param_name"], row["value_name"]))
            text_lines = [emoji]
            hover_extra = ""
            trend_symbol = ""

            if "DrugName" in row["param_name"] or "Dose_mg" in row["param_name"]:
                drug_name = params.get("DrugName", "")
                dose_value = None
                try:
                    dose_value = float(str(params.get("Dose_mg", "")).replace("mg", "").strip())
                except Exception:
                    pass
                if dose_value is not None and drug_name:
                    if drug_name in last_dose_by_drug:
                        prev_dose = last_dose_by_drug[drug_name]
                        if dose_value > prev_dose:
                            trend_symbol = "🔼"
                        elif dose_value < prev_dose:
                            trend_symbol = "🔽"
                        else:
                            trend_symbol = "⏺"
                    last_dose_by_drug[drug_name] = dose_value
                    text_lines += [drug_name, f"{int(dose_value)}mg {trend_symbol}"]
                    hover_extra += f"<br>Drug: {drug_name}<br>Dose: {dose_value}mg<br>Trend: {trend_symbol}"

            if "Seizure" in event_name:
                severity = params.get("Severity") or params.get("severity") or ""
                color = severity_colors.get(severity, "white")
                text_lines += [f"<span style='color:{color}'>{severity}</span>"]
                hover_extra += f"<br>Severity: <b style='color:{color}'>{severity}</b>"

            xs.append(row["event_time"])
            ys.append(y_value)
            texts.append("<br>".join(text_lines))
            hovers.append(
                f"<b>{emoji} {event_name}</b><br>Category: {row['category_name']}<br>"
                + "<br>".join([f"{p}: {v}" for p, v in params.items()])
                + f"{hover_extra}<br>Time: {row['event_time']:.2f}s"
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="text", text=texts,
            textfont=dict(size=14, color="white"),
            name=f"{emoji} {event_name}",
            hovertemplate="%{text}<extra></extra>",
            hovertext=hovers,
            showlegend=True
        ), row=1, col=1)

        for x in xs:
            fig.add_trace(go.Scatter(
                x=[x, x],
                y=[y_value, filtered_df["avg_density"].min()],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ), row=1, col=1)

# ==============================
# ⚡ Row 2: Electrode Activity (region colors + red focus)
# ==============================
if electrodes:
    for electrode in electrodes:
        sub = df_electrodes[df_electrodes["electrode"] == electrode].sort_values("timestamp")
        region = sub["region"].iloc[0] if "region" in sub.columns and not sub.empty else "Other"
        base_color = REGION_COLORS.get(region, REGION_COLORS["Other"])

        for start_time, end_time in window_intervals.get(electrode, []):
            focus_hit = False
            if "Focus (Spatial)" in sub.columns:
                focus_rows = sub[
                    (sub["timestamp"] >= start_time)
                    & (sub["timestamp"] <= end_time)
                    & (sub["Focus (Spatial)"] == electrode)
                ]
                focus_hit = not focus_rows.empty

            color = FOCUS_COLOR if focus_hit else base_color

            fig.add_trace(go.Scatter(
                x=[start_time, end_time],
                y=[electrode, electrode],
                mode="lines",
                line=dict(color=color, width=5),
                opacity=1.0,
                showlegend=False,
                hoverinfo="x+text",
                text=f"{electrode} ({region})"
            ), row=2, col=1)

# ==============================
# 🎨 Layout
# ==============================
fig.update_layout(
    template="plotly_dark",
    height=1650,  # slightly taller
    margin=dict(l=40, r=20, t=60, b=120),  # ✅ add bottom space for slider overlap
    hovermode="x unified",
    legend_title="EEG Bands"
)

# ✅ keep range slider only for top chart
fig.update_xaxes(range=[tmin, tmax], rangeslider=dict(visible=True, thickness=0.08), row=1, col=1)
fig.update_xaxes(range=[tmin, tmax], rangeslider=dict(visible=False), row=2, col=1)

# ✅ more spacing between subplots
fig.layout.update({"grid": {"rows": 2, "columns": 1, "pattern": "independent"}})

# ✅ keep range slider only for top chart
fig.update_xaxes(range=[tmin, tmax], rangeslider=dict(visible=True, thickness=0.08), row=1, col=1)
fig.update_xaxes(range=[tmin, tmax], rangeslider=dict(visible=False), row=2, col=1)

# ✅ show x tick labels also for the top chart
fig.update_xaxes(showticklabels=True, row=1, col=1)

# y-axis titles
fig.update_yaxes(title="Avg Density", row=1, col=1)
fig.update_yaxes(
    title="Electrodes",
    row=2, col=1,
    categoryorder="array", categoryarray=electrodes,
    automargin=True
)

# --- Display ---
st.plotly_chart(fig, use_container_width=True)

# ==============================
# 🎨 Custom Legend for Electrode Regions
# ==============================
st.markdown("""
<div style='text-align:center; font-size:16px; margin-top:10px;'>
    <b>Brain Region Colors:</b><br>
    <span style='color:#00FFFF;'>■</span> HPC &nbsp;&nbsp;
    <span style='color:#FFD700;'>■</span> PFC &nbsp;&nbsp;
    <span style='color:#ADFF2F;'>■</span> Temporal &nbsp;&nbsp;
    <span style='color:#FF69B4;'>■</span> AMYG &nbsp;&nbsp;
    <span style='color:#AAAAAA;'>■</span> Other &nbsp;&nbsp;
    <span style='color:#FF0000;'>■</span> Focus (Selected Electrode)
</div>
""", unsafe_allow_html=True)


# ==============================
# 🧠 3D Human Brain Visualization (REAL MRI BRAIN)
# ==============================

st.title("? Electrode Activity (Linear Layout)")

# Playback speed control
playback_fps = st.sidebar.slider("Playback speed (fps)", min_value=1, max_value=60, value=10, step=1)
frame_ms = max(10, int(1000 / playback_fps))

# Build frames at fine step so short activations are captured; playback speed only affects timing
time_points = build_time_frames(tmin, tmax, max_frames=400, min_step=0.05)

def frame_colors(t):
    colors = []
    for elec in electrodes:
        active = is_active(elec, t, window_intervals)
        colors.append(FOCUS_COLOR if active else "#666666")
    return colors

# Place electrodes on two lines by side prefix (L vs R), spaced out along x
x_positions = {}
y_positions = {}
left_x, right_x = 0, 0
for elec in electrodes:
    if str(elec).upper().startswith("L"):
        x_positions[elec] = left_x
        y_positions[elec] = 1
        left_x += 1
    else:
        x_positions[elec] = right_x
        y_positions[elec] = -1
        right_x += 1

x = [x_positions[e] for e in electrodes]
y = [y_positions[e] for e in electrodes]

frames = [
    go.Frame(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                marker=dict(size=16, color=frame_colors(t), opacity=0.95),
                text=electrodes,
                textposition="top center",
                textfont=dict(color="white", size=12),
                hoverinfo="text",
                hovertext=[f"{e}<br>t={t:.2f}s" for e in electrodes],
            )
        ],
        name=f"{t:.2f}",
    )
    for t in time_points
]

fig_line = go.Figure(
    data=[
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(size=16, color=frame_colors(tmin), opacity=0.95),
            text=electrodes,
            textposition="top center",
            textfont=dict(color="white", size=12),
            hoverinfo="text",
            hovertext=[f"{e}<br>t={tmin:.2f}s" for e in electrodes],
        )
    ],
    frames=frames,
)

fig_line.update_layout(
    template="plotly_dark",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, range=[-2, 2]),
    height=500,
    margin=dict(l=10, r=10, t=40, b=10),
    title=f"Electrode activity (window {tmin:.2f}s ? {tmax:.2f}s)",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(frame=dict(duration=frame_ms, redraw=True), fromcurrent=True, transition=dict(duration=0)),
                    ],
                ),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )
    ],
    sliders=[
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[f"{t:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                    label=f"{t:.2f}s",
                )
                for t in time_points
            ],
            transition=dict(duration=0),
            x=0.05,
            y=0,
            currentvalue=dict(prefix="Time: ", font=dict(color="white")),
            len=0.9,
        )
    ],
)

st.plotly_chart(fig_line, use_container_width=True)
