import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pyvista as pv
from stpyvista import stpyvista

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
if not filtered_elec.empty:
    # ✅ color map by brain region (base colors)
    region_colors = {
        "HPC": "#00FFFF",       # cyan
        "PFC": "#FFD700",       # gold
        "Temporal": "#ADFF2F",  # green-yellow
        "AMYG": "#FF69B4",      # pink
        "Other": "#AAAAAA"      # gray
    }

    focus_color = "#FF0000"  # 🔴 red for the focused electrode

    electrodes = sorted(filtered_elec["electrode"].dropna().unique())

    for electrode in electrodes:
        sub = filtered_elec[filtered_elec["electrode"] == electrode].sort_values("timestamp")
        sub["on"] = sub["spike_value"] > 0

        # region per electrode
        region = sub["region"].iloc[0] if "region" in sub.columns else "Other"
        base_color = region_colors.get(region, "#AAAAAA")

        start_time = None
        for _, row in sub.iterrows():
            # Electrode turns ON
            if row["on"] and start_time is None:
                start_time = row["spike_value"]

            # Electrode turns OFF
            elif not row["on"] and start_time is not None:
                end_time = row["timestamp"]

                # ✅ check if this electrode was in focus during this time
                focus_rows = sub[
                    (sub["timestamp"] >= start_time)
                    & (sub["timestamp"] <= end_time)
                    & (sub["Focus (Spatial)"] == electrode)
                ]

                # ✅ if electrode is focused → red, else region color
                color = focus_color if not focus_rows.empty else base_color

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

                start_time = None

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

st.title("🧠 3D Human Brain Visualization")

pv.OFF_SCREEN = True

# Load hemisphere meshes
lh = pv.read("lh.pial.obj")
rh = pv.read("rh.pial.obj")

brain = lh.merge(rh)
brain.clean(inplace=True)
brain.compute_normals(inplace=True)

plotter = pv.Plotter(window_size=[900, 650])
plotter.set_background("#0E1117")

plotter.add_mesh(
    brain,
    color="#cccccc",
    smooth_shading=True,
    opacity=0.25,
    specular=0.4,
    specular_power=20
)

plotter.enable_eye_dome_lighting()
plotter.add_axes()
plotter.camera.zoom(0.7)

# --- Helper: Sample point on cortex (outer surface)
def place_on_cortex(brain_mesh):
    faces = brain_mesh.faces.reshape(-1, 4)[:, 1:]
    idx = np.random.randint(len(faces))
    tri = brain_mesh.points[faces[idx]]
    r1, r2 = np.random.rand(2)
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return (1 - r1 - r2) * tri[0] + r1 * tri[1] + r2 * tri[2]

# --- Test: place one electrode on surface ---
pos = place_on_cortex(brain)
sphere = pv.Sphere(radius=2.5, center=pos)
plotter.add_mesh(sphere, color="#00FFFF")

# --- HPC & Amygdala approximate centers ---
hpc_left  = [-36, -20, -14]
hpc_right = [36, -20, -14]
amyg_left  = [-24, -6, -22]
amyg_right = [24, -6, -22]

# Convert MNI-like coordinates to brain mesh scale
scale_x = (brain.bounds[1] - brain.bounds[0]) / 140
scale_y = (brain.bounds[3] - brain.bounds[2]) / 180
scale_z = (brain.bounds[5] - brain.bounds[4]) / 160

def mni_to_brain(mni):
    return [
        mni[0] * scale_x,
        mni[1] * scale_y,
        mni[2] * scale_z,
    ]

# --- Add HPC spheres ---
hpc_l = pv.Sphere(radius=4.0, center=mni_to_brain(hpc_left))
hpc_r = pv.Sphere(radius=4.0, center=mni_to_brain(hpc_right))
plotter.add_mesh(hpc_l, color="#00FFFF", opacity=0.7)
plotter.add_mesh(hpc_r, color="#00FFFF", opacity=0.7)

# --- Add Amygdala spheres ---
am_l = pv.Sphere(radius=4.0, center=mni_to_brain(amyg_left))
am_r = pv.Sphere(radius=4.0, center=mni_to_brain(amyg_right))
plotter.add_mesh(am_l, color="#FF69B4", opacity=0.7)
plotter.add_mesh(am_r, color="#FF69B4", opacity=0.7)

stpyvista(plotter, key="brain_real", use_container_width=True)
