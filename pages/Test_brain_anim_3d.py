import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pyvista as pv


def build_electrode_intervals(df_elec: pd.DataFrame):
    """Build activation intervals for each electrode using spike_value as start and next timestamp as end."""
    intervals = {}
    for elec, sub in df_elec.groupby("electrode"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        spans = []
        for i in range(len(sub) - 1):
            spike = float(sub.loc[i, "spike_value"])
            next_ts = float(sub.loc[i + 1, "timestamp"])
            if spike > 0.0:
                spans.append((spike, next_ts))
        intervals[elec] = spans
    return intervals


def is_active(electrode: str, t: float, intervals: dict) -> bool:
    """Return True if time t is inside any activation interval for electrode."""
    for start, end in intervals.get(electrode, []):
        if start <= t < end:
            return True
    return False


def filter_intervals_by_window(intervals: dict, tmin: float, tmax: float) -> dict:
    """Keep only activation intervals that overlap the global time window."""
    trimmed = {}
    for elec, spans in intervals.items():
        kept = []
        for start, end in spans:
            if end >= tmin and start <= tmax:
                kept.append((max(start, tmin), min(end, tmax)))
        trimmed[elec] = kept
    return trimmed


def build_time_frames(tmin: float, tmax: float, fps: int = 20, max_frames: int = 150, min_step: float | None = None):
    """Build time points across the current window for playback."""
    if tmax <= tmin:
        return [tmin]
    if min_step is not None and min_step > 0:
        total_frames = min(max_frames, max(2, int((tmax - tmin) / min_step) + 1))
    else:
        total_frames = min(max_frames, max(2, int((tmax - tmin) * fps) + 1))
    return list(np.linspace(tmin, tmax, total_frames))


def assign_region(electrode_name: str) -> str:
    """Assign brain region based on electrode naming patterns."""
    if not isinstance(electrode_name, str) or electrode_name.strip() == "":
        return "Other"
    e = electrode_name.upper().split("-")[0]
    if e.startswith(("LA", "RA", "LMH", "RMH", "AH", "PH", "MH", "HPC")):
        return "HPC"
    if e.startswith(("LAM", "RAM", "AMY", "AMYG")):
        return "AMYG"
    if e.startswith(("LT", "RT", "TEMP", "MT")):
        return "Temporal"
    if e.startswith(("LPFC", "RPFC", "FP", "PFC", "LPF", "RPF")):
        return "PFC"
    return "Other"


REGION_COLORS = {
    "HPC": "#00FFFF",
    "PFC": "#FFD700",
    "Temporal": "#ADFF2F",
    "AMYG": "#FF69B4",
    "Other": "#AAAAAA",
}
FOCUS_COLOR = "#FF0000"

st.set_page_config(page_title="3D Brain Electrodes", layout="wide")
st.title("3D Electrode Activity on MRI Brain")


@st.cache_data(show_spinner=False)
def load_data():
    df_main = pd.read_csv("samples/merged_band_data.csv")
    df_events = pd.read_csv("samples/clinical_events_new.csv")
    df_electrodes = pd.read_csv("samples/all_patients_unified.csv")
    return df_main, df_events, df_electrodes


@st.cache_resource(show_spinner=False)
def load_brain_mesh(path: str = "A_Human_Brain_realis_0725031124_refine.obj", decimate: float | None = 0.65):
    mesh = pv.read(path)
    if decimate is not None and 0 < decimate < 1:
        mesh = mesh.decimate(decimate, inplace=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return mesh.points, faces, mesh.bounds


def find_coord_columns(df: pd.DataFrame) -> list[str] | None:
    candidates = [
        ["x", "y", "z"],
        ["x_mm", "y_mm", "z_mm"],
        ["x_coord", "y_coord", "z_coord"],
    ]
    for cols in candidates:
        if set(cols).issubset(df.columns):
            return cols
    return None


def prepare_coordinates(df_elec: pd.DataFrame, electrodes: list[str], bounds) -> tuple[pd.DataFrame | None, str | None]:
    cols = find_coord_columns(df_elec)
    if not cols:
        return None, "Add electrode coordinates (columns x,y,z or x_mm,y_mm,z_mm) to see them on the brain."
    coords = (
        df_elec.drop_duplicates("electrode")
        .set_index("electrode")[cols]
        .reindex(electrodes)
    )
    missing = coords.isna().any(axis=1)
    if missing.any():
        coords = coords[~missing]
    if coords.empty:
        return None, "No electrode has a full set of coordinates."
    coords.columns = ["x", "y", "z"]
    return coords, None


def synthetic_coordinates(electrodes: list[str], bounds) -> pd.DataFrame:
    """Deterministic fallback positions spread around the mesh bounds (approximate only)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    span_x = (xmax - xmin) * 0.2
    span_y = (ymax - ymin) * 0.45
    span_z = (zmax - zmin) * 0.15

    coords = []
    for idx, elec in enumerate(electrodes):
        side = -1 if str(elec).upper().startswith("L") else 1
        y_offset = ((idx % 10) - 4.5) / 5 * span_y
        z_offset = ((idx // 10) - 0.5) * (span_z / max(1, len(electrodes) // 8 + 1))
        coords.append((elec, x_center + side * span_x, y_center + y_offset, z_center + z_offset))
    return pd.DataFrame(coords, columns=["electrode", "x", "y", "z"]).set_index("electrode")


# --- Load data ---
try:
    df, df_events, df_electrodes = load_data()
except Exception as e:  # pragma: no cover - UI feedback
    st.error(f"Failed to load data: {e}")
    st.stop()

# Annotate regions
df_electrodes["region"] = df_electrodes["electrode"].apply(assign_region)

# Sidebar filters
st.sidebar.header("Controls")
if "patient_id" in df_events.columns:
    patients = sorted(df_events["patient_id"].dropna().unique())
    selected_patient = st.sidebar.selectbox("Patient ID", patients)
    df_events = df_events[df_events["patient_id"] == selected_patient]
    if "patient_id" in df_electrodes.columns:
        df_electrodes = df_electrodes[df_electrodes["patient_id"] == selected_patient]

admissions = sorted(df_events.get("admission_id", pd.Series(dtype=float)).dropna().unique())
selected_admission = st.sidebar.selectbox("Admission ID", admissions) if len(admissions) else None
if selected_admission is not None and "admission_id" in df_events.columns:
    df_events = df_events[df_events["admission_id"] == selected_admission]
    if "admission_id" in df_electrodes.columns:
        df_electrodes = df_electrodes[df_electrodes["admission_id"] == selected_admission]

if "band" in df.columns:
    bands = sorted(df["band"].dropna().unique())
    selected_bands = st.sidebar.multiselect("EEG Bands", bands, default=bands)
    df = df[df["band"].isin(selected_bands)]

if "event_name" in df_events.columns:
    events = sorted(df_events["event_name"].dropna().unique())
    selected_events = st.sidebar.multiselect("Clinical Events", events, default=events)
    df_events = df_events[df_events["event_name"].isin(selected_events)]

if "electrode" in df_electrodes.columns:
    all_elec = sorted(df_electrodes["electrode"].dropna().unique())
    selected_electrodes = st.sidebar.multiselect("Electrodes", all_elec, default=all_elec)
    df_electrodes = df_electrodes[df_electrodes["electrode"].isin(selected_electrodes)]

if "region" in df_electrodes.columns:
    regions = sorted(df_electrodes["region"].dropna().unique())
    selected_regions = st.sidebar.multiselect("Brain Regions", regions, default=regions)
    df_electrodes = df_electrodes[df_electrodes["region"].isin(selected_regions)]

# Time window
if not {"time_s"}.issubset(df.columns):
    st.error("merged_band_data.csv must contain time_s column")
    st.stop()

start_default = float(df["time_s"].min())
end_default = float(df["time_s"].max())
tmin, tmax = st.sidebar.slider(
    "Time Range (s)",
    start_default,
    end_default,
    (start_default, end_default),
)

# Playback speed
playback_fps = st.sidebar.slider("Playback speed (fps)", min_value=1, max_value=60, value=10, step=1)
frame_ms = max(10, int(1000 / playback_fps))

# Filter by time for events/electrodes
filtered_elec = df_electrodes[(df_electrodes["timestamp"] >= tmin) & (df_electrodes["timestamp"] <= tmax)]
electrodes = sorted(filtered_elec["electrode"].dropna().unique())

if not electrodes:
    st.warning("No electrodes in the selected filters/time window.")
    st.stop()

electrode_intervals = build_electrode_intervals(df_electrodes)
window_intervals = filter_intervals_by_window(electrode_intervals, tmin, tmax)

# Load brain mesh
points, faces, bounds = load_brain_mesh()

coords, coord_msg = prepare_coordinates(df_electrodes, electrodes, bounds)
used_fallback = False
if coords is None:
    coords = synthetic_coordinates(electrodes, bounds)
    used_fallback = True
    if coord_msg:
        st.info(coord_msg + " Using synthetic positions for preview only.")

# Keep only electrodes that have coordinates
plot_electrodes = [e for e in electrodes if e in coords.index]
if not plot_electrodes:
    st.warning("No electrodes have coordinates available to plot.")
    st.stop()

coords = coords.loc[plot_electrodes]
x = coords["x"].tolist()
y = coords["y"].tolist()
z = coords["z"].tolist()

# Build animation frames
time_points = build_time_frames(tmin, tmax, max_frames=400, min_step=0.05)


def frame_colors(t: float):
    colors = []
    for elec in plot_electrodes:
        active = is_active(elec, t, window_intervals)
        colors.append(FOCUS_COLOR if active else "#666666")
    return colors


mesh_trace = go.Mesh3d(
    x=points[:, 0], y=points[:, 1], z=points[:, 2],
    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    color="#A0A0A0", opacity=0.25, name="Brain", showscale=False,
)

frames = [
    go.Frame(
        data=[
            mesh_trace,
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers+text",
                marker=dict(size=6, color=frame_colors(t), opacity=0.95),
                text=plot_electrodes,
                textposition="top center",
                textfont=dict(color="white", size=12),
                hoverinfo="text",
                hovertext=[f"{e}<br>t={t:.2f}s" for e in plot_electrodes],
                name="Electrodes",
            ),
        ],
        name=f"{t:.2f}",
    )
    for t in time_points
]

fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    template="plotly_dark",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
    ),
    height=750,
    margin=dict(l=0, r=0, t=60, b=0),
    title=f"Electrode activity on 3D brain ({tmin:.2f}s � {tmax:.2f}s)",
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

if used_fallback:
    st.caption("Showing synthetic electrode positions (no coordinate columns found). Add x/y/z columns for real placement.")

st.plotly_chart(fig, use_container_width=True)

