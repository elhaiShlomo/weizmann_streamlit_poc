"""
build_unified_table.py
----------------------------------------
Creates a unified long-format table per patient
by merging `*_spike_times.csv` and `*_annotations.csv`.

Output:
    - one CSV per patient: sub_XX_unified.csv
    - optional: one combined file: all_patients_unified.csv
"""

import os
import re
import glob
import pandas as pd
from tqdm import tqdm


def load_and_unify_pair(spike_path: str, annot_path: str) -> pd.DataFrame:
    """
    Merge a pair of files (spike_times + annotations) into one long-format DataFrame.
    """

    # --- Extract patient_id from filename (e.g., sub_01) ---
    match = re.search(r"(sub[_-]?\d+)", os.path.basename(spike_path), re.IGNORECASE)
    patient_id = match.group(1) if match else "unknown_patient"

    print(f"Processing {patient_id} ...")

    # --- Load files ---
    spikes = pd.read_csv(spike_path)
    annotations = pd.read_csv(annot_path)

    # --- Clean annotation time window ---
    if "Time Window" not in annotations.columns:
        raise ValueError(f"'Time Window' column missing in {annot_path}")

    annotations[['start_time', 'end_time']] = (
        annotations['Time Window'].str.split(' - ', expand=True).astype(float)
    )

    # --- Melt the spike_times table (wide → long) ---
    spikes_long = spikes.melt(
        id_vars=['Time Window', 'Focus (Temporal)', 'Focus (Spatial)'],
        var_name='electrode',
        value_name='spike_value'
    )
    spikes_long['timestamp'] = spikes_long['Time Window'].astype(float)

    # --- Map annotation windows ---
    def match_annotation(row):
        match = annotations[
            (row['timestamp'] >= annotations['start_time']) &
            (row['timestamp'] <= annotations['end_time'])
        ]
        if not match.empty:
            first = match.iloc[0]
            return pd.Series([
                True,
                first['Focus (Spatial)'],
                first['start_time'],
                first['end_time'],
                first.get('LA1', None) or first.get('Event Type', None)
            ])
        else:
            return pd.Series([False, None, None, None])

    spikes_long[['annotation_flag', 'annotation_label',
                 'annotation_window_start', 'annotation_window_end',
                 'annotation_type']] = spikes_long.apply(match_annotation, axis=1)

    # --- Add patient_id ---
    spikes_long['patient_id'] = patient_id

    # --- Type cleanup ---
    spikes_long['annotation_flag'] = spikes_long['annotation_flag'].astype(bool)
    spikes_long['patient_id'] = spikes_long['patient_id'].astype('category')
    spikes_long['electrode'] = spikes_long['electrode'].astype('category')

    # --- Reorder columns for clarity ---
    cols = ['patient_id', 'timestamp', 'electrode', 'spike_value',
            'annotation_flag', 'annotation_label',
            'annotation_window_start', 'annotation_window_end']
    remaining = [c for c in spikes_long.columns if c not in cols]
    spikes_long = spikes_long[cols + remaining]

    print(f"✅ {patient_id}: {len(spikes_long):,} rows merged.")
    return spikes_long


def main():
    print("Scanning for patient files...")
    spike_files = sorted(glob.glob("samples/raw_data/*_spike_times.csv"))
    all_dfs = []

    if not spike_files:
        print("No spike_times CSV files found in the current directory.")
        return

    for spike_path in tqdm(spike_files, desc="Processing patients"):
        # Find matching annotations file
        patient_base = re.sub(r"_spike_times\.csv$", "", spike_path)
        annot_path = f"{patient_base}_annotations.csv"

        if not os.path.exists(annot_path):
            print(f"No annotations file found for {spike_path}, skipping.")
            continue

        df_unified = load_and_unify_pair(spike_path, annot_path)

        # Save per-patient unified file
        unified_path = f"{patient_base}_unified.csv"
        df_unified.to_csv(unified_path, index=False)
        print(f"Saved: {unified_path}")

        all_dfs.append(df_unified)

    # Combine all patients (optional)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv("samples/all_patients_unified.csv", index=False)
        print(f"\nCombined dataset saved as: all_patients_unified.csv")
        print(f"Total rows: {len(combined):,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
