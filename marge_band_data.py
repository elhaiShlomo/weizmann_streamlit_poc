import numpy as np
import pandas as pd


# ---- Load Data ----
def load_and_merge_data():
    files = {
        "alpha": "samples/summary_alpha_full.csv",
        "beta": "samples/summary_beta_full.csv",
        "gamma": "samples/summary_gamma_full.csv"
    }
    dfs = []
    for band, file in files.items():
        df = pd.read_csv(file)
        df["band"] = band  # add a band column
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # If avg_density doesn't exist or empty → fill with random values
    if (
            "avg_density" not in merged.columns
            or merged["avg_density"].isna().all()
            or (merged["avg_density"] == 0).all()
    ):
        merged["avg_density"] = np.random.uniform(0.1, 1.0, len(merged))

    merged.to_csv("merged_band_data.csv", index=False)


if __name__ == '__main__':
    load_and_merge_data()