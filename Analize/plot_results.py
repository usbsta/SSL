#!/usr/bin/env python3
"""
Beamforming vs U‑Net  – grouped bars, CSV summary, negative‑log handling
"""

# ────────────────────────── USER CONFIG ──────────────────────────
CSV_PATH   = "test_No_drone_unet_20250426_001310.csv"
CSV_PATH   = "test_No_drone.csv"
SKIP_HEAD  = 50
SKIP_TAIL  = 100
STD_TH     = 10.0
DIST_BINS  = [25, 50, 75, 100, 200]          # custom bin edges
OUT_DIR    = "./"
SHOW_FIGS  = True
PLOT_STD   = False                       # include ±1 STD bars?
BAR_WIDTH  = 0.35
# ─────────────────────────────────────────────────────────────────

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

csv_stem = Path(CSV_PATH).stem
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ─── LOAD & TRIM ─────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
if SKIP_HEAD: df = df.iloc[SKIP_HEAD:]
if SKIP_TAIL: df = df.iloc[: len(df) - SKIP_TAIL]

# ─── VALIDITY MASKS (common) ─────────────────────────────
unet_valid = (
    df.get("valid_prediction", False).astype(bool)
    & (df.get("az_centroid_std", np.inf) < STD_TH)
    & (df.get("el_centroid_std", np.inf) < STD_TH)
)

az_std_roll = df.get("az_est", pd.Series(dtype=float)).rolling(3).std(ddof=0)
el_std_roll = df.get("el_est", pd.Series(dtype=float)).rolling(3).std(ddof=0)
beam_valid  = (az_std_roll < STD_TH) & (el_std_roll < STD_TH)

df["unet_valid"] = unet_valid.fillna(False)
df["beam_valid"] = beam_valid.fillna(False)

# ─── NEGATIVE‑LOG BRANCH ─────────────────────────────────
if "total_distance" not in df.columns:
    tot         = len(df)
    fp_bf       = df["beam_valid"].sum()
    fp_unet     = df["unet_valid"].sum()
    fp_rate_bf  = 100.0 * fp_bf   / tot if tot else 0.0
    fp_rate_un  = 100.0 * fp_unet / tot if tot else 0.0

    print("\nNEGATIVE‑SAMPLE LOG (no drone present)")
    print("────────────────────────────────")
    print(f"Beamforming  FP: {fp_bf}/{tot}  ({fp_rate_bf:.2f} %)")
    print(f"U‑Net        FP: {fp_unet}/{tot} ({fp_rate_un:.2f} %)")
    quit()

# ─── POSITIVE‑LOG PATH ──────────────────────────────────
# 1) angular errors (invalidate when prediction invalid)
df["ang_diff"]      = np.sqrt(df["az_diff"]**2      + df["el_diff"]**2)
df["ang_diff_unet"] = np.where(
    df["unet_valid"],
    np.sqrt(df["az_diff_unet"]**2 + df["el_diff_unet"]**2), np.nan)
df.loc[~df["beam_valid"], ["az_diff", "el_diff", "ang_diff"]] = np.nan
df.loc[~df["unet_valid"], ["az_diff_unet", "el_diff_unet", "ang_diff_unet"]] = np.nan

# 2) distance bins
edges  = [0] + DIST_BINS
if df["total_distance"].max() > DIST_BINS[-1]:
    edges.append(df["total_distance"].max() + 1)
labels = [f"{int(edges[i])}–{int(edges[i+1])}"
          for i in range(len(edges)-1)]
df["dist_bin"] = pd.cut(df["total_distance"], bins=edges,
                        labels=labels, right=False)

# 3) per‑bin stats
stats = df.groupby("dist_bin").agg(
    az_bf_mean  = ("az_diff",       "mean"),
    az_bf_std   = ("az_diff",       "std"),
    el_bf_mean  = ("el_diff",       "mean"),
    el_bf_std   = ("el_diff",       "std"),
    ang_bf_mean = ("ang_diff",      "mean"),
    ang_bf_std  = ("ang_diff",      "std"),

    az_un_mean  = ("az_diff_unet",  "mean"),
    az_un_std   = ("az_diff_unet",  "std"),
    el_un_mean  = ("el_diff_unet",  "mean"),
    el_un_std   = ("el_diff_unet",  "std"),
    ang_un_mean = ("ang_diff_unet", "mean"),
    ang_un_std  = ("ang_diff_unet", "std"),

    fn_bf_cnt   = ("beam_valid",    lambda x: (~x).sum()),
    fn_un_cnt   = ("unet_valid",    lambda x: (~x).sum()),
    total_cnt   = ("beam_valid",    "size"),
).reset_index()

stats["fnr_bf"]   = 100.0 * stats["fn_bf_cnt"]   / stats["total_cnt"]
stats["fnr_unet"] = 100.0 * stats["fn_un_cnt"]   / stats["total_cnt"]

# 4) SAVE CSV SUMMARY
stats.to_csv(Path(OUT_DIR) / f"{csv_stem}_summary.csv", index=False)
print(f"Summary CSV written to {csv_stem}_summary.csv")

# 5) plotting helpers
x       = np.arange(len(stats))
off     = BAR_WIDTH / 2
err_kw  = dict(capsize=5) if PLOT_STD else dict(capsize=0)

def bar_plot(mean_bf, std_bf, mean_un, std_un,
             title, ylabel, fname):
    plt.figure(figsize=(10,5))
    plt.bar(x-off, stats[mean_bf], BAR_WIDTH,
            yerr=(stats[std_bf] if PLOT_STD else None),
            label="Beamforming", **err_kw)
    plt.bar(x+off, stats[mean_un], BAR_WIDTH,
            yerr=(stats[std_un] if PLOT_STD else None),
            label="U‑Net", **err_kw)
    plt.xticks(x, stats["dist_bin"], rotation=0, ha="right")
    plt.xlabel("Distance range (m)")
    plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.margins(x=0.03); plt.tight_layout()
    plt.savefig(Path(OUT_DIR)/f"{csv_stem}_{fname}", dpi=300)

# 6) error bar‑plots
bar_plot("az_bf_mean","az_bf_std","az_un_mean","az_un_std",
         "Azimuth Error vs Distance","Mean absolute error (deg)",
         "azimuth_error_bars.png")
bar_plot("el_bf_mean","el_bf_std","el_un_mean","el_un_std",
         "Elevation Error vs Distance","Mean absolute error (deg)",
         "elevation_error_bars.png")
bar_plot("ang_bf_mean","ang_bf_std","ang_un_mean","ang_un_std",
         "Angular Error vs Distance","Mean absolute error (deg)",
         "angular_error_bars.png")

# 7) FNR bar‑plot with sample count in x‑label
xticks = [f"{l}\n(n={int(n)})"
          for l,n in zip(stats["dist_bin"], stats["total_cnt"])]

plt.figure(figsize=(10,5))
plt.bar(x-off, stats["fnr_bf"], BAR_WIDTH, label="Beamforming")
plt.bar(x+off, stats["fnr_unet"], BAR_WIDTH, label="U‑Net")
plt.xticks(x, xticks, rotation=0, ha="right")
plt.xlabel("Distance range (m)")
plt.ylabel("False‑negative rate (%)")
plt.title("False‑Negative Rate vs Distance")
plt.legend(); plt.margins(x=0.03); plt.tight_layout()
plt.savefig(Path(OUT_DIR)/f"{csv_stem}_fnr_vs_distance_bars.png", dpi=300)

# 8) overall FN summary
overall_fn_bf   = stats["fn_bf_cnt"].sum()
overall_fn_unet = stats["fn_un_cnt"].sum()
total_rows      = stats["total_cnt"].sum()
print("\nPOSITIVE LOG RESULTS")
print(f"Beamforming  FN: {overall_fn_bf}/{total_rows} "
      f"({100.0*overall_fn_bf/total_rows:.2f} %)")
print(f"U‑Net        FN: {overall_fn_unet}/{total_rows} "
      f"({100.0*overall_fn_unet/total_rows:.2f} %)")

if SHOW_FIGS:
    plt.show()
