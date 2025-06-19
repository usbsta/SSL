#!/usr/bin/env python3
"""
Animated 3-D visualisation of four six-microphone arrays, their smoothed
bearings, and the least-squares intersection point for every frame in
*triangulation_debug_4arrays.csv*.

> Designed to replay the full 104-second experiment at its real-time speed.

Author  : Your-Name-Here
Created : 2025-06-19
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 (registers 3-D projection)
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import pathlib

# ────────────────────────── File locations ──────────────────────────────
CSV_PATH   = pathlib.Path("triangulation_debug_4arrays.csv")
OUTPUT_MP4 = pathlib.Path("triangulation_full_smooth.mp4")

# ─────────────────────── Array centres (UTM 56 S) ───────────────────────
ARRAY_CENTRES = {
    "N": np.array([322_955.1, 6_256_643.2, 0.0]),  # East, North, Up [m]
    "S": np.array([322_951.6, 6_256_580.0, 0.0]),
    "E": np.array([322_980.8, 6_256_638.4, 0.0]),
    "W": np.array([322_918.0, 6_256_605.4, 0.0]),
}

# ──────────────────────────── Parameters ────────────────────────────────
RAY_LENGTH = 20.0          # Arrow length for visualisation [m]
FALLBACK_GIF = OUTPUT_MP4.with_suffix(".gif")

# ─────────────────────── Helper functions ───────────────────────────────
def az_el_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """
    Convert ENU azimuth/elevation angles (degrees) to a 3-D unit vector.

    0° azimuth = North, +90° = East; elevation 0–90° is upward from horizon.
    """
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),   # East
        np.cos(el) * np.cos(az),   # North
        np.sin(el)                 # Up
    ])

# ───────────────────────────── Main code ────────────────────────────────
def main() -> None:
    # ── Load CSV ─────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)

    # Derive sampling interval from timestamps (assumes column 'time_s')
    if "time_s" in df.columns:
        dt  = float(np.median(np.diff(df["time_s"])))   # ≈ 0.1 s
    else:
        dt  = 0.1                                       # fallback
    fps = max(1, round(1.0 / dt))                      # integer FPS for writer

    # ── Pre-compute direction vectors for every array & frame ────────────
    for key in ARRAY_CENTRES:
        vecs = df.apply(
            lambda r: az_el_to_unit(r[f"smooth_az_{key}"], r[f"smooth_el_{key}"]),
            axis=1,
        )
        df[[f"dx_{key}", f"dy_{key}", f"dz_{key}"]] = np.stack(vecs.values)

    # ── Build the 3-D figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection="3d")

    # Static markers: array centres
    for centre in ARRAY_CENTRES.values():
        ax.scatter(*centre, marker="P", s=50)

    # Dynamic artists: one star for the source, one arrow per array
    src_scatter = ax.scatter([], [], [], marker="*", s=120)
    quivers     = {k: ax.quiver([], [], [], [], [], []) for k in ARRAY_CENTRES}

    def init() -> list:
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_zlabel("Up [m]")
        ax.set_title("Triangulation (smoothed bearings)")
        return [src_scatter] + list(quivers.values())

    def update(idx: int) -> list:
        row = df.iloc[idx]

        # Update source estimate
        src_scatter._offsets3d = ([row["x_est_m"]],
                                  [row["y_est_m"]],
                                  [row["z_est_m"]])

        # Update arrows for each array
        for k, centre in ARRAY_CENTRES.items():
            dx, dy, dz = (row[f"dx_{k}"] * RAY_LENGTH,
                          row[f"dy_{k}"] * RAY_LENGTH,
                          row[f"dz_{k}"] * RAY_LENGTH)
            quivers[k].remove()                         # erase previous arrow
            quivers[k] = ax.quiver(*centre, dx, dy, dz) # draw new arrow

        return [src_scatter] + list(quivers.values())

    # ── Construct the animation ─────────────────────────────────────────
    frames = range(len(df))   # use *all* rows → full 104-s clip
    anim = FuncAnimation(fig, update, frames=frames,
                         init_func=init, interval=dt * 1000, blit=False)

    # ── Save to file ────────────────────────────────────────────────────
    try:
        anim.save(OUTPUT_MP4, writer=FFMpegWriter(fps=fps))
        print(f"✔ Animation saved → {OUTPUT_MP4.resolve()}")
    except (RuntimeError, FileNotFoundError):
        # FFmpeg not found: fall back to a GIF
        anim.save(FALLBACK_GIF, writer=PillowWriter(fps=fps))
        print(f"✔ Animation saved → {FALLBACK_GIF.resolve()}")

    plt.close(fig)            # avoid duplicate static plot in notebooks

# ─────────────────────────── Script entry ───────────────────────────────
if __name__ == "__main__":
    main()
