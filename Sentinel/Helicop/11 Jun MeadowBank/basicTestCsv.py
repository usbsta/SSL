#!/usr/bin/env python3
"""
Animated 3-D visualisation of four 6-mic arrays, their smoothed
bearings and the least-squares intersection point.

Author  : Your-Name-Here
Created : 2025-06-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D               # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import pathlib

# ── Data ────────────────────────────────────────────────────────────────
CSV_PATH = pathlib.Path("triangulation_debug_4arrays.csv")   # <-- ajusta si es necesario
df = pd.read_csv(CSV_PATH)

# ── Array centres (UTM 56 S) ────────────────────────────────────────────
C = {                                                         # East, North, Up
    "N": np.array([322955.1, 6256643.2, 0.0]),
    "S": np.array([322951.6, 6256580.0, 0.0]),
    "E": np.array([322980.8, 6256638.4, 0.0]),
    "W": np.array([322918.0, 6256605.4, 0.0]),
}

# ── Helpers ─────────────────────────────────────────────────────────────
def az_el_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """ENU azimuth/elevation → 3-D unit vector."""
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),   # East
        np.cos(el) * np.cos(az),   # North
        np.sin(el)                 # Up
    ])

# Pre-compute direction vectors (smoothed values) for every frame
for key in C:                # N, S, E, W
    vecs = df.apply(
        lambda r: az_el_to_unit(r[f"smooth_az_{key}"], r[f"smooth_el_{key}"]),
        axis=1,
    )
    df[[f"dx_{key}", f"dy_{key}", f"dz_{key}"]] = np.stack(vecs.values)

# ── Figure & artists ────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection="3d")

# Static elements: array centres
for centre in C.values():
    ax.scatter(*centre, marker="P", s=50)      # default colour cycle

# Dynamic artists
src_scatter = ax.scatter([], [], [], marker="*", s=120)
quivers     = {k: ax.quiver([], [], [], [], [], []) for k in C}

RAY_LEN = 20.0  # visual length of arrows (m)

def init():
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Triangulation (smoothed bearings)")
    return [src_scatter] + list(quivers.values())

def update(idx: int):
    row = df.iloc[idx]

    # Update source estimate
    src_scatter._offsets3d = ([row["x_est_m"]],
                              [row["y_est_m"]],
                              [row["z_est_m"]])

    # Update one quiver per array
    for k, centre in C.items():
        dx, dy, dz = (row[f"dx_{k}"] * RAY_LEN,
                      row[f"dy_{k}"] * RAY_LEN,
                      row[f"dz_{k}"] * RAY_LEN)
        quivers[k].remove()                           # clear previous arrow
        quivers[k] = ax.quiver(*centre, dx, dy, dz)   # draw new arrow

    return [src_scatter] + list(quivers.values())

# ── Animation ───────────────────────────────────────────────────────────
# Sample every 10th frame → ~100 frames for 1 k-row file
FRAMES = range(0, len(df), 10)

ani = FuncAnimation(fig, update, frames=FRAMES,
                    init_func=init, interval=100, blit=False)

OUT_MP4 = pathlib.Path("triangulation_animation.mp4")
try:
    ani.save(OUT_MP4, writer=FFMpegWriter(fps=10))
    print(f"✔ Animation saved → {OUT_MP4.resolve()}")
except (RuntimeError, FileNotFoundError):
    # Fallback if FFmpeg is missing
    OUT_GIF = OUT_MP4.with_suffix(".gif")
    ani.save(OUT_GIF, writer=PillowWriter(fps=10))
    print(f"✔ Animation saved → {OUT_GIF.resolve()}")

plt.close(fig)   # prevent duplicate static display when run in notebooks
