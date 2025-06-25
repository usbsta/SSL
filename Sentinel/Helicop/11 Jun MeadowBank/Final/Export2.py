#!/usr/bin/env python3
"""
Flat-plane localisation with α-fading “comet” tails.

Inputs
------
• triangulation_debug_4arrays2.csv   →  raw_az_* / raw_el_* (per array)
• interpolated_flight_data_100ms.csv →  interpolated drone ground-truth

Output
------
• flat_plane_csv.mp4  –  MP4 with QuickTime-friendly encoding (yuv420p)

Author : ChatGPT (OpenAI-o3) – 2025-06-25
"""

# ────────────────────────────────────────────────────────────────────────
#  Standard library
# ────────────────────────────────────────────────────────────────────────
from collections import deque
from pathlib import Path
from typing import Dict, List

# ────────────────────────────────────────────────────────────────────────
#  Third-party
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from pyproj import Transformer
import contextily as ctx

# ────────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────────
CSV_AZEL   = Path("triangulation_debug_4arrays2.csv")
FLIGHT_CSV = Path("interpolated_flight_data_100ms.csv")
VIDEO_OUT  = Path("flat_plane_csv.mp4")

GT_SLICE   = slice(2090, 2400)     # ground-truth rows (start, end, 0-based, end excl.)
EST_SLICE  = slice( 600,  900)     # estimation rows

SMOOTH_LEN      = 140             # moving-average window for az/el
GT_SMOOTH_LEN   = 160             # moving-average window for GT (x,y)
EST_TAIL_LEN    =  25             # tail length (estimation “comet”)
GT_TAIL_LEN     =  25             # tail length (ground-truth “comet”)

PLANE_ALT_M     = 210.0           # projection altitude (metres AGL)
FPS_VIDEO       = 10
VERBOSE         = True

# ────────────────────────────────────────────────────────────────────────
#  Microphone-array centres, UTM-56S  [E, N, Z] in metres
# ────────────────────────────────────────────────────────────────────────
ARRAY_CENTRES: Dict[str, np.ndarray] = {
    "N": np.array([322_955.1, 6_256_643.2, 0.0]),
    "S": np.array([322_951.6, 6_256_580.0, 0.0]),
    "E": np.array([322_980.8, 6_256_638.4, 0.0]),
    "W": np.array([322_918.0, 6_256_605.4, 0.0]),
}
ARRAY_KEYS = tuple(ARRAY_CENTRES)

# ────────────────────────────────────────────────────────────────────────
#  CRS transform helpers
# ────────────────────────────────────────────────────────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)   # UTM-56S → WGS-84
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)    # WGS-84 → WebMerc
utm2merc = lambda e, n: np.array(_trW2M.transform(*_trU2W.transform(e, n)),
                                 dtype=np.float64)

P_merc = {k: utm2merc(*ARRAY_CENTRES[k][:2]) for k in ARRAY_KEYS}
P0     = P_merc["N"]                        # local origin (North array)

# ────────────────────────────────────────────────────────────────────────
#  Basemap & figure
# ────────────────────────────────────────────────────────────────────────
margin    = 1000
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM, yminM = centre_xy - margin
xmaxM, ymaxM = centre_xy + margin
img, ext = ctx.bounds2img(xminM, yminM, xmaxM, ymaxM,
                          zoom=19,
                          source=ctx.providers.Esri.WorldImagery)

xminL, xmaxL = ext[0] - P0[0], ext[1] - P0[0]
yminL, ymaxL = ext[2] - P0[1], ext[3] - P0[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=(xminL, xmaxL, yminL, ymaxL),
          origin="upper", zorder=0)

for k in ARRAY_KEYS:
    xy = P_merc[k] - P0
    ax.scatter(*xy, marker="^", edgecolor="k", s=80)
    ax.text(*(xy + [3, 3]), f" {k}", color="white")

# ────────────────────────────────────────────────────────────────────────
#  Scatter handles (α-fading)
# ────────────────────────────────────────────────────────────────────────
gt_scatter  = ax.scatter([], [], s=40, edgecolors="none",
                         zorder=3)
est_scatter = ax.scatter([], [], s=40, edgecolors="none",
                         zorder=4)
dist_text = ax.text(0.97, 0.02, "", transform=ax.transAxes,
                    ha="right", va="bottom",
                    color="white", backgroundcolor="black")

ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_aspect("equal")

# ────────────────────────────────────────────────────────────────────────
#  Legend (manual handles so they match the screenshot)
# ────────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0],
           color=(0.0, 1.0, 0.0),            # bright green line
           linewidth=2.5,
           marker='o',
           markersize=6,
           markerfacecolor=(0.0, 1.0, 0.0),
           markeredgecolor='black',
           label='Helicopter Estimation'),
    Line2D([0], [0],
           color=(0.0, 0.75, 1.0),           # cyan-blue line
           linewidth=2.5,
           label='Helicopter (GT)'),
]

ax.legend(handles=legend_handles,
          loc='upper right',
          framealpha=0.85,
          edgecolor='black')

# ────────────────────────────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────────────────────────────
def azel2unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation [deg] to unit ENU vector."""
    a, e = np.deg2rad([az_deg, el_deg])
    return np.array([np.cos(e) * np.sin(a),
                     np.cos(e) * np.cos(a),
                     np.sin(e)])

def ray_plane_intersect(P: np.ndarray, d: np.ndarray, z_plane: float):
    """Intersect ray  P + t·d  with horizontal plane  z = z_plane ."""
    if abs(d[2]) < 1e-6:
        return None
    t = (z_plane - P[2]) / d[2]
    return None if t <= 0 else P + t * d

def circ_mean_deg(cos_buf: deque, sin_buf: deque) -> float:
    """Circular mean (degrees) from cosine & sine buffers."""
    return (np.rad2deg(np.arctan2(np.mean(sin_buf), np.mean(cos_buf)))
            if cos_buf else np.nan)

lin_mean = lambda buf: np.mean(buf) if buf else np.nan  # noqa: E731

def update_tail_scatter(scatter: PathCollection,
                        tail: deque,
                        rgb: tuple):
    """Update scatter with α-fading tail."""
    pts = np.asarray(tail)                    # (N, 2)
    alphas = np.linspace(0.0, 1.0, len(pts))  # oldest→0, head→1
    cols = np.zeros((len(pts), 4))
    cols[:, :3] = rgb
    cols[:, 3] = alphas
    scatter.set_offsets(pts)
    scatter.set_facecolors(cols)

# ────────────────────────────────────────────────────────────────────────
#  Load CSVs
# ────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_AZEL).iloc[EST_SLICE].reset_index(drop=True)
gt = pd.read_csv(FLIGHT_CSV).iloc[GT_SLICE].reset_index(drop=True)
hx, hy = _trW2M.transform(gt["Longitude"], gt["Latitude"])

# Smoothing & tail buffers
smooth = {k: {"cos": deque(maxlen=SMOOTH_LEN),
              "sin": deque(maxlen=SMOOTH_LEN),
              "el":  deque(maxlen=SMOOTH_LEN)} for k in ARRAY_KEYS}
gt_buf_x, gt_buf_y = deque(maxlen=GT_SMOOTH_LEN), deque(maxlen=GT_SMOOTH_LEN)
est_tail = deque(maxlen=EST_TAIL_LEN + 1)   # newest right
gt_tail  = deque(maxlen=GT_TAIL_LEN + 1)

# ────────────────────────────────────────────────────────────────────────
#  Video writer
# ────────────────────────────────────────────────────────────────────────
writer = FFMpegWriter(fps=FPS_VIDEO,
                      bitrate=2400,
                      codec="libx264",
                      extra_args=["-pix_fmt", "yuv420p",
                                  "-movflags", "+faststart"])

with writer.saving(fig, str(VIDEO_OUT), dpi=150):
    for i, row in enumerate(df.itertuples(index=False)):
        # 1. Raw azimuth/elevation (each array)
        raw_az = {k: getattr(row, f"raw_az_{k}") for k in ARRAY_KEYS}
        raw_el = {k: getattr(row, f"raw_el_{k}") for k in ARRAY_KEYS}

        # 2. Update smoothing buffers
        for k in ARRAY_KEYS:
            r = np.deg2rad(raw_az[k])
            smooth[k]["cos"].append(np.cos(r))
            smooth[k]["sin"].append(np.sin(r))
            smooth[k]["el"].append(raw_el[k])

        # 3. Ray–plane intersections using smoothed az/el
        intersections = []
        for k in ARRAY_KEYS:
            az_s = circ_mean_deg(smooth[k]["cos"], smooth[k]["sin"])
            el_s = lin_mean(smooth[k]["el"])
            d_s  = azel2unit(az_s, el_s)
            P_int = ray_plane_intersect(ARRAY_CENTRES[k], d_s, PLANE_ALT_M)
            if P_int is None:
                break
            intersections.append(P_int)
        else:  # executed only if no break
            centroid = np.mean(intersections, axis=0)
            xy_est   = utm2merc(*centroid[:2]) - P0
            est_tail.append(xy_est)

        # 4. Ground-truth smoothing
        if i < len(hx):
            gt_buf_x.append(hx[i] - P0[0])
            gt_buf_y.append(hy[i] - P0[1])
            gx, gy = np.mean(gt_buf_x), np.mean(gt_buf_y)
            gt_tail.append((gx, gy))

        # 5. Refresh α-fading scatters
        if est_tail:
            update_tail_scatter(est_scatter, est_tail, (0.0, 1.0, 0.0))     # green
        if gt_tail:
            update_tail_scatter(gt_scatter,  gt_tail, (0.0, 0.75, 1.0))     # cyan-blue

        # 6. Mean distance text
        if len(intersections) == len(ARRAY_KEYS):
            mean_d = np.mean([np.linalg.norm(centroid - ARRAY_CENTRES[k])
                              for k in ARRAY_KEYS])
            dist_text.set_text(f"Mean dist = {mean_d:.1f} m")

        # 7. Write frame
        writer.grab_frame()   # internally calls fig.canvas.draw()

        if VERBOSE and i < 20:  # limited console output
            dbg = " | ".join(f"{k}: az={raw_az[k]:6.1f}, el={raw_el[k]:5.1f}"
                             for k in ARRAY_KEYS)
            print(f"[blk {i:04d}] {dbg}")

print("✅ Finished. Video saved to:", VIDEO_OUT.resolve())
plt.close(fig)
