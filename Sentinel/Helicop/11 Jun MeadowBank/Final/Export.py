#!/usr/bin/env python3
"""
Flat-plane localisation usando az/el PRE-calculados + visualización tipo “cometa”
con cola semitransparente para:
    • estimación (verde)
    • trayectoria ground-truth suavizada (celeste)

Lee:
  – triangulation_debug_4arrays2.csv  → raw_az_*, raw_el_*
  – interpolated_flight_data_100ms.csv → GT

Autor : ChatGPT (OpenAI-o3) – 2025-06-25
"""

# ─── Built-ins ───────────────────────────────────────────────────────────
from collections import deque
from pathlib import Path
from typing import Dict, List

# ─── Third-party ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PathCollection
from pyproj import Transformer
import contextily as ctx

# ───────── CONFIGURACIÓN BÁSICA ─────────────────────────────────────────
CSV_AZEL   = Path("triangulation_debug_4arrays2.csv")
FLIGHT_CSV = Path("interpolated_flight_data_100ms.csv")
VIDEO_OUT  = Path("flat_plane_csv.mp4")

# Rangos (0-based, END exclusivo)
FL_START_IDX, FL_END_IDX = 2090, 2400
START_IDX,    END_IDX    =  600,  900

ARRAY_CENTRES = {
    "N": np.array([322_955.1, 6_256_643.2, 0.0]),
    "S": np.array([322_951.6, 6_256_580.0, 0.0]),
    "E": np.array([322_980.8, 6_256_638.4, 0.0]),
    "W": np.array([322_918.0, 6_256_605.4, 0.0]),
}
ARRAY_KEYS = tuple(ARRAY_CENTRES)

SMOOTH_LEN      = 140        # ventana de suavizado az/el
GT_SMOOTH_LEN   = 160        # ventana de suavizado GT
EST_TAIL_LEN    =  25        # longitud cola estimación
GT_TAIL_LEN     =  25        # longitud cola GT
PLANE_ALT_M     = 210.0
FPS_VIDEO       = 10
VERBOSE         = True

# ───────── TRANSFORMACIONES ─────────────────────────────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)
utm2merc = lambda e, n: np.array(_trW2M.transform(*_trU2W.transform(e, n)),
                                 dtype=np.float64)

P_merc = {k: utm2merc(*ARRAY_CENTRES[k][:2]) for k in ARRAY_KEYS}
P0     = P_merc["N"]

# ───────── MAPA & FIGURA ────────────────────────────────────────────────
margin    = 1000
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM, yminM = centre_xy - margin
xmaxM, ymaxM = centre_xy + margin
img, ext = ctx.bounds2img(xminM, yminM, xmaxM, ymaxM,
                          zoom=10, source=ctx.providers.Esri.WorldImagery)
xminL, xmaxL = ext[0] - P0[0], ext[1] - P0[0]
yminL, ymaxL = ext[2] - P0[1], ext[3] - P0[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=(xminL, xmaxL, yminL, ymaxL), origin="upper", zorder=0)

for k in ARRAY_KEYS:
    xy = P_merc[k] - P0
    ax.scatter(*xy, marker="^", edgecolor="k", s=80)
    ax.text(*(xy + [3, 3]), f" {k}", color="white")

# ─── Scatter para las dos colas ─────────────────────────────────────────
gt_scatter:  PathCollection = ax.scatter([], [], s=40, edgecolors='none',
                                         zorder=3, label="GT (smooth)")
est_scatter: PathCollection = ax.scatter([], [], s=40, edgecolors='none',
                                         zorder=4, label="Estimation")

dist_text = ax.text(0.97, 0.02, "", transform=ax.transAxes,
                    ha="right", va="bottom", color="white",
                    backgroundcolor="black")

ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin)
ax.set_aspect("equal"); ax.legend(loc="upper right")

# ───────── UTILIDADES ───────────────────────────────────────────────────
def azel2unit(az_deg, el_deg):
    a, e = np.deg2rad([az_deg, el_deg])
    return np.array([np.cos(e)*np.sin(a),
                     np.cos(e)*np.cos(a),
                     np.sin(e)])

def ray_plane_intersect(P, d, z_plane):
    if abs(d[2]) < 1e-6:
        return None
    t = (z_plane - P[2]) / d[2]
    return None if t <= 0 else P + t*d

def circ_mean_deg(cosb, sinb):
    return np.rad2deg(np.arctan2(np.mean(sinb), np.mean(cosb))) if cosb else np.nan
lin_mean = lambda buf: np.mean(buf) if buf else np.nan

# ───────── CARGA CSVs ───────────────────────────────────────────────────
df = pd.read_csv(CSV_AZEL).iloc[START_IDX:END_IDX].reset_index(drop=True)
gt = pd.read_csv(FLIGHT_CSV).iloc[FL_START_IDX:FL_END_IDX].reset_index(drop=True)
hx, hy = _trW2M.transform(gt["Longitude"], gt["Latitude"])

n_blocks = len(df)

# Buffers suavizado az/el
smooth = {k: {"cos": deque(maxlen=SMOOTH_LEN),
              "sin": deque(maxlen=SMOOTH_LEN),
              "el":  deque(maxlen=SMOOTH_LEN)} for k in ARRAY_KEYS}

# Buffers GT
gt_buf_x = deque(maxlen=GT_SMOOTH_LEN)
gt_buf_y = deque(maxlen=GT_SMOOTH_LEN)

# Deques para colas
est_tail = deque(maxlen=EST_TAIL_LEN+1)  # newest right
gt_tail  = deque(maxlen=GT_TAIL_LEN+1)

# ───────── BUCLE PRINCIPAL ──────────────────────────────────────────────
with FFMpegWriter(fps=FPS_VIDEO, bitrate=2400).saving(fig, VIDEO_OUT, dpi=150):
    plt.show(block=False)
    for i, row in enumerate(df.itertuples(index=False)):
        # 1. Az/el crudos
        raw_az = {k: getattr(row, f"raw_az_{k}") for k in ARRAY_KEYS}
        raw_el = {k: getattr(row, f"raw_el_{k}") for k in ARRAY_KEYS}

        # 2. Smoothing buffers (estimación)
        for k in ARRAY_KEYS:
            rad = np.deg2rad(raw_az[k])
            smooth[k]["cos"].append(np.cos(rad))
            smooth[k]["sin"].append(np.sin(rad))
            smooth[k]["el"].append(raw_el[k])

        # 3. Rayos suavizados → plano
        P_smooth = []
        for k in ARRAY_KEYS:
            az_s = circ_mean_deg(smooth[k]["cos"], smooth[k]["sin"])
            el_s = lin_mean(smooth[k]["el"])
            d_s  = azel2unit(az_s, el_s)
            Pint = ray_plane_intersect(ARRAY_CENTRES[k], d_s, PLANE_ALT_M)
            if Pint is None:
                break
            P_smooth.append(Pint)
        else:
            centroid = np.mean(P_smooth, axis=0)
            xy_est   = utm2merc(*centroid[:2]) - P0
            est_tail.append(xy_est)

        # 4. Suavizado y cola para GT
        if i < len(hx):
            gt_buf_x.append(hx[i] - P0[0])
            gt_buf_y.append(hy[i] - P0[1])
            gx, gy = np.mean(gt_buf_x), np.mean(gt_buf_y)
            gt_tail.append((gx, gy))

        # 5. Actualiza scatters con α-fading
        def update_scatter(scatter, tail, base_rgb):
            pts = np.array(tail)
            al  = np.linspace(0.0, 1.0, len(pts))   # oldest→0, head→1
            cols = np.zeros((len(pts), 4))
            cols[:, :3] = base_rgb
            cols[:, 3]  = al
            scatter.set_offsets(pts)
            scatter.set_facecolors(cols)

        if est_tail:
            update_scatter(est_scatter, est_tail, (0.0, 1.0, 0.0))   # lime
        if gt_tail:
            update_scatter(gt_scatter,  gt_tail,  (0.0, 0.75, 1.0))  # celeste

        # Distancia media (opcional)
        if P_smooth:
            md = np.mean([np.linalg.norm(centroid-ARRAY_CENTRES[k])
                          for k in ARRAY_KEYS])
            dist_text.set_text(f"Mean dist = {md:.1f} m")

        fig.canvas.draw_idle(); fig.canvas.flush_events()

print("✅ MP4 ➔", VIDEO_OUT.resolve())
