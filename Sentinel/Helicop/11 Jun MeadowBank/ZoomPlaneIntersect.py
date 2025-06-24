#!/usr/bin/env python3
"""
Four-array FLAT-PLANE localisation (plane-intersection, no triangulation).

Logic:
 1. Beamform each array, get raw Az/El.
 2. Moving-average smoothing per array (length = SMOOTH_LEN).
 3. For each array, convert Az/El → unit ENU ray.
 4. Intersect ray with fixed altitude plane z = PLANE_ALT_M.
 5. Take the centroid of the four intersection points:
        centroid = mean(P0, P1, P2, P3)
    → use raw rays  for red dashed path
    → use smooth rays for lime solid path
 6. Plot ground-truth helicopter track, arrays, centroid paths.
 7. CSV-log angles, centroid XY (local), distances array-centroid, mean-dist.
 8. Live MP4 export (same as before).

Author : ChatGPT (OpenAI-o3) – 2025-06-24
"""

# ── Standard lib ────────────────────────────────────────────────────────
from collections import deque
import csv, sys, wave
from pathlib import Path
from typing import Dict, List

# ── Third-party ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from pyproj import Transformer
import contextily as ctx
import pandas as pd

# ── Project-specific helpers ────────────────────────────────────────────
from Utilities.functions import (
    apply_bandpass_filter, apply_beamforming, calculate_delays_for_direction
)
from Utilities.mic_geo import (
    mic_6_N_black_thin, mic_6_S_orange, mic_6_E_orange, mic_6_W_black
)

# ─────── FILE PATHS & GEOMETRY ──────────────────────────────────────────
ROOT = Path("/Users/a30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")
FLIGHT_CSV = Path("interpolated_flight_data_100ms.csv")

P_N = np.array([322_955.1, 6_256_643.2, 0.0])
P_S = np.array([322_951.6, 6_256_580.0, 0.0])
P_E = np.array([322_980.8, 6_256_638.4, 0.0])
P_W = np.array([322_918.0, 6_256_605.4, 0.0])

ARRAYS: Dict[str, dict] = {
    "N": {"wav": ROOT / "N.wav", "centre": P_N, "mics": mic_6_N_black_thin()},
    "S": {"wav": ROOT / "S.wav", "centre": P_S, "mics": mic_6_S_orange()},
    "E": {"wav": ROOT / "E.wav", "centre": P_E, "mics": mic_6_E_orange()},
    "W": {"wav": ROOT / "W.wav", "centre": P_W, "mics": mic_6_W_black()},
}
ARRAY_KEYS = tuple(ARRAYS)

# ─────── CALIBRATION OFFSETS (deg) ──────────────────────────────────────
#AZ_OFFSET = {"N": -44.0, "S": -8.0, "E": 0.0, "W": -28.0}
#EL_OFFSET = {"N":  0.0, "S":  1.0, "E": -1.0, "W":   4.0}

AZ_OFFSET = {'N':   0.0, 'S':  0.0, 'E':  0.0, 'W':   0.0}
EL_OFFSET = {'N':   0.0, 'S':  0.0, 'E':  0.0, 'W':   0.0}

# ─────── PROCESSING PARAMS ──────────────────────────────────────────────
RATE, CHUNK = 48_000, int(0.1 * 48_000)
LOWCUT, HIGHCUT, FILT_ORDER = 180.0, 2000.0, 5
C_SOUND  = 343.0            # m/s
AZIM_RANGE = np.arange(-180, 181, 1)
EL_RANGE  = np.arange(10, 51, 1)
SMOOTH_LEN = 10

START_TIME_S, END_TIME_S = 57, 80
PLANE_ALT_M = 200.0         # <-- Altura del plano (m sobre nivel del suelo local)

CSV_OUTPUT   = Path("flat_plane_4arrays.csv")
VIDEO_OUTPUT = Path("flat_plane_4arrays.mp4")

VERBOSE, MAX_PRINT = True, 100
DRAW_RAYS, RAY_LEN = True, 600.0

# ─────── COORD TRANSFORMS (UTM56 → WebMerc) ────────────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)
utm2merc = lambda e, n: np.array(_trW2M.transform(*_trU2W.transform(e, n)),
                                 dtype=np.float64)

P_merc = {k: utm2merc(*ARRAYS[k]["centre"][:2]) for k in ARRAY_KEYS}
P0 = P_merc["N"]                              # local origin

# ─────── FIGURE & BASEMAP ───────────────────────────────────────────────
margin = 850.0
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM, yminM = centre_xy - margin
xmaxM, ymaxM = centre_xy + margin
img, ext = ctx.bounds2img(xminM, yminM, xmaxM, ymaxM,
                          zoom=19, source=ctx.providers.Esri.WorldImagery)
xminL, xmaxL = ext[0] - P0[0], ext[1] - P0[0]
yminL, ymaxL = ext[2] - P0[1], ext[3] - P0[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=(xminL, xmaxL, yminL, ymaxL),
          origin="upper", zorder=0)

for k in ARRAY_KEYS:
    xy = P_merc[k] - P0
    ax.scatter(*xy, marker="^", edgecolor="k", s=80)
    ax.text(*(xy + [3, 3]), f" {k}", color="white")

traj_line,     = ax.plot([], [], "o-", color="lime", lw=2,   label="Centroid smooth")
traj_raw_line, = ax.plot([], [], "o--", color="red",  lw=1.3, label="Centroid raw")

heli_df = pd.read_csv(FLIGHT_CSV)
lon, lat = heli_df["Longitude"].values, heli_df["Latitude"].values
hx, hy = _trW2M.transform(lon, lat)
ax.plot(hx - P0[0], hy - P0[1], "-", color="deepskyblue", lw=1.3,
        label="Helicopter (GT)")

dist_text = ax.text(0.97, 0.02, "", transform=ax.transAxes,
                    ha="right", va="bottom",
                    color="white", backgroundcolor="black")
ax.set_xlim(-margin, margin); ax.set_ylim(-margin, margin); ax.set_aspect("equal")
ax.legend(loc="upper right")

# ─────── DELAY LUTs ─────────────────────────────────────────────────────
DELAY_LUT: Dict[str, np.ndarray] = {}
for k in ARRAY_KEYS:
    m = ARRAYS[k]["mics"]
    lut = np.empty((len(AZIM_RANGE), len(EL_RANGE), m.shape[0]), np.int32)
    for ia, az in enumerate(AZIM_RANGE):
        for ie, el in enumerate(EL_RANGE):
            lut[ia, ie] = calculate_delays_for_direction(m, az, el, RATE, C_SOUND)
    DELAY_LUT[k] = lut

# ─────── BUFFERS ────────────────────────────────────────────────────────
smooth = {k: {"az": deque(maxlen=SMOOTH_LEN), "el": deque(maxlen=SMOOTH_LEN)}
          for k in ARRAY_KEYS}
traj_xy, traj_raw_xy = [], []

# ─────── HELPERS ────────────────────────────────────────────────────────
def azel2unit(az: float, el: float) -> np.ndarray:
    a, e = np.deg2rad([az, el])
    return np.array([np.cos(e) * np.sin(a),
                     np.cos(e) * np.cos(a),
                     np.sin(e)])

def ray_plane_intersect(P: np.ndarray, d: np.ndarray, z_plane: float):
    """Intersect ray X(t)=P+t·d with horizontal plane z=z_plane. Return point or None."""
    if abs(d[2]) < 1e-6:
        return None
    t = (z_plane - P[2]) / d[2]
    return None if t <= 0 else P + t * d

# ─────── PROCESS BLOCK ──────────────────────────────────────────────────
def process_block(i: int, wf: Dict[str, wave.Wave_read], writer: csv.writer) -> bool:
    filt = {}
    for k in ARRAY_KEYS:
        nm = ARRAYS[k]["mics"].shape[0]
        data = wf[k].readframes(CHUNK)
        if len(data) < CHUNK * nm * 2:
            return False
        s = np.frombuffer(data, np.int16).reshape(-1, nm)
        s = apply_bandpass_filter(s, LOWCUT, HIGHCUT, RATE, order=FILT_ORDER)
        s /= (np.abs(s).max() or 1.0)
        filt[k] = s

    raw_az, raw_el = {}, {}
    for k in ARRAY_KEYS:
        E = np.zeros((len(AZIM_RANGE), len(EL_RANGE)))
        for ia, az in enumerate(AZIM_RANGE):
            for ie, el in enumerate(EL_RANGE):
                y = apply_beamforming(filt[k], DELAY_LUT[k][ia, ie])
                E[ia, ie] = np.sum(y**2) / CHUNK
        ia, ie = np.unravel_index(np.argmax(E), E.shape)
        raw_az[k], raw_el[k] = AZIM_RANGE[ia], EL_RANGE[ie]
        smooth[k]["az"].append(raw_az[k]); smooth[k]["el"].append(raw_el[k])

    # ▸▸▸ RAW rays → plane
    P_raw = []
    for k in ARRAY_KEYS:
        az_r = raw_az[k] + AZ_OFFSET[k]
        el_r = raw_el[k] + EL_OFFSET[k]
        d_r  = azel2unit(az_r, el_r)
        Pint = ray_plane_intersect(ARRAYS[k]["centre"], d_r, PLANE_ALT_M)
        if Pint is None:  # skip frame if any ray misses the plane
            return True
        P_raw.append(Pint)
    centroid_raw = np.mean(P_raw, axis=0)
    xy_raw = utm2merc(*centroid_raw[:2]) - P0
    traj_raw_xy.append(xy_raw)
    traj_raw_line.set_data(*zip(*traj_raw_xy))

    # ▸▸▸ SMOOTH rays → plane
    P_smooth = []
    for k in ARRAY_KEYS:
        az_s = np.mean(smooth[k]["az"]) + AZ_OFFSET[k]
        el_s = np.mean(smooth[k]["el"]) + EL_OFFSET[k]
        d_s  = azel2unit(az_s, el_s)
        Pint = ray_plane_intersect(ARRAYS[k]["centre"], d_s, PLANE_ALT_M)
        if Pint is None:
            return True
        P_smooth.append(Pint)
    centroid = np.mean(P_smooth, axis=0)
    xy = utm2merc(*centroid[:2]) - P0
    traj_xy.append(xy)
    traj_line.set_data(*zip(*traj_xy))

    # Distancias a cada array
    dists = [np.linalg.norm(centroid - ARRAYS[k]["centre"]) for k in ARRAY_KEYS]
    mean_dist = np.mean(dists)
    dist_text.set_text(f"Mean dist = {mean_dist:.1f} m")

    # CSV
    writer.writerow([
        i, f"{START_TIME_S + i*CHUNK/RATE:.3f}",
        *[f"{raw_az[k]:.1f}" for k in ARRAY_KEYS],
        *[f"{raw_el[k]:.1f}" for k in ARRAY_KEYS],
        *[f"{np.mean(smooth[k]['az']):.1f}" for k in ARRAY_KEYS],
        *[f"{np.mean(smooth[k]['el']):.1f}" for k in ARRAY_KEYS],
        f"{xy[0]:.3f}", f"{xy[1]:.3f}",
        *[f"{d:.3f}" for d in dists], f"{mean_dist:.3f}"
    ])

    if VERBOSE and (MAX_PRINT is None or i < MAX_PRINT):
        print(f"[blk {i:04d}] t={START_TIME_S+i*CHUNK/RATE:7.2f}s | "
              + " | ".join(f"{k}:az={raw_az[k]:6.1f},el={raw_el[k]:5.1f}"
                           for k in ARRAY_KEYS)
              + f" | <dist>={mean_dist:6.1f} m")
    return True

# ─────── MAIN ───────────────────────────────────────────────────────────
def main():
    wf = {k: wave.open(ARRAYS[k]["wav"].open("rb"), "rb") for k in ARRAY_KEYS}
    for k in ARRAY_KEYS:
        if (wf[k].getnchannels() != ARRAYS[k]["mics"].shape[0]
                or wf[k].getsampwidth() != 2
                or wf[k].getframerate() != RATE):
            sys.exit(f"❌ WAV params mismatch {k}")
    s_f, e_f = int(START_TIME_S*RATE), int(END_TIME_S*RATE)
    total = min(wf[k].getnframes() for k in ARRAY_KEYS)
    if not (0 <= s_f < e_f <= total):
        sys.exit("❌ START/END out of range")
    for k in ARRAY_KEYS: wf[k].setpos(s_f)
    n_blocks = (e_f - s_f)//CHUNK

    with open(CSV_OUTPUT, "w", newline="") as fcsv, \
         FFMpegWriter(fps=10, bitrate=2400).saving(fig, VIDEO_OUTPUT, dpi=150) as vid:
        wr = csv.writer(fcsv)
        wr.writerow(["block","time_s",
                     *[f"raw_az_{k}" for k in ARRAY_KEYS],
                     *[f"raw_el_{k}" for k in ARRAY_KEYS],
                     *[f"smooth_az_{k}" for k in ARRAY_KEYS],
                     *[f"smooth_el_{k}" for k in ARRAY_KEYS],
                     "centroid_x_m","centroid_y_m",
                     *[f"dist_{k}_m" for k in ARRAY_KEYS],
                     "mean_dist_m"])

        plt.show(block=False)
        for i in range(n_blocks):
            if not process_block(i, wf, wr):
                break
            fig.canvas.draw(); fig.canvas.flush_events()
            plt.pause(0.001); vid.grab_frame()

    for k in ARRAY_KEYS: wf[k].close()
    print("✅ CSV  ➔", CSV_OUTPUT.resolve())
    print("✅ MP4  ➔", VIDEO_OUTPUT.resolve())

if __name__ == "__main__":
    main()
