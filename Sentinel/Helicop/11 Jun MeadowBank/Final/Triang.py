#!/usr/bin/env python3
"""
Four-array localisation (6 microphones each), live plotting + on-the-fly MP4
recording.  Now uses a circular moving average for azimuth so that wrap-around
at ±180 deg is seamless.

Revision history
----------------
2025-06-24  • Added dashed red path for raw intersection (traj_raw_line)
2025-06-25  • Replaced linear azimuth smoothing with circular mean
             (store cos/sin components and recover with atan2)
"""

# ── Standard library ────────────────────────────────────────────────────
from collections import deque
import csv, sys, wave
from pathlib import Path
from typing import Dict, List

# ── Third-party ──────────────────────────────────────────────────────────
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from pyproj import Transformer
import pandas as pd

# ── Project-specific helpers ────────────────────────────────────────────
from Utilities.functions import (
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)
from Utilities.mic_geo import (
    mic_6_N_black_thin,
    mic_6_S_orange,
    mic_6_E_orange,
    mic_6_W_black,
)

# ── Paths & geometry ────────────────────────────────────────────────────
ROOT = Path("/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")

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

CSV_OUTPUT = Path("trian_raw.csv")
VIDEO_OUTPUT = Path("trian_raw.mp4")

#AZ_OFFSET = {"N": -44.0, "S": -8.0, "E": 0.0, "W": -28.0}
#EL_OFFSET = {"N": 0.0, "S": 1.0, "E": -1.0, "W": 4.0}

AZ_OFFSET = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
EL_OFFSET = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

RATE, CHUNK = 48_000, int(0.1 * 48_000)        # 100-ms blocks
LOWCUT, HIGHCUT, FILTER_ORDER = 180.0, 2000.0, 5
C_SOUND = 343.0                                 # m s⁻¹
AZIM_RANGE = np.arange(-180, 181, 1)
EL_RANGE = np.arange(10, 51, 1)
SMOOTH_LEN = 1
START_TIME_S, END_TIME_S = 57, 80

VERBOSE, MAX_PRINT = True, 100
DRAW_RAYS, RAY_LEN = True, 600.0

# ── Coordinate helpers ──────────────────────────────────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)
utm2merc = lambda e, n: np.array(_trW2M.transform(*_trU2W.transform(e, n)),
                                 dtype=np.float64)

ARRAY_KEYS = tuple(ARRAYS)
P_merc = {k: utm2merc(*ARRAYS[k]["centre"][:2]) for k in ARRAY_KEYS}
P0 = P_merc["N"]                                # local origin (North array)

# ── Basemap figure ──────────────────────────────────────────────────────
margin = 750.0
centre_xy = np.vstack(list(P_merc.values())).mean(axis=0)
xminM, yminM = centre_xy - margin
xmaxM, ymaxM = centre_xy + margin
img, ext = ctx.bounds2img(xminM, yminM, xmaxM, ymaxM,
                          zoom=19, source=ctx.providers.Esri.WorldImagery)
xminL, xmaxL = ext[0] - P0[0], ext[1] - P0[0]
yminL, ymaxL = ext[2] - P0[1], ext[3] - P0[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=(xminL, xmaxL, yminL, ymaxL), origin="upper", zorder=0)

for k in ARRAY_KEYS:
    xy = P_merc[k] - P0
    ax.scatter(*xy, marker="^", edgecolor="k", s=80)
    ax.text(*(xy + [3, 3]), f" {k}", color="white")

traj_line, = ax.plot([], [], "o-", color="lime", lw=2, label="X̂ smooth")
traj_raw_line, = ax.plot([], [], "o--", color="red", lw=1.5, label="X̂ raw")

# ── Helicopter ground-truth (optional) ──────────────────────────────────
heli_df = pd.read_csv("interpolated_flight_data_100ms.csv")
lon, lat = heli_df["Longitude"].values, heli_df["Latitude"].values
hx, hy = _trW2M.transform(lon, lat)
ax.plot(hx - P0[0], hy - P0[1], "-", color="deepskyblue", lw=1.5,
        label="Helicopter (GT)")

err_text = ax.text(0.97, 0.02, "", transform=ax.transAxes,
                   ha="right", va="bottom",
                   color="white", backgroundcolor="black")
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_aspect("equal")
ax.legend(loc="upper right")

# ── Delay look-up tables ────────────────────────────────────────────────
DELAY_LUT: Dict[str, np.ndarray] = {}
for k in ARRAY_KEYS:
    m = ARRAYS[k]["mics"]
    lut = np.empty((len(AZIM_RANGE), len(EL_RANGE), m.shape[0]), np.int32)
    for ia, az in enumerate(AZIM_RANGE):
        for ie, el in enumerate(EL_RANGE):
            lut[ia, ie] = calculate_delays_for_direction(m, az, el, RATE, C_SOUND)
    DELAY_LUT[k] = lut

# ── Moving-average buffers (circular azimuth!) ──────────────────────────
smooth = {
    k: {
        "cos": deque(maxlen=SMOOTH_LEN),   # cos(az)
        "sin": deque(maxlen=SMOOTH_LEN),   # sin(az)
        "el":  deque(maxlen=SMOOTH_LEN),   # elevation (deg)
    }
    for k in ARRAY_KEYS
}

traj_xy: List[np.ndarray] = []         # smooth path
traj_raw_xy: List[np.ndarray] = []     # raw path

# ── Helper functions ────────────────────────────────────────────────────
def azel2unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (deg) to unit ENU vector."""
    a, e = np.deg2rad([az_deg, el_deg])
    return np.array([np.cos(e) * np.sin(a),
                     np.cos(e) * np.cos(a),
                     np.sin(e)])

def circular_mean_deg(cos_buf: deque, sin_buf: deque) -> float:
    """Return circular mean of angles stored as cos()/sin()."""
    if not cos_buf:
        return np.nan
    C, S = np.mean(cos_buf), np.mean(sin_buf)
    return np.rad2deg(np.arctan2(S, C))  # (-180, 180]

def linear_mean(buf: deque) -> float:
    return np.mean(buf) if buf else np.nan

def ls_triang(orig: np.ndarray, dirs: np.ndarray):
    """Least-squares intersection of skew rays: orig + t·dirs."""
    A = b = np.zeros(3)
    M = np.zeros((3, 3))
    for P, d in zip(orig, dirs):
        Ai = np.eye(3) - np.outer(d, d)
        M += Ai
        b += Ai @ P
    X = np.linalg.solve(M, b)
    rms = np.sqrt(np.mean([
        np.linalg.norm(np.cross(d, X - P)) ** 2 for P, d in zip(orig, dirs)
    ]))
    return X, rms

# ── Per-block processing ────────────────────────────────────────────────
def process_block(i: int, wf: Dict[str, wave.Wave_read], writer: csv.writer) -> bool:
    """Process a single 100-ms block."""
    filt = {}
    # 1. Read & filter
    for k in ARRAY_KEYS:
        nm = ARRAYS[k]["mics"].shape[0]
        data = wf[k].readframes(CHUNK)
        if len(data) < CHUNK * nm * 2:        # EOF
            return False
        s = np.frombuffer(data, np.int16).reshape(-1, nm)
        s = apply_bandpass_filter(s, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
        s /= (np.abs(s).max() or 1.0)
        filt[k] = s

    # 2. Beam-form search → raw az/el
    raw_az, raw_el = {}, {}
    for k in ARRAY_KEYS:
        E = np.zeros((len(AZIM_RANGE), len(EL_RANGE)))
        for ia, az in enumerate(AZIM_RANGE):
            for ie, el in enumerate(EL_RANGE):
                y = apply_beamforming(filt[k], DELAY_LUT[k][ia, ie])
                E[ia, ie] = np.mean(y ** 2)
        ia, ie = np.unravel_index(np.argmax(E), E.shape)
        raw_az[k], raw_el[k] = AZIM_RANGE[ia], EL_RANGE[ie]

        # Store components for circular/linear smoothing
        rad = np.deg2rad(raw_az[k])
        smooth[k]["cos"].append(np.cos(rad))
        smooth[k]["sin"].append(np.sin(rad))
        smooth[k]["el"].append(raw_el[k])

    # 3a. Triangulation (smoothed)
    dirs, orig = [], []
    for k in ARRAY_KEYS:
        saz = circular_mean_deg(smooth[k]["cos"], smooth[k]["sin"]) + AZ_OFFSET[k]
        sel = linear_mean(smooth[k]["el"]) + EL_OFFSET[k]
        dirs.append(azel2unit(saz, sel))
        orig.append(ARRAYS[k]["centre"])
    dirs = np.vstack(dirs)
    orig = np.vstack(orig)
    X, rms = ls_triang(orig, dirs)
    X[2] = max(X[2], 0.0)
    traj_xy.append(utm2merc(*X[:2]) - P0)
    traj_line.set_data(*zip(*traj_xy))

    # 3b. Triangulation (raw)
    dirs_raw = []
    for k in ARRAY_KEYS:
        dirs_raw.append(azel2unit(raw_az[k] + AZ_OFFSET[k],
                                  raw_el[k] + EL_OFFSET[k]))
    dirs_raw = np.vstack(dirs_raw)
    X_raw, _ = ls_triang(orig, dirs_raw)
    X_raw[2] = max(X_raw[2], 0.0)
    traj_raw_xy.append(utm2merc(*X_raw[:2]) - P0)
    traj_raw_line.set_data(*zip(*traj_raw_xy))

    # 4. Scatter last points
    if hasattr(process_block, "pts"):
        process_block.pts.remove()
        process_block.ptraw.remove()
    ps = ax.plot(traj_xy[-1][0], traj_xy[-1][1], "o", color="lime")[0]
    pr = ax.plot(traj_raw_xy[-1][0], traj_raw_xy[-1][1], "o", color="red")[0]
    process_block.pts, process_block.ptraw = ps, pr

    # 5. Optional ray drawing
    if DRAW_RAYS:
        if hasattr(process_block, "hs"):
            for h in process_block.hs:
                h.remove()
        col = {"N": "cyan", "S": "orange", "E": "green", "W": "red"}
        hs = []
        for k, d, P in zip(ARRAY_KEYS, dirs, orig):
            p0 = utm2merc(*P[:2]) - P0
            p1 = utm2merc(*(P[:2] + RAY_LEN * d[:2])) - P0
            h, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                         color=col[k], lw=1.5, alpha=0.7)
            hs.append(h)
        process_block.hs = hs

    # 6. RMS text
    err_text.set_text(f"RMS={rms:.1f} m")

    # 7. CSV log
    writer.writerow([
        i, f"{START_TIME_S + i * CHUNK / RATE:.3f}",
        *[f"{raw_az[k]:.1f}" for k in ARRAY_KEYS],
        *[f"{raw_el[k]:.1f}" for k in ARRAY_KEYS],
        *[f"{circular_mean_deg(smooth[k]['cos'], smooth[k]['sin']) + AZ_OFFSET[k]:.1f}"
          for k in ARRAY_KEYS],
        *[f"{linear_mean(smooth[k]['el']) + EL_OFFSET[k]:.1f}" for k in ARRAY_KEYS],
        f"{traj_xy[-1][0]:.3f}", f"{traj_xy[-1][1]:.3f}", f"{rms:.3f}",
        *[f"{np.linalg.norm(X - ARRAYS[k]['centre']):.3f}" for k in ARRAY_KEYS],
    ])

    # 8. Console
    if VERBOSE and (MAX_PRINT is None or i < MAX_PRINT):
        debug = " | ".join(f"{k}:az={raw_az[k]:6.1f},el={raw_el[k]:5.1f}"
                           for k in ARRAY_KEYS)
        print(f"[blk {i:04d}] t={START_TIME_S + i * CHUNK / RATE:7.2f}s | "
              f"{debug} | RMS={rms:5.2f}")
    return True

# ── Main driver ─────────────────────────────────────────────────────────
def main():
    # 1. Open WAVs and sanity-check
    wf = {k: wave.open(ARRAYS[k]["wav"].open("rb"), "rb") for k in ARRAY_KEYS}
    for k in ARRAY_KEYS:
        if (wf[k].getnchannels() != ARRAYS[k]["mics"].shape[0]
                or wf[k].getsampwidth() != 2
                or wf[k].getframerate() != RATE):
            sys.exit(f"❌ WAV params mismatch in {k}")

    s_f, e_f = int(START_TIME_S * RATE), int(END_TIME_S * RATE)
    total = min(wf[k].getnframes() for k in ARRAY_KEYS)
    if not (0 <= s_f < e_f <= total):
        sys.exit("❌ START/END out of range")
    for k in ARRAY_KEYS:
        wf[k].setpos(s_f)
    n_blocks = (e_f - s_f) // CHUNK

    # 2. CSV + video writers
    with open(CSV_OUTPUT, "w", newline="") as fcsv, \
         FFMpegWriter(fps=10, bitrate=2400).saving(fig, VIDEO_OUTPUT, dpi=150) as vid:

        wr = csv.writer(fcsv)
        wr.writerow([
            "block", "time_s",
            *[f"raw_az_{k}" for k in ARRAY_KEYS],
            *[f"raw_el_{k}" for k in ARRAY_KEYS],
            *[f"smooth_az_{k}" for k in ARRAY_KEYS],
            *[f"smooth_el_{k}" for k in ARRAY_KEYS],
            "local_x_m", "local_y_m", "rms_m",
            *[f"dist_{k}_m" for k in ARRAY_KEYS],
        ])

        plt.show(block=False)
        # 3. Process stream
        for i in range(n_blocks):
            if not process_block(i, wf, wr):
                break
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            vid.grab_frame()

    # 4. Close WAVs
    for k in ARRAY_KEYS:
        wf[k].close()
    print("✅ CSV →", CSV_OUTPUT.resolve())
    print("✅ MP4 →", VIDEO_OUTPUT.resolve())

if __name__ == "__main__":
    main()
