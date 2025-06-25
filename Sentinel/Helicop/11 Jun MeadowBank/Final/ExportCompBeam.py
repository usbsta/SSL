#!/usr/bin/env python3
"""
Four-array FLAT-PLANE localisation (ray–plane intersection, no triangulation)
with a short, fading “comet-tail” visualisation of the estimated helicopter
trajectory (head fully opaque, tail of length TAIL_LEN progressively fading).

Key changes (2025-06-25):
    • Persistent lime line replaced by a scatter with α-fading tail.
    • CSV generation/export removed entirely (unchanged from previous patch).

Author : ChatGPT (OpenAI-o3)
"""

# ── Standard library ────────────────────────────────────────────────────
from collections import deque
import sys, wave
from pathlib import Path
from typing import Dict, List

# ── Third-party ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PathCollection       # NEW
from pyproj import Transformer
import contextily as ctx
import pandas as pd

# ── Project-specific helpers ────────────────────────────────────────────
from Utilities.functions import (
    apply_bandpass_filter,
    apply_beamforming,
    calculate_delays_for_direction,
)
from Utilities.mic_geo import (
    mic_6_N_black_thin,
    mic_6_S_orange,
    mic_6_E_orange,
    mic_6_W_black,
)

# ───────── FILE PATHS & ARRAY GEOMETRY ──────────────────────────────────
ROOT        = Path("/Users/a30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")
FLIGHT_CSV  = Path("interpolated_flight_data_100ms.csv")

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

# ───────── CALIBRATION OFFSETS (deg) ────────────────────────────────────
AZ_OFFSET = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}
EL_OFFSET = {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0}

# ───────── PROCESSING PARAMETERS ────────────────────────────────────────
RATE, CHUNK      = 48_000, int(0.1 * 48_000)      # 100-ms blocks
LOWCUT, HIGHCUT  = 180.0, 2000.0                  # Hz
FILT_ORDER       = 5
C_SOUND          = 343.0                          # m/s
AZIM_RANGE       = np.arange(-180, -30, 1)        # search grid
EL_RANGE         = np.arange(10, 51, 1)
SMOOTH_LEN       = 100                            # moving-average window
TAIL_LEN         = 10                             # length of comet tail (blocks)

START_TIME_S, END_TIME_S = 57, 80
PLANE_ALT_M     = 210.0                           # plane altitude (m AGL)

VIDEO_OUTPUT    = Path("exportcomplete.mp4")

VERBOSE, MAX_PRINT = True, 2300
DRAW_RAYS, RAY_LEN = True, 600.0                  # (ray drawing not shown)

# ───────── COORDINATE TRANSFORMS (UTM56 → WebMerc) ──────────────────────
_trU2W = Transformer.from_crs(32756, 4326, always_xy=True)
_trW2M = Transformer.from_crs(4326, 3857, always_xy=True)
utm2merc = lambda e, n: np.array(_trW2M.transform(*_trU2W.transform(e, n)),
                                 dtype=np.float64)

P_merc = {k: utm2merc(*ARRAYS[k]["centre"][:2]) for k in ARRAY_KEYS}
P0     = P_merc["N"]                              # local origin (Left array)

# ───────── FIGURE & BASEMAP ─────────────────────────────────────────────
margin     = 850.0
centre_xy  = np.vstack(list(P_merc.values())).mean(axis=0)
xminM, yminM = centre_xy - margin
xmaxM, ymaxM = centre_xy + margin
img, ext   = ctx.bounds2img(xminM, yminM, xmaxM, ymaxM,
                            zoom=19, source=ctx.providers.Esri.WorldImagery)
xminL, xmaxL = ext[0] - P0[0], ext[1] - P0[0]
yminL, ymaxL = ext[2] - P0[1], ext[3] - P0[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, extent=(xminL, xmaxL, yminL, ymaxL), origin="upper", zorder=0)

for k in ARRAY_KEYS:
    xy = P_merc[k] - P0
    ax.scatter(*xy, marker="^", edgecolor="k", s=80)
    ax.text(*(xy + [3, 3]), f" {k}", color="white")

# Scatter that will display the comet-tail (initially empty)
tail_scatter: PathCollection = ax.scatter([], [], s=36, edgecolors='none',
                                          zorder=3, label="Helicopter estimate")

# Ground-truth helicopter trajectory
heli_df = pd.read_csv(FLIGHT_CSV)
lon, lat = heli_df["Longitude"].values, heli_df["Latitude"].values
hx, hy   = _trW2M.transform(lon, lat)
ax.plot(hx - P0[0], hy - P0[1], "-", color="deepskyblue",
        lw=1.3, label="Helicopter (GT)")

dist_text = ax.text(0.97, 0.02, "", transform=ax.transAxes,
                    ha="right", va="bottom",
                    color="white", backgroundcolor="black")

ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_aspect("equal")
ax.legend(loc="upper right")

# ───────── DELAY LOOK-UP TABLES ─────────────────────────────────────────
DELAY_LUT: Dict[str, np.ndarray] = {}
for k in ARRAY_KEYS:
    m   = ARRAYS[k]["mics"]
    lut = np.empty((len(AZIM_RANGE), len(EL_RANGE), m.shape[0]), np.int32)
    for ia, az in enumerate(AZIM_RANGE):
        for ie, el in enumerate(EL_RANGE):
            lut[ia, ie] = calculate_delays_for_direction(m, az, el, RATE, C_SOUND)
    DELAY_LUT[k] = lut

# ───────── MOVING-AVERAGE BUFFERS ───────────────────────────────────────
smooth = {
    k: {
        "cos": deque(maxlen=SMOOTH_LEN),    # cos(az)
        "sin": deque(maxlen=SMOOTH_LEN),    # sin(az)
        "el":  deque(maxlen=SMOOTH_LEN),    # elevation (deg)
    }
    for k in ARRAY_KEYS
}

traj_xy: deque = deque(maxlen=TAIL_LEN + 1)        # newest at the right

# ───────── HELPER FUNCTIONS ─────────────────────────────────────────────
def azel2unit(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (deg) to unit ENU vector."""
    a, e = np.deg2rad([az_deg, el_deg])
    return np.array([np.cos(e) * np.sin(a),   # East
                     np.cos(e) * np.cos(a),   # North
                     np.sin(e)])              # Up

def ray_plane_intersect(P: np.ndarray, d: np.ndarray, z_plane: float):
    """Intersect ray X(t)=P+t·d with horizontal plane z=z_plane."""
    if abs(d[2]) < 1e-6:
        return None
    t = (z_plane - P[2]) / d[2]
    return None if t <= 0 else P + t * d

def circular_mean_deg(cos_buf: deque, sin_buf: deque) -> float:
    """Circular mean of angles stored as cos() and sin() components."""
    if not cos_buf:
        return np.nan
    C, S = np.mean(cos_buf), np.mean(sin_buf)
    return np.rad2deg(np.arctan2(S, C))       # (-180, 180]

def linear_mean(buf: deque) -> float:
    return np.mean(buf) if buf else np.nan

# ───────── PROCESS A SINGLE AUDIO BLOCK ─────────────────────────────────
def process_block(i: int, wf: Dict[str, wave.Wave_read]) -> bool:
    # 1. Read and band-pass filter the current 100-ms chunk from each WAV
    filt = {}
    for k in ARRAY_KEYS:
        nm   = ARRAYS[k]["mics"].shape[0]
        data = wf[k].readframes(CHUNK)
        if len(data) < CHUNK * nm * 2:        # end of file
            return False
        s = np.frombuffer(data, np.int16).reshape(-1, nm)
        s = apply_bandpass_filter(s, LOWCUT, HIGHCUT, RATE, order=FILT_ORDER)
        s /= (np.abs(s).max() or 1.0)         # normalise
        filt[k] = s

    # 2. Beam-forming search → raw az/el
    raw_az, raw_el = {}, {}
    for k in ARRAY_KEYS:
        E = np.zeros((len(AZIM_RANGE), len(EL_RANGE)))
        for ia, az in enumerate(AZIM_RANGE):
            for ie, el in enumerate(EL_RANGE):
                y = apply_beamforming(filt[k], DELAY_LUT[k][ia, ie])
                E[ia, ie] = np.mean(y**2)
        ia, ie = np.unravel_index(np.argmax(E), E.shape)
        raw_az[k], raw_el[k] = AZIM_RANGE[ia], EL_RANGE[ie]

        # Update smoothing buffers
        rad = np.deg2rad(raw_az[k])
        smooth[k]["cos"].append(np.cos(rad))
        smooth[k]["sin"].append(np.sin(rad))
        smooth[k]["el"].append(raw_el[k])

    # 3. SMOOTH rays → plane intersection
    P_smooth = []
    for k in ARRAY_KEYS:
        az_s = circular_mean_deg(smooth[k]["cos"], smooth[k]["sin"]) + AZ_OFFSET[k]
        el_s = linear_mean(smooth[k]["el"]) + EL_OFFSET[k]
        d_s  = azel2unit(az_s, el_s)
        Pint = ray_plane_intersect(ARRAYS[k]["centre"], d_s, PLANE_ALT_M)
        if Pint is None:
            return True
        P_smooth.append(Pint)
    centroid = np.mean(P_smooth, axis=0)
    xy = utm2merc(*centroid[:2]) - P0
    traj_xy.append(xy)                       # newest head position

    # -------- Comet-tail update -----------------------------------------
    pts = np.array(traj_xy)                  # shape (≤TAIL_LEN+1, 2)
    alphas = np.linspace(0.0, 1.0, len(pts)) # α: 0→oldest, 1→head
    colors = np.zeros((len(pts), 4))
    colors[:, 1] = 1.0                       # pure green
    colors[:, 3] = alphas                    # set α
    tail_scatter.set_offsets(pts)
    tail_scatter.set_facecolors(colors)
    # --------------------------------------------------------------------

    # 4. Distances to each array
    dists      = [np.linalg.norm(centroid - ARRAYS[k]["centre"]) for k in ARRAY_KEYS]
    mean_dist  = np.mean(dists)
    dist_text.set_text(f"Mean dist = {mean_dist:.1f} m")

    # 5. Optional console debug
    if VERBOSE and (MAX_PRINT is None or i < MAX_PRINT):
        dbg = " | ".join(f"{k}:az={raw_az[k]:6.1f},el={raw_el[k]:5.1f}"
                         for k in ARRAY_KEYS)
        print(f"[blk {i:04d}] t={START_TIME_S + i * CHUNK / RATE:7.2f}s |"
              f" {dbg} | <dist>={mean_dist:6.1f} m")
    return True

# ───────── MAIN ─────────────────────────────────────────────────────────
def main():
    # Open WAVs
    wf = {k: wave.open(ARRAYS[k]["wav"].open("rb"), "rb") for k in ARRAY_KEYS}
    # Sanity-check parameters
    for k in ARRAY_KEYS:
        if (wf[k].getnchannels() != ARRAYS[k]["mics"].shape[0]
                or wf[k].getsampwidth() != 2
                or wf[k].getframerate() != RATE):
            sys.exit(f"❌ WAV params mismatch in {k}")
    s_f, e_f = int(START_TIME_S * RATE), int(END_TIME_S * RATE)
    total_frames = min(wf[k].getnframes() for k in ARRAY_KEYS)
    if not (0 <= s_f < e_f <= total_frames):
        sys.exit("❌ START_TIME_S / END_TIME_S out of range")
    for k in ARRAY_KEYS:
        wf[k].setpos(s_f)
    n_blocks = (e_f - s_f) // CHUNK

    # MP4 writer
    with FFMpegWriter(fps=10, bitrate=2400).saving(fig, VIDEO_OUTPUT, dpi=150) as vid:
        plt.show(block=False)
        for i in range(n_blocks):
            if not process_block(i, wf):
                break
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            vid.grab_frame()

    for k in ARRAY_KEYS:
        wf[k].close()
    print("✅ MP4 ➔", VIDEO_OUTPUT.resolve())

if __name__ == "__main__":
    main()
