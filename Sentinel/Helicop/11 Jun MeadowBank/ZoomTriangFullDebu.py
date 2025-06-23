#!/usr/bin/env python3
"""
Four-array localisation (6 mics each) with per-array calibration offsets,
least-squares triangulation, basemap overlay and CSV/MP4 export.

• Arrays: N, S, E, W – geometry imported from Utilities.mic_geo
• Verbose console output for the first MAX_PRINT blocks
• Optional 2-D ray visualisation in the map
"""

# ── Standard library ────────────────────────────────────────────────────
from collections import deque
import csv
import sys
import wave
from pathlib import Path
from typing import List, Dict

# ── Third-party ─────────────────────────────────────────────────────────
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pyproj import Transformer

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

# ───────────────────────── Input configuration ──────────────────────────
ROOT = Path(
    "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25"
)

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

CSV_OUTPUT = Path("triangulation_full_NoSmooth.csv")
VIDEO_OUTPUT = Path("triangulation_full_NoSmooth.mp4")

AZ_OFFSET = {"N": -44.0, "S": -8.0, "E": 0.0, "W": -28.0}
EL_OFFSET = {"N": 0.0, "S": 1.0, "E": -1.0, "W": 4.0}

# ─────────────── Global BF / audio parameters ───────────────────────────
RATE, CHUNK = 48_000, int(0.1 * 48_000)
LOWCUT, HIGHCUT, FILTER_ORDER = 180.0, 2_000.0, 5
C_SOUND = 343.0
AZIM_RANGE = np.arange(-180, 181, 1)
EL_RANGE = np.arange(10, 51, 1)
SMOOTH_LEN = 1
START_TIME_S, END_TIME_S = 57, 70

# ─────────────── Verbosity / debugging flags ────────────────────────────
VERBOSE = True        # Toggle console print-outs
MAX_PRINT = 4        # None → print all blocks, else first N blocks
DRAW_RAYS = True      # Toggle orange 2-D rays on the map
RAY_LEN = 100.0       # metres in the EN plane for visualisation

# ─────────────── Coordinate transform helpers ───────────────────────────
_tr_utm2wgs = Transformer.from_crs(32756, 4326, always_xy=True)
_tr_wgs2merc = Transformer.from_crs(4326, 3857, always_xy=True)


def utm_to_merc(e: float, n: float) -> np.ndarray:
    lon, lat = _tr_utm2wgs.transform(e, n)
    return np.array(_tr_wgs2merc.transform(lon, lat), dtype=np.float64)


# ───────────── Basemap & figure (origin = North array) ───────────────────
ARRAY_KEYS: tuple[str, ...] = tuple(ARRAYS.keys())
P_merc = {k: utm_to_merc(*ARRAYS[k]["centre"][:2]) for k in ARRAY_KEYS}
P0_merc = P_merc["N"]

extent_margin = 750.0
all_xy = np.vstack(list(P_merc.values()))
centre_xy = all_xy.mean(axis=0)
xminM, yminM = centre_xy - extent_margin
xmaxM, ymaxM = centre_xy + extent_margin

basemap_img, extent = ctx.bounds2img(
    xminM, yminM, xmaxM, ymaxM, zoom=19, source=ctx.providers.Esri.WorldImagery
)
xminL, xmaxL = extent[0] - P0_merc[0], extent[1] - P0_merc[0]
yminL, ymaxL = extent[2] - P0_merc[1], extent[3] - P0_merc[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(
    basemap_img,
    extent=(xminL, xmaxL, yminL, ymaxL),
    interpolation="bilinear",
    origin="upper",
    zorder=0,
)
for k in ARRAY_KEYS:
    xy = P_merc[k] - P0_merc
    ax.scatter(*xy, marker="^", edgecolor="k", s=80, label=f"Array {k}")
    ax.text(*(xy + np.array([3, 3])), f" {k}", color="white")

traj_line, = ax.plot([], [], "o-", color="lime", lw=2.0, label="X̂")
err_text = ax.text(
    0.97,
    0.02,
    "",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    color="white",
    backgroundcolor="black",
)
ax.legend(loc="upper left")
ax.set_aspect("equal")

# ───────────── Delay LUT per array (depends on geometry) ────────────────
DELAY_LUT: Dict[str, np.ndarray] = {}
for k in ARRAY_KEYS:
    mics = ARRAYS[k]["mics"]
    lut = np.empty((len(AZIM_RANGE), len(EL_RANGE), mics.shape[0]), np.int32)
    for ia, az in enumerate(AZIM_RANGE):
        for ie, el in enumerate(EL_RANGE):
            lut[ia, ie] = calculate_delays_for_direction(
                mics, az, el, RATE, C_SOUND
            )
    DELAY_LUT[k] = lut

# ───────────── Smoothing buffers & runtime containers ───────────────────
smooth = {
    k: {"az": deque(maxlen=SMOOTH_LEN), "el": deque(maxlen=SMOOTH_LEN)}
    for k in ARRAY_KEYS
}
traj_xy: List[np.ndarray] = []

# ───────────── Helper math functions ────────────────────────────────────
def azel_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array(
        [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
        dtype=np.float64,
    )


def ls_triangulation(origins: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares intersection of multiple rays."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for P, d in zip(origins, dirs):
        Ai = np.eye(3) - np.outer(d, d)
        A += Ai
        b += Ai @ P
    X_hat = np.linalg.solve(A, b)
    rms = np.sqrt(
        np.mean([np.linalg.norm(np.cross(d, X_hat - P)) ** 2 for P, d in zip(origins, dirs)])
    )
    return X_hat, float(rms)


# ───────────── Core processing per block ────────────────────────────────
def process_block(idx: int, wf_dict: Dict[str, wave.Wave_read], writer: csv.writer) -> bool:
    # 1. Read and BP-filter
    filt: Dict[str, np.ndarray] = {}
    for k in ARRAY_KEYS:
        n_mic = ARRAYS[k]["mics"].shape[0]
        frames = wf_dict[k].readframes(CHUNK)
        if len(frames) < CHUNK * n_mic * 2:
            return False  # EOF
        sig = np.frombuffer(frames, np.int16).reshape(-1, n_mic)
        sig = apply_bandpass_filter(sig, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
        peak = np.abs(sig).max()
        if peak:
            sig /= peak
        filt[k] = sig

    # 2. Beamforming scan
    raw_az, raw_el = {}, {}
    for k in ARRAY_KEYS:
        energy = np.zeros((len(AZIM_RANGE), len(EL_RANGE)))
        for ia in range(len(AZIM_RANGE)):
            for ie in range(len(EL_RANGE)):
                y = apply_beamforming(filt[k], DELAY_LUT[k][ia, ie])
                energy[ia, ie] = np.sum(y ** 2) / CHUNK
        ia_max, ie_max = np.unravel_index(np.argmax(energy), energy.shape)
        raw_az[k], raw_el[k] = AZIM_RANGE[ia_max], EL_RANGE[ie_max]
        smooth[k]["az"].append(raw_az[k])
        smooth[k]["el"].append(raw_el[k])

    # 3. Smoothing + offsets → rays
    dirs, origins = [], []
    for k in ARRAY_KEYS:
        saz = np.mean(smooth[k]["az"]) + AZ_OFFSET[k]
        sel = np.mean(smooth[k]["el"]) + EL_OFFSET[k]
        dirs.append(azel_to_unit(saz, sel))
        origins.append(ARRAYS[k]["centre"])
    dirs, origins = np.vstack(dirs), np.vstack(origins)

    # 4. Triangulation
    X_hat, rms = ls_triangulation(origins, dirs)
    if X_hat[2] < 0.0:
        X_hat[2] = 0.0
    local_xy = utm_to_merc(*X_hat[:2]) - P0_merc
    traj_xy.append(local_xy)

    # 5. Plot update
    traj_line.set_data(*zip(*traj_xy))
    err_text.set_text(f"RMS error = {rms:.1f} m")

    # — Optional 2-D ray drawing —
    if DRAW_RAYS:
        # Remove previous ray segments
        if hasattr(process_block, "ray_handles"):
            for hndl in process_block.ray_handles:
                hndl.remove()
        ray_handles = []
        for d, P in zip(dirs, origins):
            end_xy = utm_to_merc(*(P[:2] + RAY_LEN * d[:2])) - P0_merc
            h, = ax.plot(
                [local_xy[0], end_xy[0]],
                [local_xy[1], end_xy[1]],
                color="orange",
                lw=1.0,
                alpha=0.6,
                zorder=4,
            )
            ray_handles.append(h)
        process_block.ray_handles = ray_handles

    # 6. CSV write
    writer.writerow(
        [
            idx,
            f"{START_TIME_S + idx * CHUNK / RATE:.3f}",
            *[f"{raw_az[k]:.1f}" for k in ARRAY_KEYS],
            *[f"{raw_el[k]:.1f}" for k in ARRAY_KEYS],
            *[
                f"{np.mean(smooth[k]['az']) + AZ_OFFSET[k]:.1f}"
                for k in ARRAY_KEYS
            ],
            *[
                f"{np.mean(smooth[k]['el']) + EL_OFFSET[k]:.1f}"
                for k in ARRAY_KEYS
            ],
            f"{local_xy[0]:.3f}",
            f"{local_xy[1]:.3f}",
            f"{rms:.3f}",
            *[
                f"{np.linalg.norm(X_hat - ARRAYS[k]['centre']):.3f}"
                for k in ARRAY_KEYS
            ],
        ]
    )

    # 7. Verbose console output
    if VERBOSE and (MAX_PRINT is None or idx < MAX_PRINT):
        az_raw_str = " | ".join(
            f"{k}: az={raw_az[k]:6.1f}°, el={raw_el[k]:5.1f}°" for k in ARRAY_KEYS
        )
        az_smooth_str = " | ".join(
            f"{k}: az={(np.mean(smooth[k]['az']) + AZ_OFFSET[k]):6.1f}°, "
            f"el={(np.mean(smooth[k]['el']) + EL_OFFSET[k]):5.1f}°"
            for k in ARRAY_KEYS
        )
        dist_str = " | ".join(
            f"{k}: {np.linalg.norm(X_hat - ARRAYS[k]['centre']):6.1f} m"
            for k in ARRAY_KEYS
        )
        print(
            f"[blk {idx:04d}] t="
            f"{START_TIME_S + idx * CHUNK / RATE:7.2f}s | raw → {az_raw_str} | "
            f"smooth → {az_smooth_str} | "
            f"X̂=({X_hat[0]:.1f},{X_hat[1]:.1f},{X_hat[2]:.1f}) m | "
            f"RMS={rms:5.2f} m | dists → {dist_str}"
        )

    return True


# ───────────── Main entry point ─────────────────────────────────────────
def main() -> None:
    # Open WAVs
    wf: Dict[str, wave.Wave_read] = {
        k: wave.open(ARRAYS[k]["wav"].open("rb"), "rb") for k in ARRAY_KEYS
    }
    for k in ARRAY_KEYS:
        if (
            wf[k].getnchannels() != ARRAYS[k]["mics"].shape[0]
            or wf[k].getsampwidth() != 2
            or wf[k].getframerate() != RATE
        ):
            sys.exit(f"❌ WAV parameters mismatch for array {k}")

    start_f = int(START_TIME_S * RATE)
    end_f = int(END_TIME_S * RATE)
    total_frames = min(wf[k].getnframes() for k in ARRAY_KEYS)
    if not (0 <= start_f < end_f <= total_frames):
        sys.exit("❌ Invalid START/END time.")
    for k in ARRAY_KEYS:
        wf[k].setpos(start_f)
    n_blocks = (end_f - start_f) // CHUNK

    # CSV file
    with open(CSV_OUTPUT, "w", newline="") as f_csv:
        wr = csv.writer(f_csv)
        wr.writerow(
            [
                "block",
                "time_s",
                *[f"raw_az_{k}" for k in ARRAY_KEYS],
                *[f"raw_el_{k}" for k in ARRAY_KEYS],
                *[f"smooth_az_{k}" for k in ARRAY_KEYS],
                *[f"smooth_el_{k}" for k in ARRAY_KEYS],
                "local_x_m",
                "local_y_m",
                "rms_m",
                *[f"dist_{k}_m" for k in ARRAY_KEYS],
            ]
        )

        # Animation
        INTRO_FRAMES, INIT_R, FINAL_R = 30, 50, 700

        def _update(frame: int):
            if frame < INTRO_FRAMES:
                r = INIT_R + (FINAL_R - INIT_R) * frame / INTRO_FRAMES
                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)
                return traj_line, err_text
            blk = frame - INTRO_FRAMES
            ok = process_block(blk, wf, wr)
            ax.set_xlim(-FINAL_R, FINAL_R)
            ax.set_ylim(-FINAL_R, FINAL_R)
            if not ok:
                plt.close(fig)
            return traj_line, err_text

        ani = FuncAnimation(
            fig,
            _update,
            frames=INTRO_FRAMES + n_blocks,
            interval=1,
            blit=True,
            repeat=False,
        )
        ani.save(VIDEO_OUTPUT, writer=FFMpegWriter(fps=10, bitrate=2400))
        plt.show()

    for k in ARRAY_KEYS:
        wf[k].close()
    print(f"✅ CSV saved → {CSV_OUTPUT.resolve()}")
    print(f"✅ MP4 saved → {VIDEO_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
