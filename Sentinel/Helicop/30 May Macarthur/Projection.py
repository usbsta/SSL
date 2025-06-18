#!/usr/bin/env python3
"""
Real-time “flat-plane” localisation debug tool
==============================================

• Two 8-mic arrays (Left & Right) process multichannel WAV files.
• For each 100 ms block:
    – Beamform to find max-energy azimuth/elevation for each array
    – Convert to a unit ENU ray
    – Project that ray to a *fixed* altitude plane (700 m)
    – Plot the resulting XY point for each array + their midpoint
    – Print / CSV-log raw & smoothed angles, midpoint distance to arrays

"""

import numpy as np
import wave
import csv
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ── External DSP helpers ──────────────────────────────────────────────────
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ── GLOBAL SETTINGS ───────────────────────────────────────────────────────
LEFT_WAV_PATH  = "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/30_05_25/1_Left/Macarthur/left.wav"
RIGHT_WAV_PATH = "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/30_05_25/2_Right/Macarthur/right.wav"

RATE            = 48_000           # Hz
CHUNK           = int(0.1 * RATE)  # 100 ms per processing block
LOWCUT, HIGHCUT = 180.0, 2_000.0   # Hz
FILTER_ORDER    = 5
C               = 343.0            # Speed of sound (m s⁻¹)

AZIMUTH_RANGE   = np.arange(-180, 181, 1)   # °
ELEV_RANGE      = np.arange(0, 91, 1)       # °

# UTM coordinates (zone 56 S) of each array, metres
P_RIGHT = np.array([287322.3, 6228859.5, 328 * 0.3048], dtype=np.float64)
P_LEFT  = np.array([287307.7, 6228856.9, 328 * 0.3048], dtype=np.float64)

TARGET_ALT_M    = 700.0 + 328 * 0.3048           # <-- fixed altitude of aircraft (m)
CSV_PATH        = "flat_plane_debug.csv"

START_TIME, END_TIME = 3438.0, 3453.0  # s
SMOOTH_LEN = 10                         # moving-average length (blocks)

# ── PRECOMPUTE DELAYS (geometry identical for both arrays) ────────────────
mic_pos   = microphone_positions_8_helicop()
NUM_MICS  = mic_pos.shape[0]
delays = np.empty((len(AZIMUTH_RANGE), len(ELEV_RANGE), NUM_MICS), np.int32)
for ia, az in enumerate(AZIMUTH_RANGE):
    for ie, el in enumerate(ELEV_RANGE):
        delays[ia, ie] = calculate_delays_for_direction(
            mic_pos, az, el, RATE, C
        )

# ── Utility functions ────────────────────────────────────────────────────
def azel_to_unit_vec(az_deg: float, el_deg: float) -> np.ndarray:
    """Convert azimuth/elevation (° ENU) ➜ unit vector (East, North, Up)."""
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),   # East
        np.cos(el) * np.cos(az),   # North
        np.sin(el)                 # Up
    ], dtype=np.float64)

def project_to_altitude(P: np.ndarray, d: np.ndarray,
                        target_z: float) -> np.ndarray | None:
    """
    Ray: X(t) = P + t·d  ⟂  Find intersection with horizontal plane z = target_z.
    Returns None if the ray never reaches that altitude in the positive direction.
    """
    if abs(d[2]) < 1e-6:
        return None  # Ray is parallel to the plane
    t = (target_z - P[2]) / d[2]
    return None if t <= 0 else P + t * d

# ── Matplotlib live figure setup ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")
ax.set_title("Aircraft XY trajectory on Z = 700 m plane")
ax.grid(True)

# ── Static markers for microphone-array positions ────────────────────────
# Scatter both arrays as black triangles and label them
array_scatter = ax.scatter(
    [P_LEFT[0], P_RIGHT[0]],           # X = Easting
    [P_LEFT[1], P_RIGHT[1]],           # Y = Northing
    marker="^",
    color="k",
    s=80,
    zorder=5,
    label="Array positions"
)

# Optional: annotate with text so you can see which is which
ax.text(P_LEFT[0],  P_LEFT[1],  "  Left",  va="bottom", ha="left")
ax.text(P_RIGHT[0], P_RIGHT[1], "  Right", va="bottom", ha="left")

left_line,  = ax.plot([], [], "o-", lw=1.2, label="Left-array projection")
right_line, = ax.plot([], [], "o-", lw=1.2, label="Right-array projection")
mid_line,   = ax.plot([], [], "o-", lw=2.5, label="Midpoint (best guess)")

ax.legend(loc="upper left")

# storage for trajectories
traj_left, traj_right, traj_mid = [], [], []

# ── MAIN processing routine ──────────────────────────────────────────────
def process_block(block_idx, wf_left, wf_right, writer):
    """Beamform, project to 700 m, update plot & CSV for one audio block."""
    frames_left  = wf_left.readframes(CHUNK)
    frames_right = wf_right.readframes(CHUNK)
    if min(len(frames_left), len(frames_right)) < CHUNK * NUM_MICS * 2:
        return False  # EOF

    audio_left  = np.frombuffer(frames_left,  np.int16).reshape(-1, NUM_MICS)
    audio_right = np.frombuffer(frames_right, np.int16).reshape(-1, NUM_MICS)

    # Band-pass, normalise
    filt_left  = apply_bandpass_filter(audio_left,  LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
    filt_right = apply_bandpass_filter(audio_right, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
    for sig in (filt_left, filt_right):
        m = np.abs(sig).max()
        if m != 0:
            sig /= m

    # ---------- Beamform left ----------
    energy_L = np.zeros((len(AZIMUTH_RANGE), len(ELEV_RANGE)))
    for ia, az in enumerate(AZIMUTH_RANGE):
        for ie, el in enumerate(ELEV_RANGE):
            y = apply_beamforming(filt_left, delays[ia, ie])
            energy_L[ia, ie] = np.sum(y ** 2) / CHUNK
    (ia_max, ie_max) = np.unravel_index(np.argmax(energy_L), energy_L.shape)
    raw_az_L, raw_el_L = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ---------- Beamform right ----------
    energy_R = np.zeros_like(energy_L)
    for ia, az in enumerate(AZIMUTH_RANGE):
        for ie, el in enumerate(ELEV_RANGE):
            y = apply_beamforming(filt_right, delays[ia, ie])
            energy_R[ia, ie] = np.sum(y ** 2) / CHUNK
    (ia_max, ie_max) = np.unravel_index(np.argmax(energy_R), energy_R.shape)
    raw_az_R, raw_el_R = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ---------- Moving average ----------
    if block_idx == 0:
        process_block.buffers = {
            "azL": deque([raw_az_L], maxlen=SMOOTH_LEN),
            "elL": deque([raw_el_L], maxlen=SMOOTH_LEN),
            "azR": deque([raw_az_R], maxlen=SMOOTH_LEN),
            "elR": deque([raw_el_R], maxlen=SMOOTH_LEN),
        }
    else:
        for key, val in zip(("azL", "elL", "azR", "elR"),
                            (raw_az_L, raw_el_L, raw_az_R, raw_el_R)):
            process_block.buffers[key].append(val)

    smooth_az_L = np.mean(process_block.buffers["azL"])
    smooth_el_L = np.mean(process_block.buffers["elL"])
    smooth_az_R = np.mean(process_block.buffers["azR"])
    smooth_el_R = np.mean(process_block.buffers["elR"])

    # ---------- Project rays to altitude ----------
    dL = azel_to_unit_vec(smooth_az_L, smooth_el_L)
    dR = azel_to_unit_vec(smooth_az_R, smooth_el_R)
    proj_L = project_to_altitude(P_LEFT,  dL, TARGET_ALT_M)
    proj_R = project_to_altitude(P_RIGHT, dR, TARGET_ALT_M)

    # If either projection fails, skip this block
    if proj_L is None or proj_R is None:
        return True

    midpoint = 0.5 * (proj_L + proj_R)
    traj_left.append(proj_L[:2])
    traj_right.append(proj_R[:2])
    traj_mid.append(midpoint[:2])

    # ---------- Update plot ----------
    left_line.set_data(*zip(*traj_left))
    right_line.set_data(*zip(*traj_right))
    mid_line.set_data(*zip(*traj_mid))
    # Auto-scale view
    ax.relim(), ax.autoscale_view()

    # ---------- Distances ----------
    dist_L = np.linalg.norm(midpoint - P_LEFT)
    dist_R = np.linalg.norm(midpoint - P_RIGHT)

    curr_time = START_TIME + block_idx * (CHUNK / RATE)
    print(f"Blk {block_idx:03d} | t={curr_time:7.2f}s | "
          f"Az/El_L=({smooth_az_L:6.1f},{smooth_el_L:4.1f})  "
          f"Az/El_R=({smooth_az_R:6.1f},{smooth_el_R:4.1f}) | "
          f"XY_mid=({midpoint[0]:.1f},{midpoint[1]:.1f}) m | "
          f"dL={dist_L:7.1f} m  dR={dist_R:7.1f} m")

    writer.writerow([
        block_idx, f"{curr_time:.3f}",
        f"{raw_az_L:.1f}", f"{raw_el_L:.1f}",
        f"{raw_az_R:.1f}", f"{raw_el_R:.1f}",
        f"{smooth_az_L:.1f}", f"{smooth_el_L:.1f}",
        f"{smooth_az_R:.1f}", f"{smooth_el_R:.1f}",
        f"{midpoint[0]:.3f}", f"{midpoint[1]:.3f}",
        f"{dist_L:.3f}", f"{dist_R:.3f}"
    ])

    return True

# ── Entry point ──────────────────────────────────────────────────────────
def main():
    # --- WAV sanity checks ------------------------------------------------
    wfL, wfR = wave.open(LEFT_WAV_PATH, "rb"), wave.open(RIGHT_WAV_PATH, "rb")
    for wf in (wfL, wfR):
        if wf.getnchannels() != NUM_MICS or wf.getsampwidth() != 2 or wf.getframerate() != RATE:
            raise RuntimeError("WAV file parameters do not match array configuration.")

    total_frames = min(wfL.getnframes(), wfR.getnframes())
    start_f, end_f = int(START_TIME * RATE), int(END_TIME * RATE)
    if not (0 <= start_f < end_f <= total_frames):
        raise RuntimeError("Start/End time out of range.")
    wfL.setpos(start_f), wfR.setpos(start_f)
    max_blocks = (end_f - start_f) // CHUNK

    # --- CSV header -------------------------------------------------------
    csv_file = open(CSV_PATH, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "block", "time_s",
        "raw_az_L", "raw_el_L", "raw_az_R", "raw_el_R",
        "smooth_az_L", "smooth_el_L", "smooth_az_R", "smooth_el_R",
        "mid_easting_m", "mid_northing_m",
        "dist_L_m", "dist_R_m"
    ])

    # --- Animation loop ---------------------------------------------------
    def _update(frame_idx):
        if frame_idx >= max_blocks:
            plt.close(fig)
            return
        more = process_block(frame_idx, wfL, wfR, writer)
        if not more:
            plt.close(fig)  # EOF

    ani = FuncAnimation(fig, _update, frames=max_blocks, interval=1, repeat=False)
    plt.show()

    csv_file.close()
    wfL.close(), wfR.close()
    print(f"✅ CSV saved to '{CSV_PATH}'")

if __name__ == "__main__":
    main()
