#!/usr/bin/env python3
"""
Real-time “flat-plane” localisation debug tool – satellite overlay
==================================================================

* Two 8-mic arrays (“Left” & “Right”) process multichannel WAV files.
* Every 100 ms block:
    1. Beamform to find the max-energy azimuth/elevation for each array.
    2. Convert Az/El → unit ENU vector.
    3. Project that ray to a **fixed** altitude plane (700 m AGL).
    4. Convert UTM → Web-Mercator and draw XY tracks on a live satellite map.
    5. Print & CSV-log raw/smoothed angles plus midpoint distances.


"""

# ── Standard library ────────────────────────────────────────────────────
from collections import deque
import csv
import wave
import time
import sys
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyproj import Transformer
import contextily as ctx

# ── External DSP helpers (YOUR package / module) ────────────────────────
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ────────────────────────────────────────────────────────────────────────
#                          GLOBAL SETTINGS
# ────────────────────────────────────────────────────────────────────────
# ─ File paths ─
LEFT_WAV_PATH  = Path("/Users/30068385/OneDrive - Western Sydney University/"
                      "recordings/Helicop/30_05_25/1_Left/Macarthur/left.wav")
RIGHT_WAV_PATH = Path("/Users/30068385/OneDrive - Western Sydney University/"
                      "recordings/Helicop/30_05_25/2_Right/Macarthur/right.wav")
CSV_PATH       = Path("flat_plane_debug.csv")

# ─ Audio & beamforming ─
RATE            = 48_000            # Hz
CHUNK           = int(0.1 * RATE)   # 100 ms per processing block
LOWCUT, HIGHCUT = 180.0, 2_000.0    # Band-pass in Hz
FILTER_ORDER    = 5                 # Butterworth order
C               = 343.0             # Speed of sound (m s⁻¹)

AZIMUTH_RANGE   = np.arange(-180, 181, 4)   # deg
ELEV_RANGE      = np.arange(0, 91, 4)       # deg

# ─ Geometry (UTM 56 S, EPSG:32756) ─
P_RIGHT = np.array([287322.3, 6228859.5, 328 * 0.3048], dtype=np.float64)
P_LEFT  = np.array([287307.7, 6228856.9, 328 * 0.3048], dtype=np.float64)

TARGET_ALT_M = 700.0 + 328 * 0.3048        # Aircraft altitude (m ASL)
SMOOTH_LEN   = 70                          # Moving-average window (blocks)

#START_TIME_S = 3439.0
START_TIME_S = 3436.6  # Playback start (s in WAV)
#END_TIME_S   = 3452.5
END_TIME_S   = 3453.5        # Playback end   (s in WAV)

# ─ Coordinate transformers ──────────────────────────────────────────────
# UTM 56 S (EPSG:32756) → WGS-84 (EPSG:4326)
_utm_to_wgs84 = Transformer.from_crs(32756, 4326, always_xy=True)
# WGS-84 → Web-Mercator (EPSG:3857)
_wgs84_to_merc = Transformer.from_crs(4326, 3857, always_xy=True)


def utm56s_to_merc(easting: float, northing: float) -> np.ndarray:
    """Convert UTM zone 56 S → Web-Mercator (x, y in m)."""
    lon, lat = _utm_to_wgs84.transform(easting, northing)
    x, y = _wgs84_to_merc.transform(lon, lat)
    return np.array([x, y], dtype=np.float64)


# ── Functions ───────────────────────────────────────────────────────────
def azel_to_unit_vec(az_deg: float, el_deg: float) -> np.ndarray:
    """Az/El (deg ENU) → unit vector (East, North, Up)."""
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array(
        [np.cos(el) * np.sin(az),   # East
         np.cos(el) * np.cos(az),   # North
         np.sin(el)],               # Up
        dtype=np.float64,
    )


def project_to_altitude(P: np.ndarray, d: np.ndarray,
                        target_z: float) -> np.ndarray | None:
    """
    Intersect ray X(t)=P+t·d with horizontal plane z=target_z.
    Returns None if the ray never reaches that altitude for t>0.
    """
    if abs(d[2]) < 1e-6:          # Ray parallel to plane
        return None
    t = (target_z - P[2]) / d[2]
    return None if t <= 0 else P + t * d


# ── PRECOMPUTE DELAYS (identical geometry for both arrays) ──────────────
mic_pos = microphone_positions_8_helicop()           # (M × 3)
NUM_MICS = mic_pos.shape[0]

delays = np.empty((len(AZIMUTH_RANGE), len(ELEV_RANGE), NUM_MICS),
                  dtype=np.int32)
for ia, az in enumerate(AZIMUTH_RANGE):
    for ie, el in enumerate(ELEV_RANGE):
        delays[ia, ie] = calculate_delays_for_direction(
            mic_pos, az, el, RATE, C
        )

# ── Matplotlib figure and satellite basemap ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

# Convert array positions once → Web-Mercator
P_LEFT_MERC = utm56s_to_merc(P_LEFT[0], P_LEFT[1])
P_RIGHT_MERC = utm56s_to_merc(P_RIGHT[0], P_RIGHT[1])

# Bounding box (±1.5 km around the arrays)
margin_m = 700
xmin = min(P_LEFT_MERC[0], P_RIGHT_MERC[0]) - margin_m
xmax = max(P_LEFT_MERC[0], P_RIGHT_MERC[0]) + margin_m
ymin = min(P_LEFT_MERC[1], P_RIGHT_MERC[1]) - margin_m
ymax = max(P_LEFT_MERC[1], P_RIGHT_MERC[1]) + margin_m
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Draw satellite tiles (once)
ctx.add_basemap(
    ax,
    crs="EPSG:3857",
    source=ctx.providers.Esri.WorldImagery,   # Free aerial imagery
    zoom=16,                                  # Adjust for your area
)
ax.set_title("Aircraft track projected on Z = 700 m plane")
ax.set_xlabel("Web-Mercator X [m]")
ax.set_ylabel("Web-Mercator Y [m]")

# Plot static array markers
array_scatter = ax.scatter(
    [P_LEFT_MERC[0], P_RIGHT_MERC[0]],
    [P_LEFT_MERC[1], P_RIGHT_MERC[1]],
    marker="^",
    color="cyan",
    edgecolor="k",
    s=80,
    zorder=5,
    label="Array positions",
)
ax.text(P_LEFT_MERC[0],  P_LEFT_MERC[1],  "  Left",  ha="left",
        va="bottom", color="white")
ax.text(P_RIGHT_MERC[0], P_RIGHT_MERC[1], "  Right", ha="left",
        va="bottom", color="white")

left_line,  = ax.plot([], [], "o-", lw=1.2, color="yellow",
                      label="Left array proj.")
right_line, = ax.plot([], [], "o-", lw=1.2, color="lime",
                      label="Right array proj.")
mid_line,   = ax.plot([], [], "o-", lw=2.0, color="red",
                      label="Midpoint (est.)")
ax.legend(loc="upper left")

# Storage for live trajectories (list of 2-D points)
traj_left:  list[np.ndarray] = []
traj_right: list[np.ndarray] = []
traj_mid:   list[np.ndarray] = []


# ── Core processing per audio block ─────────────────────────────────────
def process_block(block_idx: int,
                  wf_left: wave.Wave_read,
                  wf_right: wave.Wave_read,
                  writer: csv.writer) -> bool:
    """Beamform, project, log & update plot for one block. Returns False on EOF."""
    frames_left = wf_left.readframes(CHUNK)
    frames_right = wf_right.readframes(CHUNK)
    if min(len(frames_left), len(frames_right)) < CHUNK * NUM_MICS * 2:
        return False  # End-of-file

    # Reshape to (samples, mics)
    audio_left = np.frombuffer(frames_left, np.int16).reshape(-1, NUM_MICS)
    audio_right = np.frombuffer(frames_right, np.int16).reshape(-1, NUM_MICS)

    # Band-pass and normalise
    filt_left  = apply_bandpass_filter(audio_left,  LOWCUT, HIGHCUT,
                                       RATE, order=FILTER_ORDER)
    filt_right = apply_bandpass_filter(audio_right, LOWCUT, HIGHCUT,
                                       RATE, order=FILTER_ORDER)
    for sig in (filt_left, filt_right):
        m = np.abs(sig).max()
        if m:
            sig /= m

    # ── Beamform LEFT ──
    energy_L = np.zeros((len(AZIMUTH_RANGE), len(ELEV_RANGE)))
    for ia, _ in enumerate(AZIMUTH_RANGE):
        for ie, _ in enumerate(ELEV_RANGE):
            y = apply_beamforming(filt_left, delays[ia, ie])
            energy_L[ia, ie] = np.sum(y ** 2) / CHUNK
    ia_max, ie_max = np.unravel_index(np.argmax(energy_L), energy_L.shape)
    raw_az_L, raw_el_L = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ── Beamform RIGHT ──
    energy_R = np.zeros_like(energy_L)
    for ia, _ in enumerate(AZIMUTH_RANGE):
        for ie, _ in enumerate(ELEV_RANGE):
            y = apply_beamforming(filt_right, delays[ia, ie])
            energy_R[ia, ie] = np.sum(y ** 2) / CHUNK
    ia_max, ie_max = np.unravel_index(np.argmax(energy_R), energy_R.shape)
    raw_az_R, raw_el_R = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ── Moving average of last SMOOTH_LEN values ──
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

    # ── Ray projection to altitude ──
    dL = azel_to_unit_vec(smooth_az_L, smooth_el_L)
    dR = azel_to_unit_vec(smooth_az_R, smooth_el_R)
    proj_L = project_to_altitude(P_LEFT,  dL, TARGET_ALT_M)
    proj_R = project_to_altitude(P_RIGHT, dR, TARGET_ALT_M)
    if proj_L is None or proj_R is None:
        return True

    midpoint = 0.5 * (proj_L + proj_R)

    # ── Convert to Web-Mercator for drawing ──
    merc_L   = utm56s_to_merc(proj_L[0], proj_L[1])
    merc_R   = utm56s_to_merc(proj_R[0], proj_R[1])
    merc_mid = utm56s_to_merc(midpoint[0], midpoint[1])

    traj_left.append(merc_L)
    traj_right.append(merc_R)
    traj_mid.append(merc_mid)

    left_line.set_data(*zip(*traj_left))
    right_line.set_data(*zip(*traj_right))
    mid_line.set_data(*zip(*traj_mid))
    ax.relim()
    ax.autoscale_view()

    # ── Distance for diagnostics (in metres, still UTM) ──
    dist_L = np.linalg.norm(midpoint - P_LEFT)
    dist_R = np.linalg.norm(midpoint - P_RIGHT)

    curr_time = START_TIME_S + block_idx * (CHUNK / RATE)
    print(f"Blk {block_idx:03d} | t={curr_time:7.2f}s | "
          f"Az/El_L=({smooth_az_L:6.1f},{smooth_el_L:4.1f}) "
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
        f"{dist_L:.3f}", f"{dist_R:.3f}",
    ])
    return True


# ── Main entry point ────────────────────────────────────────────────────
def main() -> None:
    # ─ WAV sanity checks ─
    wfL = wave.open(LEFT_WAV_PATH.open("rb"), "rb")
    wfR = wave.open(RIGHT_WAV_PATH.open("rb"), "rb")
    for wf in (wfL, wfR):
        if (wf.getnchannels() != NUM_MICS or wf.getsampwidth() != 2
                or wf.getframerate() != RATE):
            sys.exit("❌ WAV parameters do not match array config.")

    total_frames = min(wfL.getnframes(), wfR.getnframes())
    start_f = int(START_TIME_S * RATE)
    end_f   = int(END_TIME_S * RATE)
    if not (0 <= start_f < end_f <= total_frames):
        sys.exit("❌ Start/End time out of range.")
    wfL.setpos(start_f)
    wfR.setpos(start_f)
    max_blocks = (end_f - start_f) // CHUNK

    # ─ CSV file ─
    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "block", "time_s",
            "raw_az_L", "raw_el_L", "raw_az_R", "raw_el_R",
            "smooth_az_L", "smooth_el_L", "smooth_az_R", "smooth_el_R",
            "mid_easting_m", "mid_northing_m",
            "dist_L_m", "dist_R_m",
        ])

        # ─ Animation loop ─

        def _update(frame_idx: int):
            if frame_idx >= max_blocks:
                plt.close(fig)
                return left_line, right_line, mid_line

            more = process_block(frame_idx, wfL, wfR, writer)
            if not more:
                plt.close(fig)

            return left_line, right_line, mid_line  # ← imprescindible con blit=True

        ani = FuncAnimation(
            fig, _update,
            frames=max_blocks,
            interval=1,
            repeat=False,
            blit=True  # ← ahora sí
        )

        plt.show()

    wfL.close()
    wfR.close()
    print(f"✅ CSV saved to {CSV_PATH.resolve()!s}")


if __name__ == "__main__":
    main()
