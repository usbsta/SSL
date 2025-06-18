#!/usr/bin/env python3
"""
Real-time “flat-plane” localisation debug tool – satellite overlay
with local coordinate system (Left = [0,0]), showing only midpoint and
the average distance between midpoint and each array.

* Two 8-mic arrays (“Left” & “Right”) process multichannel WAV files.
* Every 100 ms block:
    1. Beamform to find the max-energy azimuth/elevation for each array.
    2. Convert Az/El → unit ENU vector.
    3. Project that ray to a **fixed** altitude plane (700 m AGL).
    4. Convert UTM → Web-Mercator, then subtract P_LEFT_merc so that Left = (0,0).
    5. Fetch the basemap raster via contextily.bounds2img and plot it with imshow,
       using local coordinates (Web-Mercator – P_LEFT_merc).
    6. Plot only the midpoint trajectory in coordinates local (m east, m north
       relative to Left).
    7. Show text in the bottom-right corner of the plot for the average of
       distances dL and dR (midpoint to Left, midpoint to Right).
    8. Print & CSV-log raw/smoothed angles + distances to each array + average.
"""

# ── Standard library ────────────────────────────────────────────────────
from collections import deque
import csv
import wave
import sys
from pathlib import Path

# ── Third-party ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyproj import Transformer
import contextily as ctx

# ── External DSP helpers (YOUR package/module) ──────────────────────────
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ────────────────────────────────────────────────────────────────────────
#                            GLOBAL SETTINGS
# ────────────────────────────────────────────────────────────────────────

# ─ File paths ───────────────────────────────────────────────────────────
LEFT_WAV_PATH  = Path(
    "/Users/30068385/OneDrive - Western Sydney University/"
    "recordings/Helicop/30_05_25/1_Left/Macarthur/left.wav"
)
RIGHT_WAV_PATH = Path(
    "/Users/30068385/OneDrive - Western Sydney University/"
    "recordings/Helicop/30_05_25/2_Right/Macarthur/right.wav"
)
CSV_PATH = Path("flat_plane_debug_local_map.csv")

# ─ Audio & beamforming parameters ───────────────────────────────────────
RATE            = 48_000            # Sampling rate (Hz)
CHUNK           = int(0.1 * RATE)   # 100 ms per processing block
LOWCUT, HIGHCUT = 180.0, 2_000.0    # Band-pass in Hz
FILTER_ORDER    = 5                 # Butterworth order
C               = 343.0             # Speed of sound (m/s)

AZIMUTH_RANGE   = np.arange(-180, 181, 4)   # degrees
ELEV_RANGE      = np.arange(0, 91, 4)       # degrees

# ─ UTM geometry: zone 56 S (EPSG:32756) ─────────────────────────────────
# Real UTM coordinates of your arrays (easting, northing, elevation_ASL [m])
P_LEFT_UTM  = np.array([287_307.7, 6_228_856.9, 328 * 0.3048], dtype=np.float64)
P_RIGHT_UTM = np.array([287_322.3, 6_228_859.5, 328 * 0.3048], dtype=np.float64)

# Target altitude: 700 m AGL + ground elevation (m ASL)
TARGET_ALT_M = 700.0 + P_LEFT_UTM[2]

SMOOTH_LEN   = 70   # moving-average window (blocks)

# Playback start/end in seconds (within the WAV)
START_TIME_S = 3_436.6
END_TIME_S   = 3_453.5

# ─── Coordinate transformers ─────────────────────────────────────────────
# 1) UTM 56 S (EPSG:32756) → WGS-84 (EPSG:4326)
_utm_to_wgs84 = Transformer.from_crs(32756, 4326, always_xy=True)
# 2) WGS-84 (EPSG:4326) → Web-Mercator (EPSG:3857)
_wgs84_to_merc = Transformer.from_crs(4326, 3857, always_xy=True)

def utm56s_to_merc(easting: float, northing: float) -> np.ndarray:
    """Convert UTM zone 56 S → Web-Mercator (x, y in m)."""
    lon, lat = _utm_to_wgs84.transform(easting, northing)
    x, y     = _wgs84_to_merc.transform(lon, lat)
    return np.array([x, y], dtype=np.float64)

# ── Precompute Web-Mercator for array origins ───────────────────────────
P_LEFT_merc  = utm56s_to_merc(P_LEFT_UTM[0],  P_LEFT_UTM[1])
P_RIGHT_merc = utm56s_to_merc(P_RIGHT_UTM[0], P_RIGHT_UTM[1])

# Local offset of Right in Web-Mercator (Left becomes [0,0] after subtracting)
OFFSET_RIGHT_merc = P_RIGHT_merc - P_LEFT_merc

# ─ Beamforming delay lookup (same geometry for both arrays) ─────────────
mic_pos = microphone_positions_8_helicop()           # (M × 3) in UTM relative to each array
NUM_MICS = mic_pos.shape[0]

delays = np.empty((len(AZIMUTH_RANGE), len(ELEV_RANGE), NUM_MICS), dtype=np.int32)
for ia, az in enumerate(AZIMUTH_RANGE):
    for ie, el in enumerate(ELEV_RANGE):
        delays[ia, ie] = calculate_delays_for_direction(
            mic_pos, az, el, RATE, C
        )

# ── Figure & Axes setup ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

# ─── 1) Calculate absolute bounding box in Web-Mercator ────────────────
margin_m = 700.0
xmin_merc = min(P_LEFT_merc[0], P_RIGHT_merc[0]) - margin_m
xmax_merc = max(P_LEFT_merc[0], P_RIGHT_merc[0]) + margin_m
ymin_merc = min(P_LEFT_merc[1], P_RIGHT_merc[1]) - margin_m
ymax_merc = max(P_LEFT_merc[1], P_RIGHT_merc[1]) + margin_m

# ─── 2) Fetch the basemap image for that bounding box ──────────────────
basemap_img, basemap_extent = ctx.bounds2img(
    xmin_merc, ymin_merc, xmax_merc, ymax_merc,
    zoom=19,
    source=ctx.providers.Esri.WorldImagery
)
# basemap_extent = (xmin_merc, xmax_merc, ymin_merc, ymax_merc)

# ─── 3) Convert that absolute extent to “local” coords by subtracting P_LEFT_merc ─
xmin_loc = basemap_extent[0] - P_LEFT_merc[0]
xmax_loc = basemap_extent[1] - P_LEFT_merc[0]
ymin_loc = basemap_extent[2] - P_LEFT_merc[1]
ymax_loc = basemap_extent[3] - P_LEFT_merc[1]

# ─── 4) Display the basemap via imshow, in local coordinates ────────────
ax.imshow(
    basemap_img,
    extent=(xmin_loc, xmax_loc, ymin_loc, ymax_loc),
    origin='upper',
    interpolation='bilinear',
    zorder=0
)

# ─── 5) Fix axes limits to X ∈ [−600, 600], Y ∈ [−250, 350] ─────────────
ax.set_xlim(-600, 600)
ax.set_ylim(-250, 350)
ax.set_aspect('equal', adjustable='box')

ax.set_title("Aircraft Tracking")
ax.set_xlabel("East (m) relative to Left array")
ax.set_ylabel("North (m) relative to Left array")

# ─── 6) Plot static array marker for Left and Right in local coords ─────
array_scatter = ax.scatter(
    [0.0, OFFSET_RIGHT_merc[0]],
    [0.0, OFFSET_RIGHT_merc[1]],
    marker="^",
    color="cyan",
    edgecolor="k",
    s=80,
    zorder=5,
    label="Array positions"
)
ax.text(0.0, 0.0, " Left", ha="left", va="bottom", color="white")
ax.text(OFFSET_RIGHT_merc[0], OFFSET_RIGHT_merc[1],
        " Right", ha="left", va="top", color="white")

# ─── 7) Prepare a single line for the midpoint trajectory ───────────────
mid_line, = ax.plot([], [], "o-", lw=2.0, color="red", label="Aircraft")

# ─── 8) Add text in bottom-right to display avg distance (dL + dR)/2 ───
# Place text near (x=600, y=-250) with right/bottom alignment
distance_text = ax.text(
    590, -240,
    "",                   # initially empty
    color="white",
    backgroundcolor="black",
    fontsize=16,
    ha="right",
    va="bottom"
)

ax.legend(loc="upper left")

# ─── Storage for live midpoint points (local coords) ─────────────────────
traj_mid: list[np.ndarray] = []

# ── Helper functions ─────────────────────────────────────────────────────

def azel_to_unit_vec(az_deg: float, el_deg: float) -> np.ndarray:
    """
    Convert Azimuth/Elevation (deg ENU) → unit vector [east, north, up].
      az_deg: Azimuth, 0° = north, +90° = east
      el_deg: Elevation, 0°..90° upward
    """
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),   # East component
        np.cos(el) * np.cos(az),   # North component
        np.sin(el)                 # Up component
    ], dtype=np.float64)

def project_to_altitude(P: np.ndarray, d: np.ndarray,
                        target_z: float) -> np.ndarray | None:
    """
    Intersect ray X(t) = P + t·d with horizontal plane z = target_z.
    P: 3-vector [easting, northing, z]
    d: 3-vector unit direction (east,north,up)
    Returns the 3-vector intersection in UTM if t>0, else None.
    """
    if abs(d[2]) < 1e-6:  # Ray parallel to plane
        return None
    t = (target_z - P[2]) / d[2]
    if t <= 0:
        return None
    return P + t * d

# ── Core processing per audio block ─────────────────────────────────────

def process_block(block_idx: int,
                  wf_left: wave.Wave_read,
                  wf_right: wave.Wave_read,
                  writer: csv.writer) -> bool:
    """
    Beamform, project, log & update plot for one block.
    Returns False on EOF.
    """
    # Read CHUNK samples per mic
    frames_left = wf_left.readframes(CHUNK)
    frames_right = wf_right.readframes(CHUNK)
    # EOF check
    if min(len(frames_left), len(frames_right)) < CHUNK * NUM_MICS * 2:
        return False

    # Reshape raw bytes → (samples, mics)
    audio_left  = np.frombuffer(frames_left,  np.int16).reshape(-1, NUM_MICS)
    audio_right = np.frombuffer(frames_right, np.int16).reshape(-1, NUM_MICS)

    # Band-pass + normalize
    filt_left  = apply_bandpass_filter(audio_left,  LOWCUT, HIGHCUT,
                                       RATE, order=FILTER_ORDER)
    filt_right = apply_bandpass_filter(audio_right, LOWCUT, HIGHCUT,
                                       RATE, order=FILTER_ORDER)
    for sig in (filt_left, filt_right):
        peak = np.abs(sig).max()
        if peak:
            sig /= peak

    # ── Beamform LEFT array ──────────────────────────────────────────────
    energy_L = np.zeros((len(AZIMUTH_RANGE), len(ELEV_RANGE)))
    for ia in range(len(AZIMUTH_RANGE)):
        for ie in range(len(ELEV_RANGE)):
            y = apply_beamforming(filt_left, delays[ia, ie])
            energy_L[ia, ie] = np.sum(y**2) / CHUNK
    ia_max, ie_max = np.unravel_index(np.argmax(energy_L), energy_L.shape)
    raw_az_L, raw_el_L = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ── Beamform RIGHT array ─────────────────────────────────────────────
    energy_R = np.zeros_like(energy_L)
    for ia in range(len(AZIMUTH_RANGE)):
        for ie in range(len(ELEV_RANGE)):
            y = apply_beamforming(filt_right, delays[ia, ie])
            energy_R[ia, ie] = np.sum(y**2) / CHUNK
    ia_max, ie_max = np.unravel_index(np.argmax(energy_R), energy_R.shape)
    raw_az_R, raw_el_R = AZIMUTH_RANGE[ia_max], ELEV_RANGE[ie_max]

    # ── Moving average smoothing ──────────────────────────────────────────
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

    # ── Ray projection to fixed altitude ────────────────────────────────
    dL = azel_to_unit_vec(smooth_az_L, smooth_el_L)
    dR = azel_to_unit_vec(smooth_az_R, smooth_el_R)

    proj_L_world = project_to_altitude(P_LEFT_UTM,  dL, TARGET_ALT_M)
    proj_R_world = project_to_altitude(P_RIGHT_UTM, dR, TARGET_ALT_M)
    if proj_L_world is None or proj_R_world is None:
        return True

    midpoint_world = 0.5 * (proj_L_world + proj_R_world)

    # ── Convert to Web-Mercator ─────────────────────────────────────────
    merc_L   = utm56s_to_merc(proj_L_world[0], proj_L_world[1])
    merc_R   = utm56s_to_merc(proj_R_world[0], proj_R_world[1])
    merc_mid = utm56s_to_merc(midpoint_world[0], midpoint_world[1])

    # ── Convert to local coords by subtracting P_LEFT_merc ───────────────
    local_mid = merc_mid - P_LEFT_merc

    # Store midpoint in trajectory list
    traj_mid.append(local_mid)

    # Update only the midpoint line
    mid_line.set_data(*zip(*traj_mid))

    # Compute dL, dR (in UTM) and their average
    dist_L = np.linalg.norm(midpoint_world - P_LEFT_UTM)
    dist_R = np.linalg.norm(midpoint_world - P_RIGHT_UTM)
    avg_dist = 0.5 * (dist_L + dist_R)

    # Update text in bottom-right
    distance_text.set_text(f"Distance Aircraft - Mic. = {avg_dist:.1f} m")

    # (Do not autoscale axes, keep fixed limits)
    # ax.relim(); ax.autoscale_view()

    curr_time = START_TIME_S + block_idx * (CHUNK / RATE)
    print(
        f"Blk {block_idx:03d} | t={curr_time:7.2f}s | "
        f"Az/El_L=({smooth_az_L:6.1f},{smooth_el_L:4.1f}) "
        f"Az/El_R=({smooth_az_R:6.1f},{smooth_el_R:4.1f}) | "
        f"Local_mid_XY=({local_mid[0]:.1f},{local_mid[1]:.1f}) m | "
        f"dL={dist_L:7.1f} m  dR={dist_R:7.1f} m  avg_d={avg_dist:6.1f} m"
    )

    # Write CSV: raw & smooth angles, local midpoint, distances + average
    writer.writerow([
        block_idx, f"{curr_time:.3f}",
        f"{raw_az_L:.1f}", f"{raw_el_L:.1f}",
        f"{raw_az_R:.1f}", f"{raw_el_R:.1f}",
        f"{smooth_az_L:.1f}", f"{smooth_el_L:.1f}",
        f"{smooth_az_R:.1f}", f"{smooth_el_R:.1f}",
        f"{local_mid[0]:.3f}", f"{local_mid[1]:.3f}",
        f"{dist_L:.3f}", f"{dist_R:.3f}",
        f"{avg_dist:.3f}"
    ])
    return True

# ── Main entry point ────────────────────────────────────────────────────
def main() -> None:
    # ─ WAV file sanity checks ─────────────────────────────────────────────
    wfL = wave.open(LEFT_WAV_PATH.open("rb"), "rb")
    wfR = wave.open(RIGHT_WAV_PATH.open("rb"), "rb")
    for wf in (wfL, wfR):
        if (wf.getnchannels() != NUM_MICS or wf.getsampwidth() != 2
                or wf.getframerate() != RATE):
            sys.exit("❌ WAV parameters do not match array config.")

    total_frames = min(wfL.getnframes(), wfR.getnframes())
    start_f = int(START_TIME_S * RATE)
    end_f   = int(END_TIME_S   * RATE)
    if not (0 <= start_f < end_f <= total_frames):
        sys.exit("❌ Start/End time out of range in WAV file.")
    wfL.setpos(start_f)
    wfR.setpos(start_f)
    max_blocks = (end_f - start_f) // CHUNK

    # ─ CSV file setup ─────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "block", "time_s",
            "raw_az_L", "raw_el_L", "raw_az_R", "raw_el_R",
            "smooth_az_L", "smooth_el_L", "smooth_az_R", "smooth_el_R",
            "mid_X_local_m", "mid_Y_local_m",
            "dist_L_m", "dist_R_m", "avg_dist_m"
        ])

        # ─ Animation loop ────────────────────────────────────────────────
        def _update(frame_idx: int):
            if frame_idx >= max_blocks:
                plt.close(fig)
                return (mid_line, distance_text)

            more = process_block(frame_idx, wfL, wfR, writer)
            if not more:
                plt.close(fig)

            return (mid_line, distance_text)

        ani = FuncAnimation(
            fig, _update,
            frames=max_blocks,
            interval=1,   # milliseconds
            repeat=False,
            blit=True
        )

        plt.show()

    wfL.close()
    wfR.close()
    print(f"✅ CSV saved to {CSV_PATH.resolve()!s}")

if __name__ == "__main__":
    main()
