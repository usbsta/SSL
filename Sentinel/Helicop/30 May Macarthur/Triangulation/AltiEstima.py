import numpy as np
import wave
import time
import csv
from collections import deque

# -------------------------------------------------------------------------
# External DSP helpers (imported from your Utilities module)
# -------------------------------------------------------------------------
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# -------------------------------------------------------------------------
# GLOBAL SETTINGS (EDIT THESE AS NEEDED)
# -------------------------------------------------------------------------
LEFT_WAV_PATH  = "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/30_05_25/1_Left/Macarthur/left.wav"
RIGHT_WAV_PATH = "/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/30_05_25/2_Right/Macarthur/right.wav"

RATE            = 48000            # Hz
CHUNK           = int(0.1 * RATE)  # 100 ms per block
LOWCUT          = 180.0            # Hz
HIGHCUT         = 2000.0           # Hz
FILTER_ORDER    = 5
SPEED_OF_SOUND  = 343.0            # m/s

AZIMUTH_RANGE   = np.arange(-180, 181, 1)  # degrees
ELEVATION_RANGE = np.arange(0, 91, 1)      # degrees

# UTM coordinates of each array (zone 56 S). Units: meters (x=easting, y=northing, z=height).
P_RIGHT = np.array([
    287322.3,     # easting [m]
    6228859.5,    # northing [m]
    328 * 0.3048  # altitude [ft → m] ≈ 99.97 m
], dtype=np.float64)

P_LEFT = np.array([
    287307.7,     # easting [m]
    6228856.9,    # northing [m]
    328 * 0.3048  # altitude [ft → m] ≈ 105.43 m
], dtype=np.float64)

CSV_OUTPUT_PATH = "triangulation_debug_dist2.csv"

MIN_ALT_M = 0     # meters
MAX_ATTR_M = 2000 # meters

START_TIME = 3437.0  # seconds (e.g., 5 min 42 s)
END_TIME   = 3498.0  # seconds (e.g., 5 min 58 s)

# -------------------------------------------------------------------------
# PRECOMPUTE BEAMFORMING DELAYS FOR ONE ARRAY (same geometry on both sides)
# -------------------------------------------------------------------------
mic_positions = microphone_positions_8_helicop()
NUM_MICS      = mic_positions.shape[0]

precomputed_delays = np.empty(
    (len(AZIMUTH_RANGE), len(ELEVATION_RANGE), NUM_MICS),
    dtype=np.int32
)
for i, az in enumerate(AZIMUTH_RANGE):
    for j, el in enumerate(ELEVATION_RANGE):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, SPEED_OF_SOUND
        )

# -------------------------------------------------------------------------
# HELPER: Convert (az, el) → unit‐vector in ENU coordinates
# -------------------------------------------------------------------------
def az_el_to_unit_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az_rad = np.deg2rad(azimuth_deg)
    el_rad = np.deg2rad(elevation_deg)
    dx = np.cos(el_rad) * np.sin(az_rad)  # East component
    dy = np.cos(el_rad) * np.cos(az_rad)  # North component
    dz = np.sin(el_rad)                   # Up component
    vec = np.array([dx, dy, dz], dtype=np.float64)
    return vec / np.linalg.norm(vec)

# -------------------------------------------------------------------------
# HELPER: Triangulate two rays → midpoint and uncertainty
# -------------------------------------------------------------------------
def triangulate_two_rays(P1: np.ndarray, d1: np.ndarray,
                         P2: np.ndarray, d2: np.ndarray) -> (np.ndarray, float):
    """
    Given two array positions P1, P2 and their unit‐direction vectors d1, d2,
    compute the closest midpoint X_est and the uncertainty (half distance between the two
    closest points).
    """
    c = np.dot(d1, d2)                # cos(angle between d1 and d2)
    DeltaP = P2 - P1
    dot_p_d1 = np.dot(DeltaP, d1)
    dot_p_d2 = np.dot(DeltaP, d2)
    denom = 1.0 - c * c
    if abs(denom) < 1e-6:
        return None, None

    lambda1 = (dot_p_d1 - c * dot_p_d2) / denom
    lambda2 = (dot_p_d2 - c * dot_p_d1) / denom

    Q1 = P1 + lambda1 * d1
    Q2 = P2 + lambda2 * d2
    X_est = 0.5 * (Q1 + Q2)
    uncertainty = 0.5 * np.linalg.norm(Q1 - Q2)
    return X_est, uncertainty

# -------------------------------------------------------------------------
# MAIN: Simplified “debug” loop that prints Z-est and error, now also distance
# -------------------------------------------------------------------------
def run_debug_triangulation():
    # Open both WAV files
    wf_left  = wave.open(LEFT_WAV_PATH,  "rb")
    wf_right = wave.open(RIGHT_WAV_PATH, "rb")

    if wf_left.getnchannels() != NUM_MICS or wf_right.getnchannels() != NUM_MICS:
        raise RuntimeError("Channel count mismatch with microphone array.")
    if wf_left.getsampwidth() != 2 or wf_right.getsampwidth() != 2:
        raise RuntimeError("Expecting 16-bit PCM WAV files.")
    if wf_left.getframerate() != RATE or wf_right.getframerate() != RATE:
        raise RuntimeError(f"Expecting sample rate = {RATE} Hz.")

    total_frames_left  = wf_left.getnframes()
    total_frames_right = wf_right.getnframes()
    start_frame = int(START_TIME * RATE)
    end_frame   = int(END_TIME   * RATE)

    if not (0 <= start_frame < total_frames_left) or not (0 <= start_frame < total_frames_right):
        raise RuntimeError("START_TIME out of range.")
    if not (start_frame < end_frame <= total_frames_left) or not (start_frame < end_frame <= total_frames_right):
        raise RuntimeError("END_TIME out of range or <= START_TIME.")

    wf_left.setpos(start_frame)
    wf_right.setpos(start_frame)
    max_blocks = (end_frame - start_frame) // CHUNK

    # Prepare moving-average buffers
    buffer_az1 = deque(maxlen=3)
    buffer_el1 = deque(maxlen=3)
    buffer_az2 = deque(maxlen=3)
    buffer_el2 = deque(maxlen=3)

    # Open CSV for logging
    with open(CSV_OUTPUT_PATH, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Add two new columns for distance from each array to the estimated source
        csvwriter.writerow([
            "block_index",
            "time_s",
            "raw_az1", "raw_el1",
            "raw_az2", "raw_el2",
            "smooth_az1", "smooth_el1",
            "smooth_az2", "smooth_el2",
            "z_est_m", "triang_error_m",
            "dist_left_m", "dist_right_m"
        ])

    print("Starting debug loop...")
    for block_idx in range(max_blocks):
        frames_left  = wf_left.readframes(CHUNK)
        frames_right = wf_right.readframes(CHUNK)

        if len(frames_left) < CHUNK * NUM_MICS * 2 or len(frames_right) < CHUNK * NUM_MICS * 2:
            break

        # Convert raw bytes → int16 arrays of shape (samples, NUM_MICS)
        audio_left  = np.frombuffer(frames_left,  dtype=np.int16).reshape((-1, NUM_MICS))
        audio_right = np.frombuffer(frames_right, dtype=np.int16).reshape((-1, NUM_MICS))

        # Band-pass filter each channel
        filtered_left  = apply_bandpass_filter(audio_left,  LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)
        filtered_right = apply_bandpass_filter(audio_right, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # Normalize so max(|signal|)=1
        if filtered_left.max() != 0:
            filtered_left  = filtered_left  / np.abs(filtered_left).max()
        if filtered_right.max() != 0:
            filtered_right = filtered_right / np.abs(filtered_right).max()

        # Compute energy maps for Left array
        energy1 = np.zeros((len(AZIMUTH_RANGE), len(ELEVATION_RANGE)))
        for i, _az in enumerate(AZIMUTH_RANGE):
            for j, _el in enumerate(ELEVATION_RANGE):
                sig = apply_beamforming(filtered_left, precomputed_delays[i, j, :])
                energy1[i, j] = np.sum(sig**2) / CHUNK
        idx1 = np.unravel_index(np.argmax(energy1), energy1.shape)
        raw_az1, raw_el1 = AZIMUTH_RANGE[idx1[0]], ELEVATION_RANGE[idx1[1]]

        # Compute energy maps for Right array
        energy2 = np.zeros_like(energy1)
        for i, _az in enumerate(AZIMUTH_RANGE):
            for j, _el in enumerate(ELEVATION_RANGE):
                sig = apply_beamforming(filtered_right, precomputed_delays[i, j, :])
                energy2[i, j] = np.sum(sig**2) / CHUNK
        idx2 = np.unravel_index(np.argmax(energy2), energy2.shape)
        raw_az2, raw_el2 = AZIMUTH_RANGE[idx2[0]], ELEVATION_RANGE[idx2[1]]

        # Moving average
        buffer_az1.append(raw_az1)
        buffer_el1.append(raw_el1)
        buffer_az2.append(raw_az2)
        buffer_el2.append(raw_el2)

        if len(buffer_az1) == buffer_az1.maxlen:
            smooth_az1 = float(np.mean(buffer_az1))
            smooth_el1 = float(np.mean(buffer_el1))
            smooth_az2 = float(np.mean(buffer_az2))
            smooth_el2 = float(np.mean(buffer_el2))
        else:
            smooth_az1, smooth_el1 = raw_az1, raw_el1
            smooth_az2, smooth_el2 = raw_az2, raw_el2

        # Triangulation
        d1 = az_el_to_unit_vector(smooth_az1, smooth_el1)
        d2 = az_el_to_unit_vector(smooth_az2, smooth_el2)
        X_est, error = triangulate_two_rays(P_LEFT,  d1, P_RIGHT, d2)

        if X_est is None:
            z_est = np.nan
            error_val = np.nan
            dist_left = np.nan
            dist_right = np.nan
        else:
            _, _, z_est = X_est
            # Only keep z_est if within valid altitude range
            if not (MIN_ALT_M <= z_est <= MAX_ATTR_M):
                z_est = np.nan
                error_val = np.nan
                dist_left = np.nan
                dist_right = np.nan
            else:
                error_val = error
                # Compute distance from each array to the estimated source
                dist_left = np.linalg.norm(X_est - P_LEFT)
                dist_right = np.linalg.norm(X_est - P_RIGHT)

        current_time = START_TIME + block_idx * (CHUNK / RATE)

        # Print to console for quick debugging, including distances
        print(
            f"Block {block_idx:3d} | Time {current_time:7.2f} s | "
            f"raw Az/El → ({raw_az1:6.1f}, {raw_el1:4.1f}) / "
            f"({raw_az2:6.1f}, {raw_el2:4.1f}) | "
            f"smooth Az/El → ({smooth_az1:6.1f}, {smooth_el1:4.1f}) / "
            f"({smooth_az2:6.1f}, {smooth_el2:4.1f}) | "
            f"z_est = {z_est:7.2f} m | error = {error_val:6.2f} m | "
            f"dist_left = {dist_left:6.2f} m | dist_right = {dist_right:6.2f} m"
        )

        # Append to CSV
        with open(CSV_OUTPUT_PATH, mode="a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                block_idx,
                f"{current_time:.3f}",
                f"{raw_az1:.1f}", f"{raw_el1:.1f}",
                f"{raw_az2:.1f}", f"{raw_el2:.1f}",
                f"{smooth_az1:.1f}", f"{smooth_el1:.1f}",
                f"{smooth_az2:.1f}", f"{smooth_el2:.1f}",
                f"{z_est:.3f}", f"{error_val:.3f}",
                f"{dist_left:.3f}", f"{dist_right:.3f}"
            ])

        # If you want to slow down to “real‐time” speed, uncomment the next line:
        # time.sleep(CHUNK / RATE)

    wf_left.close()
    wf_right.close()
    print(f"✅ Debug CSV saved at '{CSV_OUTPUT_PATH}'")

if __name__ == "__main__":
    run_debug_triangulation()
