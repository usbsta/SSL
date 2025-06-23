

import numpy as np
import wave
import csv
import time
from collections import deque
from pathlib import Path

# ── External DSP helpers ──────────────────────────────────────────────────
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

# ── GLOBAL SETTINGS ───────────────────────────────────────────────────────
RATE            = 48_000            # Hz
CHUNK           = int(0.1 * RATE)   # 100 ms blocks
LOWCUT, HIGHCUT = 180.0, 2000.0     # Hz
FILTER_ORDER    = 5
SPEED_OF_SOUND  = 343.0             # m · s⁻¹

AZ_RANGE = np.arange(-180, 181, 1)  # °
EL_RANGE = np.arange(10, 51, 1)      # °

# UTM zone 56 S centres (Easting, Northing, Alt-m)
P_N = np.array([322955.1, 6256643.2, 0.0])
P_S = np.array([322951.6, 6256580.0, 0.0])
P_E = np.array([322980.8, 6256638.4, 0.0])
P_W = np.array([322918.0, 6256605.4, 0.0])

# WAV paths – edit as needed
ROOT = Path("/Users/a30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25")
ARRAYS = {
    'N': {
        'wav': ROOT / "N.wav",
        'centre': P_N,
        'mics': mic_6_N_black_thin(),
    },
    'S': {
        'wav': ROOT / "S.wav",
        'centre': P_S,
        'mics': mic_6_S_orange(),
    },
    'E': {
        'wav': ROOT / "E.wav",
        'centre': P_E,
        'mics': mic_6_E_orange(),
    },
    'W': {
        'wav': ROOT / "W.wav",
        'centre': P_W,
        'mics': mic_6_W_black(),
    },
}

CSV_OUTPUT = "triangulation_debug_4arrays2.csv"
MIN_ALT, MAX_ALT = 0.0, 2000.0       # altitude validity range [m]
START_TIME, END_TIME = 0.0, 104.0  # [s] in the WAV files

# ── Maths helpers ─────────────────────────────────────────────────────────
def az_el_to_unit_vector(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.deg2rad([az_deg, el_deg])
    return np.array([
        np.cos(el) * np.sin(az),   # East
        np.cos(el) * np.cos(az),   # North
        np.sin(el)                 # Up
    ])

def triangulate_multi_rays(P_list, d_list):
    """Least-squares intersection of ≥2 rays (see accompanying visualiser)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for P, d in zip(P_list, d_list):
        Ai = np.eye(3) - np.outer(d, d)
        A += Ai
        b += Ai @ P
    X_hat = np.linalg.solve(A, b)

    # Residual diagnostics
    res = []
    for P, d in zip(P_list, d_list):
        proj = P + d * np.dot(X_hat - P, d)
        res.append(np.linalg.norm(X_hat - proj))
    rms_err = float(np.sqrt(np.mean(np.square(res))))
    return X_hat, rms_err

# ── Pre-compute delay tables (array-specific) ─────────────────────────────
delays = {}
for key, cfg in ARRAYS.items():
    mpos = cfg['mics']
    num_mics = mpos.shape[0]
    table = np.empty((len(AZ_RANGE), len(EL_RANGE), num_mics), np.int32)
    for i, az in enumerate(AZ_RANGE):
        for j, el in enumerate(EL_RANGE):
            table[i, j, :] = calculate_delays_for_direction(
                mpos, az, el, RATE, SPEED_OF_SOUND
            )
    delays[key] = table

# ── WAV file preparation ─────────────────────────────────────────────────
handles = {k: wave.open(str(cfg['wav']), "rb") for k, cfg in ARRAYS.items()}
num_mics = {k: h.getnchannels() for k, h in handles.items()}
if len({v for v in num_mics.values()}) != 1:
    raise RuntimeError("All WAV files must have identical channel counts.")
NUM_MICS = list(num_mics.values())[0]

for k, h in handles.items():
    if h.getframerate() != RATE or h.getsampwidth() != 2:
        raise RuntimeError(f"{k}: WAV must be 48 kHz / 16-bit PCM.")
total_frames = {k: h.getnframes() for k, h in handles.items()}

start_frame = int(START_TIME * RATE)
end_frame   = int(END_TIME   * RATE)
if not all(0 <= start_frame < total_frames[k] for k in ARRAYS):
    raise RuntimeError("START_TIME out of range.")
if not all(start_frame < end_frame <= total_frames[k] for k in ARRAYS):
    raise RuntimeError("END_TIME out of range.")

for h in handles.values():
    h.setpos(start_frame)
max_blocks = (end_frame - start_frame) // CHUNK

# ── Per-array smoothing buffers ──────────────────────────────────────────
buffers = {
    k: {'az': deque(maxlen=3),
        'el': deque(maxlen=3)}
    for k in ARRAYS
}

# ── CSV header ───────────────────────────────────────────────────────────
with open(CSV_OUTPUT, "w", newline="") as f:
    w = csv.writer(f)
    header = ["block", "time_s"]
    for k in ARRAYS:
        header += [f"raw_az_{k}", f"raw_el_{k}",
                   f"smooth_az_{k}", f"smooth_el_{k}"]
    header += ["x_est_m", "y_est_m", "z_est_m", "rms_err_m"]
    header += [f"dist_{k}_m" for k in ARRAYS]
    w.writerow(header)

print("▶ Starting 4-array debug loop…")
# ── Main processing loop ─────────────────────────────────────────────────
for blk in range(max_blocks):
    # --- read CHUNK frames from each array ---
    frames = {k: h.readframes(CHUNK) for k, h in handles.items()}
    if any(len(fr) < CHUNK * NUM_MICS * 2 for fr in frames.values()):
        break

    # --- per-array DSP ---
    azel_raw = {}
    for k, cfg in ARRAYS.items():
        # int16 → shape (samples, mics)
        audio = np.frombuffer(frames[k], np.int16).reshape(-1, NUM_MICS)
        filtd = apply_bandpass_filter(audio, LOWCUT, HIGHCUT,
                                      RATE, order=FILTER_ORDER)
        filtd = filtd / np.abs(filtd).max() if filtd.max() else filtd

        # Energy map
        e_map = np.zeros((len(AZ_RANGE), len(EL_RANGE)))
        tbl = delays[k]
        for i, az in enumerate(AZ_RANGE):
            for j, el in enumerate(EL_RANGE):
                beam = apply_beamforming(filtd, tbl[i, j, :])
                e_map[i, j] = np.sum(beam ** 2) / CHUNK
        idx = np.unravel_index(np.argmax(e_map), e_map.shape)
        azel_raw[k] = (AZ_RANGE[idx[0]], EL_RANGE[idx[1]])

        # update buffers
        buffers[k]['az'].append(azel_raw[k][0])
        buffers[k]['el'].append(azel_raw[k][1])

    # --- smoothing ---
    azel_smooth = {}
    for k in ARRAYS:
        if len(buffers[k]['az']) == buffers[k]['az'].maxlen:
            azel_smooth[k] = (float(np.mean(buffers[k]['az'])),
                              float(np.mean(buffers[k]['el'])))
        else:
            azel_smooth[k] = azel_raw[k]

    # --- build ray lists ---
    origins = []
    dirs    = []
    for k, cfg in ARRAYS.items():
        origins.append(cfg['centre'])
        dirs.append(az_el_to_unit_vector(*azel_smooth[k]))
    origins = np.vstack(origins)
    dirs    = np.vstack(dirs)

    # --- triangulate ---
    X_est, rms_err = triangulate_multi_rays(origins, dirs)
    z_ok = MIN_ALT <= X_est[2] <= MAX_ALT
    dists = np.linalg.norm(X_est - origins, axis=1) if z_ok else [np.nan]*len(ARRAYS)

    # --- time stamp ---
    t_now = START_TIME + blk * CHUNK / RATE

    # --- console log ---
    azel_str = " | ".join(
        f"{k}:({azel_smooth[k][0]:6.1f},{azel_smooth[k][1]:4.1f})"
        for k in ARRAYS
    )
    print(f"Blk {blk:3d} @ {t_now:7.2f}s | {azel_str} | "
          f"X̂=({X_est[0]:.1f},{X_est[1]:.1f},{X_est[2]:.1f}) m | "
          f"RMS={rms_err:5.2f} m")

    # --- CSV log ---
    with open(CSV_OUTPUT, "a", newline="") as f:
        w = csv.writer(f)
        row = [blk, f"{t_now:.3f}"]
        for k in ARRAYS:
            row += [f"{azel_raw[k][0]:.1f}", f"{azel_raw[k][1]:.1f}",
                    f"{azel_smooth[k][0]:.1f}", f"{azel_smooth[k][1]:.1f}"]
        row += [f"{X_est[0]:.3f}", f"{X_est[1]:.3f}", f"{X_est[2]:.3f}",
                f"{rms_err:.3f}"]
        row += [f"{d:.3f}" for d in dists]
        w.writerow(row)

    # Optional real-time pace:
    # time.sleep(CHUNK / RATE)

# ── tidy up ───────────────────────────────────────────────────────────────
for h in handles.values():
    h.close()
print(f"✅ CSV written to '{CSV_OUTPUT}'")
