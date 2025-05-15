import numpy as np
import matplotlib.pyplot as plt
import wave
import time
import csv
import os

# ------------------------------------------------------------------
# External beamforming utilities (unchanged)
# ------------------------------------------------------------------
from Utilities.functions import (
    microphone_positions_8_helicop,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ------------------------------------------------------------------
# Parameters and pre‑computations
# ------------------------------------------------------------------
RATE          = 48_000                     # Hz
FRAME_MS      = 100                        # one estimate every 100 ms
CHUNK         = int(RATE * FRAME_MS / 1000)
LOWCUT        = 400.0                      # Hz
HIGHCUT       = 3_000.0                    # Hz
FILTER_ORDER  = 5
C_SPEED       = 343.0                      # m / s

azimuth_range    = np.arange(-180, 181, 4) # deg
elevation_range  = np.arange(0,   91,  4)  # deg

mic_positions = microphone_positions_8_helicop()
CHANNELS      = mic_positions.shape[0]

# Pre‑compute integer delay samples for every (az, el)
precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS),
    dtype=np.int32,
)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, C_SPEED
        )

# ------------------------------------------------------------------
# WAV‑processing function (now writes a CSV)
# ------------------------------------------------------------------
def process_audio_file(wav_filename: str) -> None:
    """
    Process a multichannel WAV in 100‑ms chunks, perform delay‑and‑sum
    beamforming, update a real‑time heat‑map, and log one CSV row per
    chunk:  time_s , az_deg , el_deg , confidence
    """
    wf = wave.open(wav_filename, "rb")

    # Basic sanity checks ---------------------------------------------------
    if wf.getnchannels() != CHANNELS:
        raise RuntimeError(
            f"{wav_filename}: expected {CHANNELS} channels, "
            f"found {wf.getnchannels()}"
        )
    if wf.getsampwidth() != 2:
        raise RuntimeError(f"{wav_filename}: only 16‑bit PCM supported")
    if wf.getframerate() != RATE:
        raise RuntimeError(f"{wav_filename}: sampling rate must be {RATE} Hz")

    # --- CSV logger --------------------------------------------------------
    stem          = os.path.splitext(os.path.basename(wav_filename))[0]
    csv_path      = stem + "_angles.csv"
    csv_fh        = open(csv_path, "w", newline="")
    csv_writer    = csv.writer(csv_fh)
    csv_writer.writerow(["time_s", "az_deg", "el_deg", "confidence"])
    print(f"[INFO] Logging angle estimates → {csv_path}")

    # --- Real‑time plot ----------------------------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beamforming Energy Map (real‑time)")
    max_marker, = ax.plot([], [], "ro", label="Peak")
    ax.legend(); ax.grid(True)
    fig.colorbar(heatmap, ax=ax, label="Energy")

    # --- Main processing loop ---------------------------------------------
    frame_idx = 0
    while True:
        data = wf.readframes(CHUNK)
        if len(data) < CHUNK * CHANNELS * 2:           # EOF
            break

        # Convert to int16 → (samples, channels)
        audio_chunk = np.frombuffer(data, np.int16).reshape(-1, CHANNELS)

        # Band‑pass filter
        filtered_chunk = apply_bandpass_filter(
            audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER
        )

        # Energy map (nested loops kept for clarity; vectorisation optional)
        energy_map = np.zeros((len(azimuth_range), len(elevation_range)))
        for i in range(len(azimuth_range)):
            for j in range(len(elevation_range)):
                beam_sig = apply_beamforming(
                    filtered_chunk, precomputed_delays[i, j, :]
                )
                energy_map[i, j] = np.sum(beam_sig**2)

        # Peak detection
        peak_idx            = np.unravel_index(np.argmax(energy_map),
                                               energy_map.shape)
        est_az              = azimuth_range[peak_idx[0]]
        est_el              = elevation_range[peak_idx[1]]
        peak_energy         = energy_map[peak_idx]
        confidence          = float(peak_energy / (energy_map.sum() + 1e-12))
        time_s              = frame_idx * CHUNK / RATE
        frame_idx          += 1

        # Write CSV row
        csv_writer.writerow([f"{time_s:.4f}", est_az, est_el,
                             f"{confidence:.6f}"])

        # Update plot
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        max_marker.set_data([est_az], [est_el])
        fig.canvas.draw(); fig.canvas.flush_events()

        # Keep real‑time pacing (≈ 100 ms per frame)
        time.sleep(FRAME_MS / 1000.0)

    # ----------------------------------------------------------------------
    plt.ioff(); plt.show()
    csv_fh.close(); wf.close()
    print(f"[INFO] Finished {wav_filename}")

# ------------------------------------------------------------------
# Main loop: list your WAVs here
# ------------------------------------------------------------------
wav_filenames = [
    "heli_12052025.wav",
    # Add more paths as needed
]

for wav_file in wav_filenames:
    print(f"Processing file: {wav_file}")
    process_audio_file(wav_file)
