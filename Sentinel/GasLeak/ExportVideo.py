

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
import os
import sys
import time

from Utilities.functions import (
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

RATE          = 192_000                   # Hz
CHUNK         = int(0.1 * RATE)           # 100 ms
LOWCUT        = 7_000.0                  # Hz
HIGHCUT       = 32_000.0                  # Hz
FILTER_ORDER  = 5
c             = 343                       # Speed of sound (m/s)

azimuth_range    = np.arange(-180, 181, 2)   # degrees
elevation_range  = np.arange(-0, 90, 1)    # degrees


mic_positions = [
    (0.0, 0.0, 0.02),         # Rec 1 Ch 1
    (0.0, 0.01, 0.02),        # Rec 2 Ch 1
    (0.0, -0.015, 0.0),       # Rec 1 Ch 2
    (0.0, 0.025, 0.0),        # Rec 2 Ch 2
    (0.005, 0.005, 0.02),     # Rec 1 Ch 3
    (-0.005, 0.005, 0.02),    # Rec 2 Ch 3
    (0.02, 0.005, 0.0),       # Rec 1 Ch 4
    (-0.02, 0.005, 0.0)       # Rec 2 Ch 4
]
CHANNELS = len(mic_positions)


precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32
)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, c
        )

# ---------- Helper: WAV-to-video ----------
def process_audio_file(wav_path: str, video_out: str, dpi: int = 150) -> None:
    """Beamforms `wav_path` and writes an MP4 file to `video_out`."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    wf = wave.open(wav_path, "rb")

    # -------- Parameter checks --------
    if wf.getnchannels() != CHANNELS:
        wf.close()
        raise ValueError(
            f"Expected {CHANNELS} channels, got {wf.getnchannels()} channels."
        )
    if wf.getsampwidth() != 2:
        wf.close()
        raise ValueError("Only 16-bit PCM WAV files are supported.")
    if wf.getframerate() != RATE:
        wf.close()
        raise ValueError(
            f"Sampling rate mismatch: WAV={wf.getframerate()} Hz vs RATE={RATE} Hz."
        )

    total_frames   = wf.getnframes()
    frames_per_step = CHUNK
    total_steps    = total_frames // frames_per_step
    chunk_duration = CHUNK / RATE  # seconds (0.1 s)

    # -------- Matplotlib setup --------
    plt.ioff()  # Do not show an interactive window

    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower", aspect="auto", cmap="inferno",
    )
    fig.colorbar(heatmap, ax=ax, label="Energy")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beamforming Energy Map")
    max_marker, = ax.plot([], [], "ro", label="Max Energy")
    ax.legend()
    ax.grid(True)

    # Time overlay text (upper-left, axes coordinates)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes,
        color="white", fontsize=10, ha="left", va="top",
        bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"),
    )

    # -------- Video writer --------
    fps = int(round(1.0 / chunk_duration))  # 10 fps for 100 ms chunks
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)

    print(f"Exporting video → {video_out}  |  {total_steps} frames @ {fps} fps")
    start_time = time.time()

    with writer.saving(fig, video_out, dpi=dpi):
        for step in range(total_steps):
            # ----- Read audio chunk -----
            data = wf.readframes(CHUNK)
            if len(data) < CHUNK * CHANNELS * 2:  # Incomplete chunk ⇒ stop
                break
            audio_chunk = (
                np.frombuffer(data, dtype=np.int16)
                .reshape((-1, CHANNELS))
            )

            # ----- Band-pass filter -----
            filtered_chunk = apply_bandpass_filter(
                audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER
            )

            # ----- Beamforming energy map -----
            energy_map = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    bf_signal = apply_beamforming(
                        filtered_chunk, precomputed_delays[i, j, :]
                    )
                    energy_map[i, j] = np.sum(bf_signal ** 2)

            # ----- Update plot -----
            max_idx = np.unravel_index(
                np.argmax(energy_map), energy_map.shape
            )
            est_az = azimuth_range[max_idx[0]]
            est_el = elevation_range[max_idx[1]]

            heatmap.set_data(energy_map.T)
            #heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
            heatmap.set_clim(1e7, vmax=np.max(energy_map))
            max_marker.set_data([est_az], [est_el])

            # Overlay current time in seconds on video
            current_time_s = step * chunk_duration
            time_text.set_text(f"t = {current_time_s:.2f} s")

            writer.grab_frame()

            # ----- Console progress -----
            progress = (step + 1) / total_steps * 100.0
            sys.stdout.write(f"\rProgress: {progress:6.2f}%")
            sys.stdout.flush()

    wf.close()
    elapsed = time.time() - start_time
    sys.stdout.write("\nDone. Video saved.\n")
    print(f"Encoding time: {elapsed:.1f} s "
          f"({elapsed/total_steps:.3f} s per frame ≈ real-time x{chunk_duration/(elapsed/total_steps):.2f})")

# ---------- Main ----------
if __name__ == "__main__":
    WAV_FILES = [
        # r"/path/to/your/8-channel_recording.wav"
        r"C:\Users\30068385\OneDrive - Western Sydney University\recordings\GasLeake\robotF2.wav"
    ]

    for wav_path in WAV_FILES:
        stem      = os.path.splitext(os.path.basename(wav_path))[0]
        video_out = f"{stem}_0to90_7to32.mp4"
        process_audio_file(wav_path, video_out)
