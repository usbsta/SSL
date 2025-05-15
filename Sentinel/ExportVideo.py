"""
Real‑time beamforming + on‑the‑fly MP4 export
Author: <your‑name>
Date   : 2025‑04‑17
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
import time
import os

# ---------- External beamforming utilities ----------
from Utilities.functions import (
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter
)
# from Utilities.functions import microphone_positions_8_medium   # if you prefer

# ---------- Parameters ----------
RATE          = 192_000                     # Hz
CHUNK         = int(0.1 * RATE)             # 100 ms
LOWCUT        = 10_000.0                    # Hz
HIGHCUT       = 70_000.0                    # Hz
FILTER_ORDER  = 5
c             = 343                         # m/s
CHANNELS      = 8                           # number of mics

# Grid definition
azimuth_range    = np.arange(-180, 181, 1)  # deg
elevation_range  = np.arange(0, 90, 1)      # deg

# Mic coordinates (m)
mic_positions = [
    (0.0,   0.0,  0.02),   # 11
    (0.0,   0.01, 0.02),   # 12
    (0.0,  -0.015, 0),     # 13
    (0.0,   0.025, 0),     # 14
    (0.005, 0.005, 0.02),  # 21
    (-0.005,0.005, 0.02),  # 22
    (0.02,  0.005, 0),     # 23
    (-0.02, 0.005, 0)      # 24
]

# ---------- Pre‑compute integer delay samples ----------
precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32
)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, c
        )

# ---------- Processing function ----------
def process_audio_file(wav_filename, video_filename="beamforming.mp4",
                       png_dir=None, fps=10, clim=(1e7, 1e9)):
    """
    Stream beamforming heat‑map and write an H.264 MP4 in real time.

    Parameters
    ----------
    wav_filename : str  – input multichannel WAV
    video_filename : str – output MP4 path
    png_dir : str | None – if given, save each frame as PNG inside this folder
    fps : int           – frames per second for the MP4
    clim : (float,float) – colour‑map limits for consistent scaling
    """
    # ---------- WAV I/O checks ----------
    wf = wave.open(wav_filename, 'rb')
    assert wf.getnchannels() == CHANNELS, "Unexpected channel count"
    assert wf.getsampwidth() == 2,        "Require 16‑bit PCM"
    assert wf.getframerate() == RATE,     "Sampling rate mismatch"

    # ---------- Matplotlib figure ----------
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin='lower', aspect='auto', cmap='inferno', vmin=clim[0], vmax=clim[1]
    )
    fig.colorbar(heatmap, ax=ax, label='Energy')
    #ax.set_xlabel('Azimuth (deg)')
    #ax.set_ylabel('Elevation (deg)')
    #ax.set_title('Beamforming Energy Map')
    max_marker, = ax.plot([], [], 'ro', label='Max energy')
    #ax.legend();  ax.grid(True);  fig.tight_layout()

    # ---------- Video writer ----------
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='USB STA'), bitrate=2400)

    # Create PNG directory if requested
    if png_dir is not None:
        os.makedirs(png_dir, exist_ok=True)

    # ---------- Main loop with on‑the‑fly capture ----------
    frame_idx = 0
    with writer.saving(fig, video_filename, dpi=200):
        while True:
            raw = wf.readframes(CHUNK)
            if len(raw) < CHUNK * CHANNELS * 2:
                break  # reached EOF

            # Convert to shape (CHUNK, CHANNELS)
            chunk = np.frombuffer(raw, dtype=np.int16).reshape((-1, CHANNELS))

            # Band‑pass filter
            chunk_filt = apply_bandpass_filter(
                chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER
            )

            # Energy map
            energy = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    bf = apply_beamforming(chunk_filt, precomputed_delays[i, j])
                    energy[i, j] = np.sum(bf ** 2)

            # Update plot
            idx_max = np.unravel_index(np.argmax(energy), energy.shape)
            est_az  = azimuth_range[idx_max[0]]
            est_el  = elevation_range[idx_max[1]]

            heatmap.set_data(energy.T)
            heatmap.set_clim(vmin=np.min(energy), vmax=np.max(energy))
            #heatmap.set_clim(1e7, 1e9)
            max_marker.set_data([est_az], [est_el])

            # Draw once, then grab frame
            fig.canvas.draw()
            writer.grab_frame()           # <-- adds frame to MP4
            frame_idx += 1

            # Optional PNG export (comment‑in if needed)
            # if png_dir is not None:
            #     fig.savefig(f"{png_dir}/frame_{frame_idx:05d}.png", dpi=200)

            # Simulate real‑time delay
            time.sleep(CHUNK / RATE)

    plt.ioff()
    plt.close(fig)
    wf.close()
    print(f"Finished: {video_filename} ({frame_idx} frames)")

# ---------- Example run ----------
if __name__ == "__main__":
    wav_files = [
        r"C:\Users\30068385\OneDrive - Western Sydney University\recordings\GasLeake\GasLeakeCompSel2.wav"
    ]
    for wav in wav_files:
        out_mp4 = os.path.splitext(os.path.basename(wav))[0] + "_beamforming3.mp4"
        process_audio_file(wav, video_filename=out_mp4, png_dir=None, fps=10)
