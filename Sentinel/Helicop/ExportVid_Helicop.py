"""
Real-time delay-and-sum beamforming visualisation for an 8-mic pyramid array,
with optional MP4 export of the animated energy map.

Prerequisites
-------------
$ pip install numpy matplotlib
# FFmpeg must also be installed and discoverable on the system PATH.
"""

import numpy as np
import matplotlib.pyplot as plt
import wave
import time
from matplotlib.animation import FFMpegWriter

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
# Global parameters
# -------------------------------------------------------------------------
RATE = 48_000            # Sampling rate (Hz)
CHUNK = int(0.1 * RATE)  # 100 ms of audio per processing block
LOWCUT = 180.0           # Band-pass filter low cut-off (Hz)
HIGHCUT = 800.0         # Band-pass filter high cut-off (Hz)
FILTER_ORDER = 5         # Butterworth order
c = 343                  # Speed of sound in air (m s-1)

azimuth_range = np.arange(-180, 181, 4)  # −180°…180° in 4° steps
elevation_range = np.arange(0, 91, 4)    # 0°…90° in 4° steps

# Microphone geometry
mic_positions = microphone_positions_8_helicop()
CHANNELS = mic_positions.shape[0]

# -------------------------------------------------------------------------
# Pre-compute integer delay samples for every (az, el) pair
# -------------------------------------------------------------------------
precomputed_delays = np.empty(
    (len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32
)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(
            mic_positions, az, el, RATE, c
        )

# -------------------------------------------------------------------------
# Real-time plotting only (no recording)
# -------------------------------------------------------------------------
def process_audio_file(wav_filename: str) -> None:
    """
    Streams a multichannel WAV file block-by-block, applies band-pass
    filtering, beamforms each (az, el) point with pre-computed delays,
    and updates a colour-mapped energy plot live.
    """
    wf = wave.open(wav_filename, "rb")

    # Basic file-format sanity checks
    if wf.getnchannels() != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels; got {wf.getnchannels()}")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV sample width must be 16-bit.")
    if wf.getframerate() != RATE:
        raise ValueError(f"WAV sampling rate must be {RATE} Hz.")

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
    fig.colorbar(heatmap, ax=ax, label="Energy")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beamforming Energy Map (live)")

    # Marker for the maximum-energy direction
    max_marker, = ax.plot([], [], "ro", label="Max energy")
    ax.legend()
    ax.grid(True)

    while True:
        frames = wf.readframes(CHUNK)
        if len(frames) < CHUNK * CHANNELS * 2:  # EOF
            break

        audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS))
        filtered = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE,
                                         order=FILTER_ORDER)
        filtered /= filtered.max()

        # Energy map
        energy = np.zeros((len(azimuth_range), len(elevation_range)))
        for i in range(len(azimuth_range)):
            for j in range(len(elevation_range)):
                sig = apply_beamforming(filtered, precomputed_delays[i, j, :])
                energy[i, j] = np.sum(sig ** 2) / CHUNK

        # Update plot
        idx_max = np.unravel_index(np.argmax(energy), energy.shape)
        est_az, est_el = azimuth_range[idx_max[0]], elevation_range[idx_max[1]]

        heatmap.set_data(energy.T)
        heatmap.set_clim(vmin=energy.min(), vmax=energy.max())
        max_marker.set_data([est_az], [est_el])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)  # maintain real-time pacing

    plt.ioff()
    plt.show()
    wf.close()


# -------------------------------------------------------------------------
# Real-time plotting + MP4 export
# -------------------------------------------------------------------------
def process_audio_file_to_video(
    wav_filename: str,
    video_filename: str = "beamforming_video.mp4",
    fps: int = 10,
    dpi: int = 200,
) -> None:
    """
    Same processing pipeline as `process_audio_file`, but captures each
    rendered frame with `matplotlib.animation.FFMpegWriter` and writes
    an H.264-encoded MP4.

    Parameters
    ----------
    wav_filename : str
        Input multichannel WAV path.
    video_filename : str
        Output MP4 path.
    fps : int
        Frames per second for the video (≈ 1 / (CHUNK / RATE)).
    dpi : int
        Resolution of saved frames.
    """
    wf = wave.open(wav_filename, "rb")
    if wf.getnchannels() != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels; got {wf.getnchannels()}")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV sample width must be 16-bit.")
    if wf.getframerate() != RATE:
        raise ValueError(f"WAV sampling rate must be {RATE} Hz.")

    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="jet",
    )
    fig.colorbar(heatmap, ax=ax, label="Energy")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beamforming Energy Map (export)")

    max_marker, = ax.plot([], [], "ro", label="Max energy")
    ax.legend()
    ax.grid(True)

    meta = {
        "title": "Beamforming Energy Map",
        "artist": "SSL pipeline",
        "comment": "Delay-and-sum beamforming visualisation",
    }
    writer = FFMpegWriter(fps=fps, metadata=meta)

    with writer.saving(fig, video_filename, dpi=dpi):
        while True:
            frames = wf.readframes(CHUNK)
            if len(frames) < CHUNK * CHANNELS * 2:
                break

            audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS))
            filtered = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE,
                                             order=FILTER_ORDER)
            filtered /= filtered.max()

            energy = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    sig = apply_beamforming(filtered, precomputed_delays[i, j, :])
                    energy[i, j] = np.sum(sig ** 2)

            idx_max = np.unravel_index(np.argmax(energy), energy.shape)
            est_az, est_el = azimuth_range[idx_max[0]], elevation_range[idx_max[1]]

            heatmap.set_data(energy.T)
            heatmap.set_clim(vmin=energy.min(), vmax=energy.max())
            max_marker.set_data([est_az], [est_el])

            fig.canvas.draw()
            writer.grab_frame()
            time.sleep(0.1)

    wf.close()
    print(f"✅ Video exported → {video_filename}")


# -------------------------------------------------------------------------
# Script entry-point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    wav_files = ["heli_12052025.wav"]

    for wav in wav_files:
        print(f"Processing and exporting {wav} …")
        process_audio_file_to_video(
            wav_filename=wav,
            video_filename=f"{wav.rsplit('.', 1)[0]}_beamforming.mp4",
            fps=int(1 / 0.1),  # 10 fps ≈ real-time for 100 ms chunks
        )
