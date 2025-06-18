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

def mic_6_E_orange():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.16, 0.0]
    r = [0.13, 0.62]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_S_orange():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.16, 0.0]
    r = [0.13, 0.62]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_W_black():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.09, 0.0]
    r = [0.13, 0.5]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

def mic_6_N_black_thin():

    a = [0, -120, -240]

    # progressive distance configuration
    h = [1.09, 0.0]
    r = [0.11, 0.54]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], # mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 6
    ])
    return mic_positions

# -------------------------------------------------------------------------
# Global parameters
# -------------------------------------------------------------------------
RATE = 48_000            # Sampling rate (Hz)
CHUNK = int(0.1 * RATE)  # 100 ms of audio per processing block
LOWCUT = 180.0           # Band-pass filter low cut-off (Hz)
HIGHCUT = 2000.0          # Band-pass filter high cut-off (Hz)
FILTER_ORDER = 5         # Butterworth order
c = 343                  # Speed of sound in air (m/s)

azimuth_range = np.arange(-180, 181, 1)  # −180°…180° in 4° steps
elevation_range = np.arange(0, 91, 1)    # 0°…90° in 4° steps

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
# Real-time plotting + MP4 export with time indicator and segment processing
# -------------------------------------------------------------------------
def process_audio_segment_to_video(
    wav_filename: str,
    video_filename: str = "beamforming_video.mp4",
    fps: int = 10,
    dpi: int = 200,
    start_time: float = 0.0,
    end_time: float = None,
):


    wf = wave.open(wav_filename, "rb")
    if wf.getnchannels() != CHANNELS:
        raise ValueError(f"Expected {CHANNELS} channels; got {wf.getnchannels()}")
    if wf.getsampwidth() != 2:
        raise ValueError("WAV sample width must be 16-bit.")
    if wf.getframerate() != RATE:
        raise ValueError(f"WAV sampling rate must be {RATE} Hz.")

    total_frames = wf.getnframes()
    # Compute start and end frame indices
    start_frame = int(start_time * RATE)
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError("start_time out of range.")
    wf.setpos(start_frame)

    if end_time is not None:
        end_frame = int(end_time * RATE)
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError("end_time out of range or <= start_time.")
        max_blocks = (end_frame - start_frame) // CHUNK
    else:
        max_blocks = (total_frames - start_frame) // CHUNK

    # Prepare interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(
        np.zeros((len(azimuth_range), len(elevation_range))).T,
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="Blues_r",
    )
    cbar = fig.colorbar(heatmap, ax=ax, label="Energy")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_title("Beamforming Energy Map")

    # Marker for maximum-energy direction
    max_marker, = ax.plot([], [], "ro", label="Max energy")
    ax.legend()
    ax.grid(True)

    # Time indicator text (top-left corner)
    time_text = ax.text(
        0.02, 0.95, "Time: 0.00 s",
        transform=ax.transAxes,
        color="white", fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )

    # FFmpeg writer setup
    meta = {
        "title": "Beamforming Energy Map",
        "artist": "SSL pipeline",
        "comment": "Delay-and-sum beamforming visualisation",
    }
    writer = FFMpegWriter(fps=fps, metadata=meta)

    blocks_processed = 0
    with writer.saving(fig, video_filename, dpi=dpi):
        while blocks_processed < max_blocks:
            frames = wf.readframes(CHUNK)
            if len(frames) < CHUNK * CHANNELS * 2:
                break

            # Convert to NumPy array and normalize
            audio = np.frombuffer(frames, dtype=np.int16).reshape((-1, CHANNELS))
            filtered = apply_bandpass_filter(audio, LOWCUT, HIGHCUT, RATE,
                                             order=FILTER_ORDER)
            if filtered.max() != 0:
                filtered /= filtered.max()

            # Compute energy map
            energy = np.zeros((len(azimuth_range), len(elevation_range)))
            for i in range(len(azimuth_range)):
                for j in range(len(elevation_range)):
                    sig = apply_beamforming(filtered, precomputed_delays[i, j, :])
                    energy[i, j] = np.sum(sig ** 2) / CHUNK

            # Find global maximum
            idx_max = np.unravel_index(np.argmax(energy), energy.shape)
            est_az = azimuth_range[idx_max[0]]
            est_el = elevation_range[idx_max[1]]

            # Update heatmap and marker
            heatmap.set_data(energy.T)
            #heatmap.set_clim(vmin=energy.min(), vmax=energy.max())
            heatmap.set_clim(vmin=np.percentile(energy, 50), vmax=energy.max())

            max_marker.set_data([est_az], [est_el])

            # Compute current time (relative to WAV start)
            current_time = start_time + blocks_processed * (CHUNK / RATE)
            time_text.set_text(f"Time: {current_time:.2f} s")
            print(f"Time: {current_time:.2f} s")  # print to console

            # Render live plot
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Grab frame for video
            writer.grab_frame()

            blocks_processed += 1
            time.sleep(CHUNK / RATE)  # maintain real-time pacing

    wf.close()
    plt.ioff()
    plt.close(fig)
    print(f"✅ Video exported → {video_filename}")

# -------------------------------------------------------------------------
# Script entry-point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    wav_files = ["/Users/30068385/OneDrive - Western Sydney University/recordings/Helicop/11_06_25/Zoom2W.wav"]
    for wav in wav_files:
        process_audio_segment_to_video(
            wav_filename=wav,
            video_filename=f"{wav.rsplit('.', 1)[0]}_{LOWCUT}_{HIGHCUT}.mp4",
            fps=int(1 / 0.1),  # 10 fps ≈ real-time for 100 ms chunks
            start_time=4447573.0,
            end_time=4551521.0
        )
